import copy

import torch
import torch.nn.functional as F

from .debug_utils import draw_loss
from .drag_utils import (check_handle_reach_target, interpolate_feature_patch,
                         point_tracking)


def update_at_one_step(model, init_code, text_emb, t, mask,
                       target_points, handle_points,
                       n_pix_step, restart_interval, args):

    init_code_orig = init_code.clone()
    with torch.no_grad():
        # unet_output, F0 = model.forward_unet_features(
        #     init_code,
        #     t,
        #     encoder_hidden_states=text_emb,
        #     layer_idx=args.unet_feature_idx,
        #     interp_res_h=args.sup_res_h,
        #     interp_res_w=args.sup_res_w)
        unet_output, F0 = model.forward_unet_features(
            torch.cat([init_code_orig, init_code], dim=0),
            t,
            encoder_hidden_states=torch.cat([text_emb, text_emb], dim=0),
            layer_idx=args.unet_feature_idx,
            interp_res_h=args.sup_res_h,
            interp_res_w=args.sup_res_w)
        unet_output = unet_output[1:2] # fetch the second output
        F0 = F0[1:2]
        x_prev_0, _ = model.step(unet_output, t, init_code)

    # prepare optimizable init_code and optimizer
    init_code.requires_grad_(True)
    optimizer = torch.optim.Adam([init_code], lr=args.lr)

    # prepare for point tracking and background regularization
    handle_points_init = copy.deepcopy(handle_points)
    interp_mask = F.interpolate(mask, (init_code.shape[2], init_code.shape[3]),
                                mode='nearest')

    # prepare amp scaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler()
    loss_cache = []  # save the loss value during drag
    dist_cache = []  # save the distance between src and tar
    restart_cache = []  # save whether do restart
    feature_cache = []  # save the feature *before* drag

    reach_target = False
    for step_idx in range(n_pix_step):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            unet_output, F1 = model.forward_unet_features(
                torch.cat([init_code_orig, init_code], dim=0),
                t,
                encoder_hidden_states=torch.cat([text_emb, text_emb], dim=0),
                layer_idx=args.unet_feature_idx,
                interp_res_h=args.sup_res_h,
                interp_res_w=args.sup_res_w)
            unet_output = unet_output[1:2]
            F1 = F1[1:2]
            x_prev_updated, _ = model.step(unet_output, t, init_code)

            # do point tracking to update handle points before computing motion supervision loss
            if step_idx != 0:
                handle_points = point_tracking(F0, F1, handle_points,
                                               handle_points_init, args)
                print('new handle points', handle_points)

            # break if all handle points have reached the targets
            if check_handle_reach_target(handle_points, target_points):
                reach_target = True
                break

            loss = 0.0
            for i in range(len(handle_points)):
                pi, ti = handle_points[i], target_points[i]
                # skip if the distance between target and source is less than 1
                if (ti - pi).norm() < 2.:
                    continue

                di = (ti - pi) / (ti - pi).norm()

                # motion supervision
                f0_patch = F1[:, :,
                              int(pi[0]) - args.r_m:int(pi[0]) + args.r_m + 1,
                              int(pi[1]) - args.r_m:int(pi[1]) + args.r_m +
                              1].detach()
                f1_patch = interpolate_feature_patch(F1, pi[0] + di[0],
                                                     pi[1] + di[1], args.r_m)
                loss += ((2 * args.r_m + 1)**2) * F.l1_loss(f0_patch, f1_patch)

            # masked region must stay unchanged
            loss += args.lam * ((x_prev_updated - x_prev_0) *
                                (1.0 - interp_mask)).abs().sum()
            # loss += args.lam * ((init_code_orig-init_code)*(1.0-interp_mask)).abs().sum()
            print('loss total=%f' % (loss.item()))
            loss_cache.append(loss.item())
            dist = (torch.stack(handle_points) - torch.stack(target_points)).norm().mean()
            dist_cache.append(dist)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if (step_idx + 1) % restart_interval == 0 and args.restart_strategy is not None:
            from utils.restart_utils import restart

            with torch.no_grad():
                unet_output, F1 = model.forward_unet_features(
                    torch.cat([init_code_orig, init_code], dim=0),
                    t,
                    encoder_hidden_states=torch.cat([text_emb, text_emb], dim=0),
                    layer_idx=args.unet_feature_idx,
                    interp_res_h=args.sup_res_h,
                    interp_res_w=args.sup_res_w)
                unet_output = unet_output[1:2]
                x_prev_updated, _ = model.step(unet_output, t, init_code)
                del init_code, optimizer
                init_code = restart(x_prev_updated, t, model, args)
                init_code = init_code.requires_grad_(True)
                optimizer = torch.optim.Adam([init_code], lr=args.lr)
            restart_cache.append(1)
        else:
            restart_cache.append(0)
    save_suffix = f't{t}'
    draw_loss(loss_cache, dist_cache, restart_cache, args.save_dir,
              save_suffix=save_suffix)

    return init_code, reach_target, handle_points


def drag_diffusion_update_restart_multi(
    model, init_code, t, handle_points, target_points, mask, args):

    restart_interval = args.restart_interval
    n_pix_step = args.n_pix_step

    assert restart_interval < args.n_pix_step, (
        f'"restart_interval" ({restart_interval}) must be less than '
        f'"n_pix_step" ({n_pix_step})')
    assert n_pix_step % restart_interval == 0, (
        f'"n_pix_step" ({n_pix_step} must be divided by '
        f'"restart_interval" ({restart_interval})')

    assert len(handle_points) == len(target_points), \
        "number of handle point must equals target points"

    n_drag_steps = args.drag_steps

    text_emb = model.get_text_embeddings(args.prompt).detach()
    scheduler = model.scheduler
    t_idx = torch.nonzero(scheduler.timesteps == t)[0][0]

    for idx in range(n_drag_steps):
        t = scheduler.timesteps[t_idx + idx]

        init_code_orig = init_code.clone()
        init_code, reach_target, handle_points = update_at_one_step(
            model, init_code, text_emb, t, mask,
            target_points, handle_points, n_pix_step,
            restart_interval, args)

        if not reach_target and idx < n_drag_steps - 1:
            with torch.no_grad():
                unet_output, F1 = model.forward_unet_features(
                    torch.cat([init_code_orig, init_code], dim=0),
                    t,
                    encoder_hidden_states=torch.cat([text_emb, text_emb], dim=0),
                    layer_idx=args.unet_feature_idx,
                    interp_res_h=args.sup_res_h,
                    interp_res_w=args.sup_res_w)
                init_code, _ = model.step(unet_output, t, init_code)
                init_code = init_code[1:2]

        else:
            break

    return init_code, idx
