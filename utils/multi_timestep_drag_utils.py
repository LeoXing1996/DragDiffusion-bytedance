import copy

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler

from .attn_utils import offload_masactrl, register_attention_editor_diffusers
from .debug_utils import draw_loss
from .drag_utils import (check_handle_reach_target, interpolate_feature_patch,
                         point_tracking)
from .restart_utils import restart_from_t


def update_at_one_step(model, init_code, text_emb, t, mask,
                       target_points, handle_points,
                       n_pix_step, args):
    # 2.1 run single denoising step without masactrl,
    #     and save F0 and x_prev_0 for drag
    with torch.no_grad():
        print('init: ', init_code.shape)
        unet_output, F0 = model.forward_unet_features(
            init_code,
            t,
            encoder_hidden_states=text_emb,
            layer_idx=args.unet_feature_idx,
            interp_res_h=args.sup_res_h,
            interp_res_w=args.sup_res_w)
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
        # with torch.autocast(device_type='cuda', dtype=torch.float16):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            unet_output, F1 = model.forward_unet_features(
                torch.cat([init_code], dim=0),
                t,
                encoder_hidden_states=text_emb,
                layer_idx=args.unet_feature_idx,
                interp_res_h=args.sup_res_h,
                interp_res_w=args.sup_res_w)
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

    save_suffix = f't{t}'
    draw_loss(loss_cache, dist_cache, restart_cache, args.save_dir,
              save_suffix=save_suffix)

    return init_code.detach(), reach_target, handle_points


def update_at_one_step_residual(model, init_code, text_emb, t, mask,
                                target_points, handle_points,
                                n_pix_step, args):
    # 2.1 run single denoising step without masactrl,
    #     and save F0 and x_prev_0 for drag
    with torch.no_grad():
        unet_output, F0 = model.forward_unet_features(
            init_code,
            t,
            encoder_hidden_states=text_emb,
            layer_idx=args.unet_feature_idx,
            interp_res_h=args.sup_res_h,
            interp_res_w=args.sup_res_w)
        x_prev_0, _ = model.step(unet_output, t, init_code)
        down_residuals_size_list, mid_residuals_size = \
            model.unet.forward(
                init_code,
                t,
                encoder_hidden_states=text_emb,
                return_residual_size=True)

    # prepare optimizable init_code and optimizer
    h, w = init_code.shape[-2:]
    down_residuals = [torch.zeros(*shape).cuda() for shape in down_residuals_size_list]
    mid_residuals = torch.zeros(*mid_residuals_size).cuda()

    for ten in down_residuals:
        ten.requires_grad_(True)
    mid_residuals.requires_grad_(True)

    # init_code.requires_grad_(True)
    optimizer = torch.optim.Adam([*down_residuals, mid_residuals], lr=0.1)

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
            try:
                unet_output, F1 = model.forward_unet_features(
                    torch.cat([init_code], dim=0),
                    t,
                    encoder_hidden_states=text_emb,
                    layer_idx=args.unet_feature_idx,
                    interp_res_h=args.sup_res_h,
                    interp_res_w=args.sup_res_w,
                    down_residuals=down_residuals,
                    mid_residuals=mid_residuals,
                )
            except Exception as exp:
                print(exp)
                import ipdb
                ipdb.set_trace()
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

    save_suffix = f't{t}'
    draw_loss(loss_cache, dist_cache, restart_cache, args.save_dir,
              save_suffix=save_suffix)

    return {'down': [ten.detach() for ten in down_residuals], 'mid': mid_residuals}, reach_target, handle_points


def no_tracking_update_at_one_step(model, init_code, text_emb, t, mask,
                                   target_points, handle_points, n_pix_step,
                                   args):
    scale = args.loss_scale
    # 2.1 run single denoising step without masactrl,
    #     and save F0 and x_prev_0 for drag
    with torch.no_grad():
        unet_output, F0 = model.forward_unet_features(
            init_code,
            t,
            encoder_hidden_states=text_emb,
            layer_idx=args.unet_feature_idx,
            interp_res_h=args.sup_res_h,
            interp_res_w=args.sup_res_w)
        x_prev_0, _ = model.step(unet_output, t, init_code)

    # prepare optimizable init_code and optimizer
    # init_code_orig = init_code.clone()
    init_code.requires_grad_(True)

    # optimizer = torch.optim.Adam([init_code], lr=args.lr)
    # prepare for point tracking and background regularization
    handle_points_init = copy.deepcopy(handle_points)
    interp_mask = F.interpolate(mask, (init_code.shape[2], init_code.shape[3]),
                                mode='nearest')

    init_code.requires_grad_(True)
    optimizer = torch.optim.Adam([init_code], lr=args.lr)

    # prepare amp scaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler()
    loss_cache = []  # save the loss value during drag
    dist_cache = []  # save the distance between src and tar
    restart_cache = []  # save whether do restart
    feature_cache = []  # save the feature *before* drag

    scheduler: DDIMScheduler = model.scheduler
    for step_idx in range(n_pix_step):
        with torch.autocast(device_type='cuda', dtype=torch.float16):

            unet_output, F1 = model.forward_unet_features(
                torch.cat([init_code], dim=0),
                t,
                encoder_hidden_states=text_emb,
                layer_idx=args.unet_feature_idx,
                interp_res_h=args.sup_res_h,
                interp_res_w=args.sup_res_w)
            x_prev_updated, _ = model.step(unet_output, t, init_code)

            loss = 0.0
            for i in range(len(handle_points)):
                pi, ti = handle_points[i], target_points[i]
                # skip if the distance between target and source is less than 1
                if (ti - pi).norm() < 2.:
                    continue

                di = (ti - pi) / (ti - pi).norm()

                # motion supervision
                f0_patch = F1[:, :,
                            int(pi[0]) - args.r_m: int(pi[0]) + args.r_m + 1,
                            int(pi[1]) - args.r_m:int(pi[1]) + args.r_m + 1].detach()
                f1_patch = interpolate_feature_patch(F1, pi[0] + di[0],
                                                        pi[1] + di[1], args.r_m)
                loss += ((2 * args.r_m + 1)**2) * F.l1_loss(f0_patch, f1_patch)

            # masked region must stay unchanged
            loss += args.lam * ((x_prev_updated - x_prev_0) *
                                (1.0 - interp_mask)).abs().sum()


        print('loss total=%f' % (loss.item()))
        loss_cache.append(loss.item())
        dist = (torch.stack(handle_points) - torch.stack(target_points)).norm().mean()
        dist_cache.append(dist)

        t_idx = torch.nonzero(scheduler.timesteps == t)[0][0]
        t_prev = scheduler.timesteps[t_idx + 1]
        alpha_prod_t = scheduler.alphas_cumprod[t_prev]
        beta_prod_t = 1 - alpha_prod_t
        scale_ = scale * (beta_prod_t ** 0.5)

        loss = loss * scale_
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    save_suffix = f't{t}'
    draw_loss(loss_cache, dist_cache, restart_cache, args.save_dir,
              save_suffix=save_suffix)
    # return gradient, False, handle_points
    return init_code.detach(), False, handle_points


def drag_diffusion_update_restart_multi(
        model, init_code, t, handle_points, target_points, mask, args, editor):

    max_restart_times = args.restart_times
    n_pix_step = args.n_pix_step

    assert len(handle_points) == len(target_points), \
        "number of handle point must equals target points"

    n_drag_steps = args.drag_steps

    text_emb = model.get_text_embeddings(args.prompt).detach()

    # 1. run the reference branch independently,
    #    save the orig init code for masactrl.
    init_code_orig_list, _, _, t_list = \
        run_naive_denoising_wo_inj(
            model, init_code, text_emb, t, n_drag_steps, args, False)

    init_code_updated = init_code.clone()  # copy the latent to be updated
    for re_counter in range(max_restart_times + 1):
        # 2. run drag without masactrl.
        for idx in range(n_drag_steps):
            t = t_list[idx]
            init_code_orig = init_code_orig_list[idx]

            # 2. run drag without masactrl.
            if args.wo_pt:
                init_code_updated, reach_target, handle_points = no_tracking_update_at_one_step(
                    model, init_code_updated, text_emb, t, mask,
                    target_points, handle_points, n_pix_step,
                    args)

            elif args.update_residual:
                residual, reach_target, handle_points = update_at_one_step_residual(
                    model, init_code_updated, text_emb, t, mask,
                    target_points, handle_points, n_pix_step,
                    args)

            else:
                init_code_updated, reach_target, handle_points = update_at_one_step(
                    model, init_code_updated, text_emb, t, mask,
                    target_points, handle_points, n_pix_step,
                    args)

            # apply diffedit to the updated latent code
            with torch.no_grad():
                if args.diffedit:
                    interp_mask = F.interpolate(
                        mask, (init_code_updated.shape[2], init_code_updated.shape[3]),
                        mode='nearest')
                    init_code_updated = init_code_updated * interp_mask + \
                        init_code_orig * (1 - interp_mask)

            if not reach_target and idx < n_drag_steps - 1:
                # 3. switch to the next step with MasaCtrl
                with torch.no_grad():

                    register_attention_editor_diffusers(model, editor, 'lora_attn_proc')

                    if not args.update_residual:
                        unet_output, _ = model.forward_unet_features(
                            torch.cat([init_code_orig, init_code_updated], dim=0),
                            t,
                            encoder_hidden_states=torch.cat([text_emb, text_emb], dim=0),
                            layer_idx=args.unet_feature_idx,
                            interp_res_h=args.sup_res_h,
                            interp_res_w=args.sup_res_w)
                    else:
                        down_residual = residual['down']
                        mid_residual = residual['mid']

                        down_residual_masactrl = [
                            torch.cat([torch.zeros_like(ten), ten], dim=0) for ten in down_residual
                        ]
                        mid_residual_masactrl = torch.cat((torch.zeros_like(mid_residual), mid_residual), dim=0)

                        unet_output, _ = model.forward_unet_features(
                            torch.cat([init_code_orig, init_code_updated], dim=0),
                            t,
                            encoder_hidden_states=torch.cat([text_emb, text_emb], dim=0),
                            layer_idx=args.unet_feature_idx,
                            interp_res_h=args.sup_res_h,
                            interp_res_w=args.sup_res_w,
                            down_residual=down_residual_masactrl,
                            mid_residual=mid_residual_masactrl)
                        pass
                    unet_output = unet_output[1:2]
                    init_code_updated, _ = model.step(unet_output, t, init_code_updated)
                    offload_masactrl(model)

            else:
                break

        if reach_target:
            break
        # 4. do restart
        if re_counter < max_restart_times:
            init_code_updated = restart_from_t(
                init_code_updated, t_list[idx], t_list[0], model, text_emb, args)
            # reset idx to 0 manually, since we restarted
            idx = 0

    if args.restart_fix:
        print('Run Restart Fix.')
        init_code_updated = restart_from_t(
            init_code_updated, t_list[idx], t_list[0], model, text_emb, args)
        idx = 0

    if args.update_residual:
        # move to the next timestep since outer denoising function do not support residual as input
        down_residual = residual['down']
        mid_residual = residual['mid']

        down_residual_masactrl = [
            torch.cat([torch.zeros_like(ten), ten], dim=0) for ten in down_residual
        ]
        mid_residual_masactrl = torch.cat((torch.zeros_like(mid_residual), mid_residual), dim=0)

        unet_output, _ = model.forward_unet_features(
            torch.cat([init_code_orig, init_code_updated], dim=0),
            t,
            encoder_hidden_states=torch.cat([text_emb, text_emb], dim=0),
            layer_idx=args.unet_feature_idx,
            interp_res_h=args.sup_res_h,
            interp_res_w=args.sup_res_w,
            down_residuals=down_residual_masactrl,
            mid_residuals=mid_residual_masactrl)
        unet_output = unet_output[1:2]
        init_code_updated, pred_x0 = model.step(unet_output, t_list[idx], init_code_updated)
        from PIL import Image
        import os.path as osp
        Image.fromarray(model.latent2image(pred_x0)).save(osp.join(args.save_dir, 'pred_x0.png'))
        idx = 1
        offload_masactrl(model)

    return init_code_updated, idx


def run_naive_denoising_wo_inj(model, init_code, text_emb,
                               t_start, n_drag_steps, args,
                               need_grad, editor=None, init_code_orig=None):
    scheduler = model.scheduler

    context_manager = torch.enable_grad if need_grad else torch.no_grad
    if need_grad:
        assert init_code.requires_grad, 'init_code must require grad'

    t_idx = torch.nonzero(scheduler.timesteps == t_start)[0][0]
    t_idx_list = [scheduler.timesteps[t_idx + idx] for idx in range(n_drag_steps)]

    F0_list, x_prev_0_list = [], []
    init_code_orig_list = []

    if editor is not None:
        if init_code_orig is None:
            print('init_code_orig is None but editor is passed.')
            print('Use init_code instead!')
            init_code_orig = init_code.clone()
        register_attention_editor_diffusers(model, editor, 'lora_attn_proc')
        print('Add MasaCtrl!')
        input_embeddings = torch.cat([text_emb] * 2, dim=0)
        input_latent = torch.cat([init_code_orig, init_code], dim=0)
    else:
        input_embeddings = text_emb
        input_latent = init_code

    with context_manager():
        for t in t_idx_list:

            unet_output, F0 = model.forward_unet_features(
                input_latent,
                t,
                encoder_hidden_states=input_embeddings,
                layer_idx=args.unet_feature_idx,
                interp_res_h=args.sup_res_h,
                interp_res_w=args.sup_res_w)
            x_prev_0, _ = model.step(unet_output, t, input_latent)

            if editor is None:
                F0_list.append(F0.clone())
                x_prev_0_list.append(x_prev_0.clone())
                init_code_orig_list.append(input_latent.clone())
            else:
                F0_list.append(F0[1:2].clone())
                x_prev_0_list.append(x_prev_0[1:2].clone())
                init_code_orig_list.append(input_latent[0:1].clone())

            input_latent = x_prev_0

    if editor is not None:
        offload_masactrl(model, True)
        print('Offload MasaCtrl!')

    return init_code_orig_list, F0_list, x_prev_0_list, t_idx_list
