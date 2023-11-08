import copy

import torch
import torch.nn.functional as F

from .attn_utils import offload_masactrl, register_attention_editor_diffusers
from .debug_utils import draw_loss
from .drag_utils import (check_handle_reach_target, interpolate_feature_patch,
                         point_tracking)
from .restart_utils import restart_from_t


def update_at_one_step(model, init_code, text_emb, t, mask,
                       target_points, handle_points,
                       n_pix_step, restart_interval, args):
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
            # unet_output, F1 = model.forward_unet_features(
            #     torch.cat([init_code_orig, init_code], dim=0),
            #     t,
            #     encoder_hidden_states=torch.cat([text_emb, text_emb], dim=0),
            #     layer_idx=args.unet_feature_idx,
            #     interp_res_h=args.sup_res_h,
            #     interp_res_w=args.sup_res_w)
            # unet_output = unet_output[1:2]
            # F1 = F1[1:2]
            # x_prev_updated, _ = model.step(unet_output, t, init_code)
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

        # no restart in single update
        # if (step_idx + 1) % restart_interval == 0 and step_idx != n_pix_step - 1 and args.restart_strategy is not None:
        #     from utils.restart_utils import restart

        #     with torch.no_grad():
        #         unet_output, F1 = model.forward_unet_features(
        #             torch.cat([init_code_orig, init_code], dim=0),
        #             t,
        #             encoder_hidden_states=torch.cat([text_emb, text_emb], dim=0),
        #             layer_idx=args.unet_feature_idx,
        #             interp_res_h=args.sup_res_h,
        #             interp_res_w=args.sup_res_w)
        #         unet_output = unet_output[1:2]
        #         x_prev_updated, _ = model.step(unet_output, t, init_code)
        #         del init_code, optimizer
        #         init_code = restart(x_prev_updated, t, model, args)
        #         init_code = init_code.requires_grad_(True)
        #         optimizer = torch.optim.Adam([init_code], lr=args.lr)
        #     restart_cache.append(1)
        # else:
        #     restart_cache.append(0)
    save_suffix = f't{t}'
    draw_loss(loss_cache, dist_cache, restart_cache, args.save_dir,
              save_suffix=save_suffix)

    return init_code.detach(), reach_target, handle_points


def drag_diffusion_update_restart_multi(
        model, init_code, t, handle_points, target_points, mask, args, editor):

    restart_interval = args.restart_interval
    max_restart_times = args.restart_times
    n_pix_step = args.n_pix_step

    # assert restart_interval < args.n_pix_step, (
    #     f'"restart_interval" ({restart_interval}) must be less than '
    #     f'"n_pix_step" ({n_pix_step})')
    # assert n_pix_step % restart_interval == 0, (
    #     f'"n_pix_step" ({n_pix_step} must be divided by '
    #     f'"restart_interval" ({restart_interval})')

    assert len(handle_points) == len(target_points), \
        "number of handle point must equals target points"

    n_drag_steps = args.drag_steps

    text_emb = model.get_text_embeddings(args.prompt).detach()
    # scheduler = model.scheduler
    # t_idx = torch.nonzero(scheduler.timesteps == t)[0][0]

    # init_code_orig_list, F0_list, x_prev_0_list, t_idx_list = \
    #     run_naive_denoising(
    #         model, init_code, text_emb, t, n_drag_steps, args, False)

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
            init_code_updated, reach_target, handle_points = update_at_one_step(
                model, init_code_updated, text_emb, t, mask,
                target_points, handle_points, n_pix_step,
                restart_interval, args)

            # apply diffedit to the updated latent code
            with torch.no_grad():
                if args.diffedit:
                    interp_mask = F.interpolate(
                        mask, (init_code_updated.shape[2], init_code_updated.shape[3]),
                        mode='nearest')
                    init_code_updated = init_code_updated * (
                        1 - interp_mask) + init_code_orig * interp_mask

            if not reach_target and idx < n_drag_steps - 1:
                # 3. switch to the next step with MasaCtrl
                with torch.no_grad():

                    register_attention_editor_diffusers(model, editor, 'lora_attn_proc')

                    unet_output, _ = model.forward_unet_features(
                        torch.cat([init_code_orig, init_code_updated], dim=0),
                        t,
                        encoder_hidden_states=torch.cat([text_emb, text_emb], dim=0),
                        layer_idx=args.unet_feature_idx,
                        interp_res_h=args.sup_res_h,
                        interp_res_w=args.sup_res_w)
                    unet_output = unet_output[1:2]
                    init_code_updated, _ = model.step(unet_output, t, init_code_updated)
                    offload_masactrl(model)

            else:
                break

        if reach_target:
            break
        # 4. do restart
        if re_counter < max_restart_times or args.restart_fix:
            init_code_updated = restart_from_t(
                init_code_updated, t_list[-1], t_list[0], model, text_emb, args)
            # reset idx to 0 manually, since we restarted
            idx = 0

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


# def drag_diffusion_update_restart_multi_in_one(
#         model, init_code, t, handle_points, target_points, mask, args):

#     restart_interval = args.restart_interval
#     n_pix_step = args.n_pix_step

#     assert restart_interval < args.n_pix_step, (
#         f'"restart_interval" ({restart_interval}) must be less than '
#         f'"n_pix_step" ({n_pix_step})')
#     assert n_pix_step % restart_interval == 0, (
#         f'"n_pix_step" ({n_pix_step} must be divided by '
#         f'"restart_interval" ({restart_interval})')

#     assert len(handle_points) == len(target_points), \
#         "number of handle point must equals target points"

#     n_drag_steps = args.drag_steps

#     text_emb = model.get_text_embeddings(args.prompt).detach()
#     # scheduler = model.scheduler
#     # t_idx = torch.nonzero(scheduler.timesteps == t)[0][0]

#     init_code_orig_list, F0_list, x_prev_0_list, t_idx_list = \
#         run_naive_denoising(
#             model, init_code, text_emb, t, n_drag_steps, args, False)

#     # TODO: here we have a bug, we should use the correct init code!!!!!
#     for idx in range(n_drag_steps):
#         # t = scheduler.timesteps[t_idx + idx]
#         t = t_idx_list[idx]
#         init_code_orig = init_code_orig_list[idx]
#         F0 = F0_list[idx]
#         x_prev_0 = x_prev_0_list[idx]

#         # init_code_orig = init_code.clone()
#         init_code, reach_target, handle_points = update_at_one_step(
#             model, init_code, F0, x_prev_0, text_emb, t, mask,
#             target_points, handle_points, n_pix_step,
#             restart_interval, args)

#         if not reach_target and idx < n_drag_steps - 1:
#             with torch.no_grad():
#                 unet_output, F1 = model.forward_unet_features(
#                     torch.cat([init_code_orig, init_code], dim=0),
#                     t,
#                     encoder_hidden_states=torch.cat([text_emb, text_emb], dim=0),
#                     layer_idx=args.unet_feature_idx,
#                     interp_res_h=args.sup_res_h,
#                     interp_res_w=args.sup_res_w)
#                 init_code, _ = model.step(unet_output, t, init_code)
#                 init_code = init_code[1:2]

#         else:
#             break

#     return init_code, idx
