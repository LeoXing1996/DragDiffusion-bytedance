# *************************************************************************
# Copyright (2023) Bytedance Inc.
#
# Copyright (2023) DragDiffusion Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *************************************************************************

import copy
import torch
import torch.nn.functional as F
from .debug_utils import draw_loss


def cosine_guidance(feat1, feat2, scale=1, alpha=1, beta=1):

    bz, nch = feat1.shape[0], feat1.shape[1]

    # NOTE: reshape for feat1
    feat1 = feat1.reshape(bz, nch, -1)
    feat2 = feat2.view(bz, nch, -1)
    if feat1.shape[2] > feat2.shape[2]:
        feat1 = feat1[:, :, :feat2.shape[2]]
    if feat2.shape[2] > feat1.shape[2]:
        feat2 = feat2[:, :, :feat1.shape[2]]

    guidance = (F.cosine_similarity(feat1, feat2) + 1) / 2

    return scale / (guidance.mean() * beta + alpha)


def point_tracking(F0, F1, handle_points, handle_points_init, args):
    with torch.no_grad():
        for i in range(len(handle_points)):
            pi0, pi = handle_points_init[i], handle_points[i]
            f0 = F0[:, :, int(pi0[0]), int(pi0[1])]

            r1, r2 = int(pi[0]) - args.r_p, int(pi[0]) + args.r_p + 1
            c1, c2 = int(pi[1]) - args.r_p, int(pi[1]) + args.r_p + 1
            F1_neighbor = F1[:, :, r1:r2, c1:c2]
            all_dist = (f0.unsqueeze(dim=-1).unsqueeze(dim=-1) -
                        F1_neighbor).abs().sum(dim=1)
            all_dist = all_dist.squeeze(dim=0)
            # WARNING: no boundary protection right now
            row, col = divmod(all_dist.argmin().item(), all_dist.shape[-1])
            handle_points[i][0] = pi[0] - args.r_p + row
            handle_points[i][1] = pi[1] - args.r_p + col
        return handle_points


def check_handle_reach_target(handle_points, target_points):
    # dist = (torch.cat(handle_points,dim=0) - torch.cat(target_points,dim=0)).norm(dim=-1)
    all_dist = list(
        map(lambda p, q: (p - q).norm(), handle_points, target_points))
    return (torch.tensor(all_dist) < 2.0).all()


# obtain the bilinear interpolated feature patch centered around (x, y) with radius r
def interpolate_feature_patch(feat, y, x, r):
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())

    Ia = feat[:, :, y0 - r:y0 + r + 1, x0 - r:x0 + r + 1]
    Ib = feat[:, :, y1 - r:y1 + r + 1, x0 - r:x0 + r + 1]
    Ic = feat[:, :, y0 - r:y0 + r + 1, x1 - r:x1 + r + 1]
    Id = feat[:, :, y1 - r:y1 + r + 1, x1 - r:x1 + r + 1]

    return Ia * wa + Ib * wb + Ic * wc + Id * wd


def drag_diffusion_update(model, init_code, t, handle_points, target_points,
                          mask, args):

    assert len(handle_points) == len(target_points), \
        "number of handle point must equals target points"

    text_emb = model.get_text_embeddings(args.prompt).detach()
    # the init output feature of unet
    with torch.no_grad():
        unet_output, F0 = model.forward_unet_features(
            init_code,
            t,
            encoder_hidden_states=text_emb,
            layer_idx=args.unet_feature_idx,
            interp_res_h=args.sup_res_h,
            interp_res_w=args.sup_res_w)
        x_prev_0, _ = model.step(unet_output, t, init_code)
        # init_code_orig = copy.deepcopy(init_code)

    # prepare optimizable init_code and optimizer
    init_code.requires_grad_(True)
    optimizer = torch.optim.Adam([init_code], lr=args.lr)

    # prepare for point tracking and background regularization
    handle_points_init = copy.deepcopy(handle_points)
    interp_mask = F.interpolate(mask, (init_code.shape[2], init_code.shape[3]),
                                mode='nearest')

    # prepare amp scaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler()
    for step_idx in range(args.n_pix_step):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            unet_output, F1 = model.forward_unet_features(
                init_code,
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

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return init_code


def drag_diffusion_update_restart(
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

    text_emb = model.get_text_embeddings(args.prompt).detach()
    # the init output feature of unet
    with torch.no_grad():
        unet_output, F0 = model.forward_unet_features(
            init_code,
            t,
            encoder_hidden_states=text_emb,
            layer_idx=args.unet_feature_idx,
            interp_res_h=args.sup_res_h,
            interp_res_w=args.sup_res_w)
        x_prev_0, _ = model.step(unet_output, t, init_code)
        # init_code_orig = copy.deepcopy(init_code)

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

    for step_idx in range(n_pix_step):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            unet_output, F1 = model.forward_unet_features(
                init_code,
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
                if args.loss_type == 'cosine':
                    loss = cosine_guidance(f0_patch, f1_patch, scale=args.loss_scale)
                elif args.loss_type == 'l1':
                    loss += ((2 * args.r_m + 1)**2) * F.l1_loss(f0_patch, f1_patch)

            # masked region must stay unchanged
            loss += args.lam * ((x_prev_updated - x_prev_0) *
                                (1.0 - interp_mask)).abs().sum()
            # loss += args.lam * ((init_code_orig-init_code)*(1.0-interp_mask)).abs().sum()
            print('loss total=%f' % (loss.item()))
            loss_cache.append(loss.item())

            dist = (torch.stack(handle_points) - torch.stack(target_points)).norm().mean()
            dist_cache.append(dist)
            # dist_cache.append()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if (step_idx + 1) % restart_interval == 0:
            from utils.restart_utils import restart

            with torch.no_grad():
                unet_output, F1 = model.forward_unet_features(
                    init_code,
                    t,
                    encoder_hidden_states=text_emb,
                    layer_idx=args.unet_feature_idx,
                    interp_res_h=args.sup_res_h,
                    interp_res_w=args.sup_res_w)
                x_prev_updated, _ = model.step(unet_output, t, init_code)
                del init_code, optimizer
                init_code = restart(x_prev_updated, t, model, args)
                init_code = init_code.requires_grad_(True)
                optimizer = torch.optim.Adam([init_code], lr=args.lr)
            restart_cache.append(1)
        else:
            restart_cache.append(0)
    draw_loss(loss_cache, dist_cache, restart_cache, args.save_dir)
    return init_code


def drag_diffusion_update_gen(model, init_code, t, handle_points,
                              target_points, mask, args):

    assert len(handle_points) == len(target_points), \
        "number of handle point must equals target points"

    # positive prompt embedding
    text_emb = model.get_text_embeddings(args.prompt).detach()
    if args.guidance_scale > 1.0:
        unconditional_input = model.tokenizer([args.neg_prompt],
                                              padding="max_length",
                                              max_length=77,
                                              return_tensors="pt")
        unconditional_emb = model.text_encoder(
            unconditional_input.input_ids.to(text_emb.device))[0].detach()
        text_emb = torch.cat([unconditional_emb, text_emb], dim=0)

    # the init output feature of unet
    with torch.no_grad():
        if args.guidance_scale > 1.:
            model_inputs_0 = copy.deepcopy(torch.cat([init_code] * 2))
        else:
            model_inputs_0 = copy.deepcopy(init_code)
        unet_output, F0 = model.forward_unet_features(
            model_inputs_0,
            t,
            encoder_hidden_states=text_emb,
            layer_idx=args.unet_feature_idx,
            interp_res_h=args.sup_res_h,
            interp_res_w=args.sup_res_w)
        if args.guidance_scale > 1.:
            # strategy 1: discard the unconditional branch feature maps
            # F0 = F0[1].unsqueeze(dim=0)
            # strategy 2: concat pos and neg branch feature maps for motion-sup and point tracking
            # F0 = torch.cat([F0[0], F0[1]], dim=0).unsqueeze(dim=0)
            # strategy 3: concat pos and neg branch feature maps with guidance_scale consideration
            coef = args.guidance_scale / (2 * args.guidance_scale - 1.0)
            F0 = torch.cat([(1 - coef) * F0[0], coef * F0[1]],
                           dim=0).unsqueeze(dim=0)

            unet_output_uncon, unet_output_con = unet_output.chunk(2, dim=0)
            unet_output = unet_output_uncon + args.guidance_scale * (
                unet_output_con - unet_output_uncon)
        x_prev_0, _ = model.step(unet_output, t, init_code)
        # init_code_orig = copy.deepcopy(init_code)

    # prepare optimizable init_code and optimizer
    init_code.requires_grad_(True)
    optimizer = torch.optim.Adam([init_code], lr=args.lr)

    # prepare for point tracking and background regularization
    handle_points_init = copy.deepcopy(handle_points)
    interp_mask = F.interpolate(mask, (init_code.shape[2], init_code.shape[3]),
                                mode='nearest')

    # prepare amp scaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler()
    for step_idx in range(args.n_pix_step):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            if args.guidance_scale > 1.:
                model_inputs = init_code.repeat(2, 1, 1, 1)
            else:
                model_inputs = init_code
            unet_output, F1 = model.forward_unet_features(
                model_inputs,
                t,
                encoder_hidden_states=text_emb,
                layer_idx=args.unet_feature_idx,
                interp_res_h=args.sup_res_h,
                interp_res_w=args.sup_res_w)
            if args.guidance_scale > 1.:
                # strategy 1: discard the unconditional branch feature maps
                # F1 = F1[1].unsqueeze(dim=0)
                # strategy 2: concat positive and negative branch feature maps for motion-sup and point tracking
                # F1 = torch.cat([F1[0], F1[1]], dim=0).unsqueeze(dim=0)
                # strategy 3: concat pos and neg branch feature maps with guidance_scale consideration
                coef = args.guidance_scale / (2 * args.guidance_scale - 1.0)
                F1 = torch.cat([(1 - coef) * F1[0], coef * F1[1]],
                               dim=0).unsqueeze(dim=0)

                unet_output_uncon, unet_output_con = unet_output.chunk(2,
                                                                       dim=0)
                unet_output = unet_output_uncon + args.guidance_scale * (
                    unet_output_con - unet_output_uncon)
            x_prev_updated, _ = model.step(unet_output, t, init_code)

            # do point tracking to update handle points before computing motion supervision loss
            if step_idx != 0:
                handle_points = point_tracking(F0, F1, handle_points,
                                               handle_points_init, args)
                print('new handle points', handle_points)

            # break if all handle points have reached the targets
            if check_handle_reach_target(handle_points, target_points):
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
            # loss += args.lam * ((init_code_orig - init_code)*(1.0-interp_mask)).abs().sum()
            print('loss total=%f' % (loss.item()))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return init_code
