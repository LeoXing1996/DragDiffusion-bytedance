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

import datetime
# run results of DragDiffusion
import os
import pickle
import sys
from argparse import ArgumentParser
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, DPMSolverMultistepScheduler
from einops import rearrange
from PIL import Image
from pytorch_lightning import seed_everything
from torchvision.utils import save_image

sys.path.insert(0, '../')  # noqa

from drag_pipeline import DragPipeline
from utils.attn_utils import (MutualSelfAttentionControl,
                              register_attention_editor_diffusers)
from utils.drag_utils import drag_diffusion_update
from utils.multi_timestep_drag_utils import drag_diffusion_update_restart_multi


def registry_masactrl(model, editor, lora_path):

    if lora_path == "":
        register_attention_editor_diffusers(model,
                                            editor,
                                            attn_processor='attn_proc')
    else:
        register_attention_editor_diffusers(model,
                                            editor,
                                            attn_processor='lora_attn_proc')

    return model


def preprocess_image(image, device):
    image = torch.from_numpy(image).float() / 127.5 - 1  # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--quick-run', action='store_true')

    parser.add_argument('--cate', type=str)
    parser.add_argument('--work-dir', type=str)

    parser.add_argument('--restart-strategy', type=str)
    parser.add_argument('--restart-interval', type=int, default=20)
    parser.add_argument('--restart-with-masactrl', action='store_true')
    parser.add_argument('--restart-times', type=int, default=0)
    parser.add_argument('--restart-fix', action='store_true')

    parser.add_argument('--wo-pt', action='store_true')

    parser.add_argument('--inversion-strength', type=float, default=0.7)
    parser.add_argument('--drag-steps', type=int, default=1)
    parser.add_argument('--n-pix-step', type=int, default=80)
    parser.add_argument('--diffedit', action='store_true')

    parser.add_argument('--loss-type', type=str, default='l1')
    parser.add_argument('--loss-scale', type=float, default=10.0)
    return parser.parse_args()


# copy the run_drag function to here
def run_drag(
    source_image,
    # image_with_clicks,
    mask,
    prompt,
    points,
    inversion_strength,
    lam,
    latent_lr,
    n_pix_step,
    model_path,
    vae_path,
    lora_path,
    start_step,
    start_layer,
    # save_dir="./results"
    user_args,
    save_dir,
):
    # initialize model
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    # if user_args.restart_strategy:
    #     scheduler = DPMSolverMultistepScheduler(
    #         beta_start=0.00085,
    #         beta_end=0.012,
    #         beta_schedule="scaled_linear",
    #         clip_sample=False,
    #         set_alpha_to_one=False,
    #         use_karras_sigmas=True,
    #         steps_offset=1)
    # else:
    #     scheduler = DDIMScheduler(beta_start=0.00085,
    #                               beta_end=0.012,
    #                               beta_schedule="scaled_linear",
    #                               clip_sample=False,
    #                               set_alpha_to_one=False,
    #                               steps_offset=1)
    scheduler = DDIMScheduler(beta_start=0.00085,
                              beta_end=0.012,
                              beta_schedule="scaled_linear",
                              clip_sample=False,
                              set_alpha_to_one=False,
                              steps_offset=1)
    model = DragPipeline.from_pretrained(model_path,
                                         scheduler=scheduler).to(device)
    # call this function to override unet forward function,
    # so that intermediate features are returned after forward
    model.modify_unet_forward()

    # set vae
    if vae_path != "default":
        model.vae = AutoencoderKL.from_pretrained(vae_path).to(
            model.vae.device, model.vae.dtype)

    # initialize parameters
    seed = 42  # random seed used by a lot of people for unknown reason
    seed_everything(seed)

    args = SimpleNamespace()
    args.prompt = prompt
    args.points = points
    args.n_inference_step = 50
    args.n_actual_inference_step = round(inversion_strength *
                                         args.n_inference_step)
    args.guidance_scale = 1.0

    args.unet_feature_idx = [3]

    args.r_m = 1
    args.r_p = 3
    args.lam = lam

    args.lr = latent_lr
    args.n_pix_step = n_pix_step

    full_h, full_w = source_image.shape[:2]
    args.sup_res_h = int(0.5 * full_h)
    args.sup_res_w = int(0.5 * full_w)

    args.save_dir = save_dir

    # add restart-related args to namespace
    args.restart_strategy = user_args.restart_strategy
    args.restart_interval = user_args.restart_interval
    args.restart_times = user_args.restart_times
    args.restart_fix = user_args.restart_fix
    args.restart_with_masactrl = user_args.restart_with_masactrl
    args.wo_pt = user_args.wo_pt
    args.diffedit = user_args.diffedit
    args.drag_steps = user_args.drag_steps

    # add consine loss func args
    args.loss_type = user_args.loss_type
    args.loss_scale = user_args.loss_scale
    args.baseline = user_args.baseline

    print(args)

    source_image = preprocess_image(source_image, device)
    # image_with_clicks = preprocess_image(image_with_clicks, device)

    # set lora
    if lora_path == "":
        print("applying default parameters")
        model.unet.set_default_attn_processor()
    else:
        print("applying lora: " + lora_path)
        model.unet.load_attn_procs(lora_path)

    # invert the source image
    # the latent code resolution is too small, only 64*64
    invert_code = model.invert(
        source_image,
        prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.n_inference_step,
        num_actual_inference_steps=args.n_actual_inference_step)

    mask = torch.from_numpy(mask).float() / 255.
    mask[mask > 0.0] = 1.0
    mask = rearrange(mask, "h w -> 1 1 h w").cuda()
    mask = F.interpolate(mask, (args.sup_res_h, args.sup_res_w),
                         mode="nearest")

    handle_points = []
    target_points = []
    # here, the point is in x,y coordinate
    for idx, point in enumerate(points):
        cur_point = torch.tensor([
            point[1] / full_h * args.sup_res_h,
            point[0] / full_w * args.sup_res_w
        ])
        cur_point = torch.round(cur_point)
        if idx % 2 == 0:
            handle_points.append(cur_point)
        else:
            target_points.append(cur_point)
    print('handle points:', handle_points)
    print('target points:', target_points)

    init_code = invert_code
    init_code_orig = deepcopy(init_code)
    model.scheduler.set_timesteps(args.n_inference_step)
    t = model.scheduler.timesteps[args.n_inference_step -
                                  args.n_actual_inference_step]

    # hijack the attention module
    # inject the reference branch to guide the generation
    editor = MutualSelfAttentionControl(start_step=start_step,
                                        start_layer=start_layer,
                                        total_steps=args.n_inference_step,
                                        guidance_scale=args.guidance_scale)

    # feature shape: [1280,16,16], [1280,32,32], [640,64,64], [320,64,64]
    # update according to the given supervision
    if args.baseline:
        shift = 0
        updated_init_code = drag_diffusion_update(model, init_code, t,
                                                  handle_points, target_points,
                                                  mask, args)
        # for baseline, registry masactrl after drag
        # model = registry_masactrl(model, editor, lora_path)
    else:
        # for our implementation, registry masactrl before drag
        # model = registry_masactrl(model, editor, lora_path)
        updated_init_code, shift = drag_diffusion_update_restart_multi(
            model, init_code, t,
            handle_points, target_points,
            mask, args, editor)

    model = registry_masactrl(model, editor, lora_path)
    args.n_actual_inference_step += shift

    # inference the synthesized image
    gen_image = model(
        prompt=args.prompt,
        batch_size=2,
        latents=torch.cat([init_code_orig, updated_init_code], dim=0),
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.n_inference_step,
        num_actual_inference_steps=args.n_actual_inference_step)[1].unsqueeze(
            dim=0)

    # resize gen_image into the size of source_image
    # we do this because shape of gen_image will be rounded to multipliers of 8
    gen_image = F.interpolate(gen_image, (full_h, full_w), mode='bilinear')

    # save the original image, user editing instructions, synthesized image
    # save_result = torch.cat([
    #     source_image * 0.5 + 0.5,
    #     torch.ones((1,3,full_h,25)).cuda(),
    #     image_with_clicks * 0.5 + 0.5,
    #     torch.ones((1,3,full_h,25)).cuda(),
    #     gen_image[0:1]
    # ], dim=-1)

    # if not os.path.isdir(save_dir):
    #     os.mkdir(save_dir)
    # save_prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    # save_image(save_result, os.path.join(save_dir, save_prefix + '.png'))

    out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    out_image = (out_image * 255).astype(np.uint8)
    return out_image


if __name__ == '__main__':
    args = get_args()
    all_category = [
        'art_work',
        'land_scape',
        'building_city_view',
        'building_countryside_view',
        'animals',
        'human_head',
        'human_upper_body',
        'human_full_body',
        'interior_design',
        'other_objects',
    ]
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

    if args.cate is not None:
        assert args.cate in all_category, (
            f'User defined cate ({args.cate}) must be in all categorys.')
        print(f'Run drag on cate {args.cate}')
        all_category = [args.cate]

    if args.quick_run:
        all_category = ['animals', 'building_city_view', 'art_work', 'building_city_view', 'human_full_body']

    # assume root_dir and lora_dir are valid directory
    root_dir = 'drag_bench_data'
    lora_dir = 'drag_bench_lora'
    if args.work_dir is not None:
        result_dir = args.work_dir
    else:
        result_dir = 'drag_diffusion_res'

    # mkdir if necessary
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
        for cat in all_category:
            os.makedirs(os.path.join(result_dir, cat), exist_ok=True)

    for cat in all_category:
        file_dir = os.path.join(root_dir, cat)
        for sample_name in os.listdir(file_dir):
            if sample_name.startswith('.'):
                continue
            sample_path = os.path.join(file_dir, sample_name)
            result_path = os.path.join(result_dir, cat, sample_name,
                                       'dragged_image.png')
            if os.path.exists(result_path):
                continue

            # read image file
            source_image = Image.open(
                os.path.join(sample_path, 'original_image.png'))
            source_image = np.array(source_image)

            # load meta data
            with open(os.path.join(sample_path, 'meta_data.pkl'), 'rb') as f:
                meta_data = pickle.load(f)
            prompt = meta_data['prompt']
            mask = meta_data['mask']
            points = meta_data['points']

            save_dir = os.path.join(result_dir, cat, sample_name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

            # load lora
            # using LoRA @ 200 steps
            lora_path = os.path.join(lora_dir, cat, sample_name, str(200))
            print("applying lora: " + lora_path)

            out_image = run_drag(
                source_image,
                mask,
                prompt,
                points,
                inversion_strength=args.inversion_strength,
                lam=0.1,
                latent_lr=0.01,
                n_pix_step=args.n_pix_step,
                model_path="runwayml/stable-diffusion-v1-5",
                vae_path="default",
                lora_path=lora_path,
                start_step=0,
                start_layer=10,
                user_args=args,
                save_dir=save_dir
            )
            Image.fromarray(out_image).save(
                os.path.join(save_dir, 'dragged_image.png'))
