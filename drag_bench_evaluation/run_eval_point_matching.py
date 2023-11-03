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

# run evaluation of mean distance between the desired target points and the position of final handle points
import os
import pickle
import numpy as np
import PIL
from PIL import Image
from torchvision.transforms import PILToTensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from dift_sd import SDFeaturizer
from pytorch_lightning import seed_everything
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default='drag_diffusion_res')
    parser.add_argument('--is-batch', action='store_true')
    parser.add_argument('--save-path', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # using SD-2.1
    dift = SDFeaturizer('stabilityai/stable-diffusion-2-1')

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

    original_img_root = 'drag_bench_data/'
    # you may put more root path of your results here
    if args.is_batch:
        evaluate_root = [os.path.join(args.root, folder) for folder in os.listdir(args.root)]
    else:
        evaluate_root = [args.root]

    evaluation_result = dict()
    for target_root in evaluate_root:
        # fixing the seed for semantic correspondence
        seed_everything(42)

        all_dist = []
        tar_result = dict()
        for cat in all_category:
            cat_dist = []
            for file_name in os.listdir(os.path.join(original_img_root, cat)):
                if file_name.startswith('.'):
                    continue
                with open(os.path.join(original_img_root, cat, file_name, 'meta_data.pkl'), 'rb') as f:
                    meta_data = pickle.load(f)
                prompt = meta_data['prompt']
                points = meta_data['points']

                # here, the point is in x,y coordinate
                handle_points = []
                target_points = []
                for idx, point in enumerate(points):
                    # from now on, the point is in row,col coordinate
                    cur_point = torch.tensor([point[1], point[0]])
                    if idx % 2 == 0:
                        handle_points.append(cur_point)
                    else:
                        target_points.append(cur_point)

                source_image_path = os.path.join(original_img_root, cat, file_name, 'original_image.png')
                dragged_image_path = os.path.join(target_root, cat, file_name, 'dragged_image.png')
                if not os.path.exists(dragged_image_path):
                    dragged_image_path_ = os.path.join(target_root, cat, file_name, 'drag.png')
                    assert os.path.exists(dragged_image_path_), (
                        f'Both {dragged_image_path} and {dragged_image_path_} can not be found, '
                        'Please check your folder structure.')
                    dragged_image_path = dragged_image_path_

                source_image_PIL = Image.open(source_image_path)
                dragged_image_PIL = Image.open(dragged_image_path)
                dragged_image_PIL = dragged_image_PIL.resize(source_image_PIL.size,PIL.Image.BILINEAR)

                source_image_tensor = (PILToTensor()(source_image_PIL) / 255.0 - 0.5) * 2
                dragged_image_tensor = (PILToTensor()(dragged_image_PIL) / 255.0 - 0.5) * 2

                _, H, W = source_image_tensor.shape

                with torch.no_grad():
                    ft_source = dift.forward(source_image_tensor,
                        prompt=prompt,
                        t=261,
                        up_ft_index=1,
                        ensemble_size=8)
                    ft_source = F.interpolate(ft_source, (H, W), mode='bilinear')

                    ft_dragged = dift.forward(dragged_image_tensor,
                        prompt=prompt,
                        t=261,
                        up_ft_index=1,
                        ensemble_size=8)
                    ft_dragged = F.interpolate(ft_dragged, (H, W), mode='bilinear')

                    cos = nn.CosineSimilarity(dim=1)
                    for pt_idx in range(len(handle_points)):
                        hp = handle_points[pt_idx]
                        tp = target_points[pt_idx]

                        num_channel = ft_source.size(1)
                        src_vec = ft_source[0, :, hp[0], hp[1]].view(1, num_channel, 1, 1)
                        cos_map = cos(src_vec, ft_dragged).cpu().numpy()[0]  # H, W
                        max_rc = np.unravel_index(cos_map.argmax(), cos_map.shape) # the matched row,col

                        # calculate distance
                        dist = (tp - torch.tensor(max_rc)).float().norm()
                        all_dist.append(dist)
                        cat_dist.append(dist)
            tar_result[cat] = dict(mean_dist=np.mean(cat_dist))

        if args.is_batch:
            prefix = target_root.split('/')[-1]
        else:
            prefix = target_root
        tar_result['AVG'] = dict(mean_dist=np.mean(all_dist))
        evaluation_result[prefix] = tar_result
        print(prefix + ' mean distance: ', torch.tensor(all_dist).mean().item())

    if args.save_path:
        if os.path.exists(args.save_path):
            import datetime
            now = datetime.datetime.now()
            timestamp_str = now.strftime('%m-%dT%H:%M')
            print(f'{args.save_path} already exist, '
                  f'saved with timestamp {timestamp_str}.')
            suffix = args.save_path.split('.')[-1]
            prefix = args.save_path[:-len(suffix)-1]
            save_path = f'{prefix}-{timestamp_str}.{suffix}'
        else:
            save_path = args.save_path
        with open(save_path, 'wb') as file:
            pickle.dump(evaluation_result, file)
