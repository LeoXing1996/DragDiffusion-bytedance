"""This script run the dragBench from DragDiffusion
"""
import pickle
import json
import time
import os
import os.path as osp
from argparse import ArgumentParser
from mmengine import Config


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--benchmark')
    parser.add_argument('--save-path', default='loras')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--step', type=int)

    parser.add_argument('--launcher', type=str)

    return parser.parse_args()


def train_one_lora(data_item, data_root, promp_template, args):
    cls_name = data_item['class']
    img_name = data_item['image'].split('.')[0]
    prompt = promp_template.format(cls_name)

    train_step = data_item.get('lora_step', 200)
    if args.step is not None:
        train_step = args.step

    image_path = osp.join(data_root, data_item['image'])

    train_cmd = ('python train_lora_baseline.py\n'
                 f'    --instance_data_dir {image_path}\n'
                 f'    --instance_prompt "{prompt}"\n'
                 f'    --validation_prompt "{prompt}"\n'
                 f'    --train_batch_size 1\n'
                 f'    --max_train_steps {train_step}\n'
                 f'    --output_dir {args.save_path}/{img_name}')

    print('Call:')
    print(train_cmd)
    if not args.dry_run:
        os.system(train_cmd.replace('\n', ' '))


def train_loras_sbatch(meta, img_path, save_root, save_path, img_id, args):

    prompt = meta['prompt']

    train_step = 200

    # sbatch args
    job_script = (f'#!/bin/bash\n'
                  f'#SBATCH --output {save_root}/{img_id}.out\n'
                  f'#SBATCH --partition=mm_lol\n'
                  f'#SBATCH --job-name lora-{img_id}\n'
                  f'#SBATCH --gres=gpu:1\n'
                  f'#SBATCH --ntasks-per-node=1\n'
                  f'#SBATCH --ntasks=1\n'
                  f'#SBATCH --cpus-per-task=4\n\n')

    # train cmds
    cur_path = osp.dirname(osp.abspath(__file__))
    train_entry = osp.join(cur_path, 'train_lora_baseline.py')
    train_entry = 'lora/train_dreambooth_lora.py'
    model_id = 'runwayml/stable-diffusion-v1-5'
    train_cmd = (f'python {train_entry}\\\n'
                 f'    --pretrained_model_name_or_path {model_id} \\\n'
                 f'    --instance_data {img_path}\\\n'
                 f'    --instance_prompt "{prompt}"\\\n'
                 f'    --validation_prompt "{prompt}"\\\n'
                 f'    --train_batch_size 1\\\n'
                 f'    --max_train_steps {train_step}\\\n'
                 f'    --output_dir {save_path}\\\n'
                 f'    --lr_scheduler constant\\\n'
                 f'    --lr_warmup_steps 0 \\\n'
                 f'    --max_train_steps 200 \\\n'
                 f'    --lora_rank 16\\\n'
                 f'    --seed 0')

    job_script += train_cmd

    script_path = osp.join(save_root, f'lora-{img_id}.sh')
    os.makedirs(save_root, exist_ok=True)
    with open(script_path, 'w') as f:
        f.write(job_script)

    print(f'\n\nTraining command for index {img_id}:')
    print(job_script)

    sbatch_cmd = f'sbatch {script_path}'
    if not args.dry_run:
        os.system(sbatch_cmd)
        time.sleep(1)
    return sbatch_cmd


def main():
    args = parse_args()

    benchmark = args.benchmark
    catetories = os.listdir(benchmark)
    for cate in catetories:
        if cate.startswith('.'):
            continue
        cate_dir = osp.join(benchmark, cate)
        for data in os.listdir(cate_dir):
            if data.startswith('.'):
                continue
            data_dir = osp.join(cate_dir, data)
            meta_path = osp.join(data_dir, 'meta_data.pkl')
            with open(meta_path, 'rb') as file:
                meta = pickle.load(file)
            img_path = osp.join(data_dir, 'original_image.png')
            img_id = data

            save_root = osp.join(args.save_path, cate)
            save_path = osp.join(args.save_path, cate, data)
            train_loras_sbatch(meta, img_path, save_root, save_path, img_id, args)


if __name__ == '__main__':
    main()
