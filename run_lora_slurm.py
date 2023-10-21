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
    parser.add_argument('--data', default='./data/')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--only-train-sa', action='store_true')
    parser.add_argument('--only-train-ca', action='store_true')
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
    if args.only_train_sa:
        train_cmd += '\n    --only_train_sa'
    if args.only_train_ca:
        train_cfg += '\n    --only_train_ca'

    print('Call:')
    print(train_cmd)
    if not args.dry_run:
        os.system(train_cmd.replace('\n', ' '))


def train_loras_sbatch(data_item, data_root, promp_template, args):

    cls_name = data_item['class_name']
    img_index = data_item['index']
    image_path = osp.join(data_root, data_item['src_path'])
    prompt = promp_template.format(cls_name)

    train_step = data_item.get('lora_step', 200)
    if args.step is not None:
        train_step = args.step

    # sbatch args
    job_script = (f'#!/bin/bash\n'
                  f'#SBATCH --output {args.save_path}/{img_index}.out\n'
                  f'#SBATCH --partition=mm_lol\n'
                  f'#SBATCH --job-name lora-{img_index}\n'
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
                 f'    --instance_data {image_path}\\\n'
                 f'    --instance_prompt "{prompt}"\\\n'
                 f'    --validation_prompt "{prompt}"\\\n'
                 f'    --train_batch_size 1\\\n'
                 f'    --max_train_steps {train_step}\\\n'
                 f'    --output_dir {args.save_path}/{img_index}\\\n'
                 f'    --lr_scheduler constant\\\n'
                 f'    --lr_warmup_steps 0 \\\n'
                 f'    --max_train_steps 200 \\\n'
                 f'    --lora_rank 16\\\n'
                 f'    --seed 0')

    job_script += train_cmd

    script_path = osp.join(args.save_path, f'lora-{img_index}.sh')
    os.makedirs(args.save_path, exist_ok=True)
    with open(script_path, 'w') as f:
        f.write(job_script)

    print(f'\n\nTraining command for index {img_index}:')
    print(job_script)

    sbatch_cmd = f'sbatch {script_path}'
    if not args.dry_run:
        os.system(sbatch_cmd)
        time.sleep(1)
    return sbatch_cmd


def main():
    args = parse_args()

    benchmark = Config.fromfile(args.benchmark)
    benchmark_items = benchmark['samples']
    data_root = osp.dirname(args.benchmark)
    prompt_template = 'a photo of {}'

    if args.launcher == 'slurm':
        cmd_list = []
        for item in benchmark_items:
            cmd = train_loras_sbatch(item, data_root, prompt_template, args)
            cmd_list.append(cmd)
        cmd_str = '\n'.join(cmd_list)
        # if not args.dry_run:
        #     os.system(cmd_str)
    else:
        for item in benchmark_items:
            train_one_lora(item, data_root, prompt_template, args)


if __name__ == '__main__':
    main()
