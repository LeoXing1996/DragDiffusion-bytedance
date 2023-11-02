import os
import os.path as osp
import pickle
from argparse import ArgumentParser
from time import sleep
from typing import Dict

from mmengine import Config
from PIL import Image


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--benchmark',
                        type=str,
                        help='The benchmark file to run.')
    parser.add_argument('--resume',
                        type=str,
                        default='./loras-drag-bench',
                        help='The folder to save lora models for benchmark')

    parser.add_argument('--work-dir', type=str, default='./work_dirs')

    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def run_one_item(bench_root: str, cate: str, img_id: str, args):

    data_root = osp.join(bench_root, cate, img_id)

    work_dir = osp.join(args.work_dir, cate)
    sub_work_dir = osp.join(args.work_dir, cate, img_id)

    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(sub_work_dir, exist_ok=True)

    meta_path = osp.join(data_root, 'meta_data.pkl')
    with open(meta_path, 'rb') as file:
        meta = pickle.load(file)
    img_path = osp.join(data_root, 'original_image.png')

    mask = meta['mask']
    mask_path = osp.join(sub_work_dir, 'mask.png')
    Image.fromarray(mask * 255).save(mask_path)

    config = dict()
    config['lora'] = osp.join(args.resume, cate, img_id)
    config['mask'] = mask_path
    config['points'] = meta['points']
    config['prompt'] = meta['prompt']
    config['src_path'] = img_path

    config_path = osp.join(work_dir, f'{img_id}.py')

    base_config = Config(config)
    base_config.dump(config_path)

    out_path = osp.join(work_dir, f'{img_id}.out')
    job_script = (f'srun --async -p mm_lol '
                  f'--job-name "drag-bytedance-{img_id}" '
                  f'-o "{out_path}" '
                  f'-n1 -N1 --gres=gpu:1 --cpus-per-task 4 ')

    # NOTE: rerun all latent in benchmark
    entry = 'run_drag.py'

    run_cmd = (f'python {entry}'
               f' --config "{config_path}"'
               f' --work-dir  "{sub_work_dir}"')

    script_path = osp.join(work_dir, f'drag-{img_id}.sh')
    srun_cmd = f'{job_script} {run_cmd}'

    print(f'\n\nTraining command for {img_id}:')
    print(run_cmd)
    print(f'srun command for {img_id}')
    print(srun_cmd)

    with open(script_path, 'w') as f:
        f.write(run_cmd)

    if not args.dry_run:

        os.system(f'/bin/bash -c "{srun_cmd}"')
        sleep(1)


def main():
    args = parse_args()
    # run with benchmark
    benchmark = args.benchmark

    catetories = os.listdir(benchmark)
    for cate in catetories:

        if cate != 'animals':
            continue

        if cate.startswith('.'):
            continue
        cate_dir = osp.join(benchmark, cate)
        for data in os.listdir(cate_dir):
            if data.startswith('.'):
                continue
            img_id = data
            # save_path = osp.join(args.save_path, cate, data)
            # run_one_item(meta, img_root, save_root, save_path, img_id, args)
            run_one_item(benchmark, cate, img_id, args)


if __name__ == '__main__':
    main()
