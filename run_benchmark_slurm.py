import os
import os.path as osp
from argparse import ArgumentParser
from time import sleep
from typing import Dict

from mmengine import Config


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        help='The base config file for drag')
    parser.add_argument('--benchmark',
                        type=str,
                        help='The benchmark file to run.')
    parser.add_argument('--resume',
                        type=str,
                        default='./loras',
                        help='The folder to save lora models for benchmark')

    parser.add_argument('--work-dir', type=str, default='./work_dirs')

    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def run_one_item(data_item: Dict,
                 data_root: str,
                 work_dir_root: str,
                 args,
                 suffix: str = ''):

    index = data_item['index']

    image_path = osp.join(data_root, data_item['src_path'])
    image_tar_path = osp.join(data_root, data_item['tar_path'])

    cls_name = data_item['class_name']
    prompt_template = 'a photo of a {}'
    prompt = prompt_template.format(cls_name)

    src_point = data_item['points']['src']
    tar_point = data_item['points']['tar']

    base_config = dict()
    base_config['prompt'] = prompt
    # base_config['src_point'] = src_point
    # base_config['tar_point'] = tar_point

    points = []
    for s, t in zip(src_point, tar_point):
        points.append(s)
        points.append(t)
    base_config['points'] = points

    base_config['src_path'] = image_path
    base_config['tar_path'] = image_tar_path

    if data_item.get('mask', None):
        mask_path = osp.join(data_root, data_item['mask'])
        base_config['mask'] = mask_path

    base_config['lora'] = osp.join(
        args.resume, str(data_item['index']))

    if os.path.exists(work_dir_root):
        work_dir = osp.join(work_dir_root, suffix[1:])
    elif work_dir_root.endswith('/'):
        work_dir = f'{work_dir_root[:-1]}{suffix}'
    else:
        work_dir = f'{work_dir_root}{suffix}'
    os.makedirs(work_dir, exist_ok=True)
    config_path = osp.join(work_dir, f'{index}.py')

    base_config = Config(base_config)
    base_config.dump(config_path)

    out_path = osp.join(work_dir, f'{index}.out')
    job_script = (f'srun --async -p mm_lol '
                  f'--job-name "drag-bytedance-{index}" '
                  f'-o "{out_path}" '
                  f'-n1 -N1 --gres=gpu:1 --cpus-per-task 4 ')

    # NOTE: rerun all latent in benchmark
    entry = 'run_drag.py'
    sub_work_dir = osp.join(work_dir, str(index))
    os.makedirs(sub_work_dir, exist_ok=True)

    run_cmd = (f'python {entry}'
               f' --config "{config_path}"'
               f' --work-dir  "{sub_work_dir}"')

    script_path = osp.join(work_dir, f'drag-{index}.sh')
    srun_cmd = f'{job_script} {run_cmd}'

    print(f'\n\nTraining command for {index}:')
    print(run_cmd)
    print(f'srun command for {index}')
    print(srun_cmd)

    with open(script_path, 'w') as f:
        f.write(run_cmd)

    if not args.dry_run:

        os.system(f'/bin/bash -c "{srun_cmd}"')
        sleep(1)


def main():
    args = parse_args()
    # run with benchmark
    benchmark = Config.fromfile(args.benchmark)

    benchmark_items = benchmark['samples']
    data_root = osp.dirname(args.benchmark)

    for item in benchmark_items:
        run_one_item(item, data_root, args.work_dir, args)


if __name__ == '__main__':
    main()
