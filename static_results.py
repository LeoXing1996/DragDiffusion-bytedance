"""This script is used to static the results for the entire benchmark
"""
import os
import os.path as osp
from argparse import ArgumentParser
from collections import defaultdict

from mmengine import Config


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--work-dir', type=str)
    parser.add_argument('--subset', type=str)
    parser.add_argument('--benchmark', type=str)
    return parser.parse_args()


def main():
    args = get_args()

    # filter subset
    subset_item_index = []
    if args.subset is not None:
        assert args.benchmark is not None, (
            'benchamrk file must be passed when subset evaluation is set.')
        benchmark = Config.fromfile(args.benchmark)
        for sample in benchmark['samples']:
            tag, item_index = sample['tag'], sample['index']
            if tag.upper() == args.subset.upper():
                subset_item_index.append(item_index)

        print(
            f'Found {len(subset_item_index)} items for subset {args.subset}.')

    work_dir = args.work_dir
    benchmark_results = [
        osp.join(work_dir, f) for f in os.listdir(work_dir)
        if osp.isdir(osp.join(work_dir, f)) and not f.startswith('.')
    ]
    # benchmark_results = [osp.join(work_dir, f) for f in os.listdir(work_dir)]
    valid_count = 0
    error_file = []
    global_res_dict = defaultdict(float)
    for idx, res_dir in enumerate(benchmark_results):

        item_index = int(osp.basename(res_dir))
        if subset_item_index and item_index not in subset_item_index:
            continue

        res_file = osp.join(res_dir, 'res.py')
        if not osp.exists(res_file):
            error_file.append(res_file)
            continue
        res = Config.fromfile(res_file)

        # check keys
        if idx != 0:
            keys = list(res.keys())
            if not all([k in keys for k in global_res_dict.keys()]):
                error_file.append(res_file)
                continue

        for k, v in res.items():
            global_res_dict[k] += v

        valid_count += 1

    for k, v in global_res_dict.items():
        global_res_dict[k] = v / valid_count

    if args.subset:
        res_save_path = osp.join(args.work_dir, f'res_{args.subset}.py')
    else:
        res_save_path = osp.join(args.work_dir, 'res.py')
    global_res = Config(global_res_dict)
    global_res.dump(res_save_path)
    print(f'Save results to {res_save_path}')
    if error_file:
        print('Find following invalid results:')
        for f in error_file:
            print(f'    {f}')


if __name__ == '__main__':
    main()
