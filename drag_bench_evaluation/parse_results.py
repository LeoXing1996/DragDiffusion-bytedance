import os
import os.path as osp
import pickle
from argparse import ArgumentParser
from copy import deepcopy

from mmengine import Config
from pandas import DataFrame


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--md-result', type=str)
    parser.add_argument('--sim-result', type=str)
    parser.add_argument('--work-dir', type=str)
    parser.add_argument('--save-path', type=str)
    return parser.parse_args()


def get_config(work_dir, prefix):
    if work_dir is None:
        return dict()

    setting_root = osp.join(work_dir, prefix)
    cates = os.listdir(setting_root)
    cates_dir = [osp.join(setting_root, folder) for folder in cates]
    # NOTE: all cates use the same base config,
    # therefore we only load the first
    first_cate_dir = cates_dir[0]
    first_subset_dir = osp.join(first_cate_dir, os.listdir(first_cate_dir)[0])
    config_path = osp.join(first_subset_dir, 'config.py')
    return Config.fromfile(config_path)


def get_duration(config):
    start = config.get('drag_step', 1)
    duration = config.get('restart_iter', 50)
    return f's{start}, e{start+duration-1}'


def get_feat(config):
    return config.get('feature_index', [1, 2])


def get_var(config):
    return config.get('feature_chn_dict', None) is not None


def get_n_frames(config):
    return config.get('n_frames', 1)


def get_restart_strategy(config):
    return config.get('restart_strategy', None)


def main():
    args = get_args()
    md_result_file = args.md_result
    sim_result_file = args.sim_result

    if md_result_file is not None:
        with open(md_result_file, 'rb') as file:
            md_result = pickle.load(file)
    else:
        md_result = {}
    if sim_result_file is not None:
        with open(sim_result_file, 'rb') as file:
            sim_result = pickle.load(file)
    else:
        sim_result = {}

    # merge result
    assert md_result or sim_result, (
        'MD result and SIM result must be passed at least one.')

    if not md_result:
        result = sim_result
    elif not sim_result:
        result = md_result
    elif md_result is not None and sim_result is not None:
        # check key consistency
        result = deepcopy(sim_result)

        # check prefix
        prefix_sim = sim_result.keys()
        prefix_md = md_result.keys()
        assert prefix_md == prefix_sim, ('prefix for SIM and MD are not same.')
        for prefix in list(prefix_sim):

            # check cat
            cat_sim = sim_result[prefix].keys()
            cat_md = md_result[prefix].keys()
            assert cat_sim == cat_md, (
                f'category for prefix {prefix} are not same.')
            for cat in list(cat_sim):
                result[prefix][cat]['mean_dist'] = md_result[prefix][cat]['mean_dist']

    else:
        raise ValueError(
            'MD result and SIM result must be passed at least one.')

    data_frame = DataFrame(columns=[
        'Name', 'Subset', 'nFrames', 'timesteps', 'Restart Strategy', 'Feat',
        'CLIP', 'LPIPS', 'MD'
    ])

    for prefix, prefix_result in result.items():
        config = get_config(args.work_dir, prefix)
        for cate, cate_result in prefix_result.items():
            name = f'{prefix} {cate}'
            md = cate_result.get('mean_dist', -1)
            clip = cate_result.get('clip_sim', -1)
            lpips = cate_result.get('lpips', -1)

            row = [
                name,
                cate,
                get_n_frames(config),
                get_duration(config),
                get_restart_strategy(config),
                get_feat(config),
                clip,
                lpips,
                md,
            ]
            data_frame.loc[len(data_frame)] = row

    data_frame.to_csv(args.save_path)
    print(data_frame)


if __name__ == '__main__':
    main()
