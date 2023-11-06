import os
import os.path as osp
from matplotlib import pyplot as plt

DEBUG = int(os.environ.get('DEBUG', 0)) == 1


def act_when_debug(force=False):
    def decorator(func):
        def wrapped_func(*args, **kwargs):
            DEBUG = int(os.environ.get('DEBUG', 0)) == 1
            if DEBUG or force:
                return func(*args, **kwargs)
        return wrapped_func
    return decorator


@act_when_debug()
def draw_loss(loss_list, dist_list, restart_list, save_dir, save_suffix=None):
    save_name = osp.join(save_dir, 'loss.png')
    index = list(range(len(loss_list)))

    fig, ax1 = plt.subplots()
    ax1.set_ylabel('Loss')
    for idx in index:
        if restart_list[idx]:
            ax1.vlines(x=idx, ymin=0, ymax=max(loss_list), colors='r', linestyles='dashed')
    ax1.plot(index, loss_list, color='b')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Mean Dist')
    ax2.plot(index, dist_list, color='g')

    if save_suffix is not None:
        orig_suffix = save_name.split('.')[-1]
        orig_prefix = save_name[:-len(orig_suffix)-1]
        save_name = orig_prefix + '_' + save_suffix + '.' + orig_suffix
    fig.savefig(save_name)
    plt.close()


@act_when_debug()
def draw_feature():
    pass
