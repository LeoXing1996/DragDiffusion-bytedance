from argparse import ArgumentParser
import os
import os.path as osp
import torch
from einops import rearrange

import cv2
import numpy as np
from lpips import LPIPS
from mmengine import Config
from PIL import Image

from utils.ui_utils import run_drag as _run_drag


def ssim(img1, img2):
    """
    img1 and img2's input range is [-1, 1]
    """
    # convert [H, W, 3] in [0, 255]
    # img1 = ((img1 + 1) / 2 * 255).permute(0, 2, 3, 1).cpu().numpy()[0]
    # img2 = ((img2 + 1) / 2 * 255).permute(0, 2, 3, 1).cpu().numpy()[0]

    ssims = []
    for idx in range(img1.shape[-1]):
        ssims.append(_ssim(img1[..., idx], img2[..., idx]))

    return np.array(ssims).mean()


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`ssim`.

    Args:
        img1, img2 (np.ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: SSIM result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def psnr(img1, img2):
    # convert [H, W, 3] in [0, 255]
    # img1 = ((img1 + 1) / 2 * 255).permute(0, 2, 3, 1).cpu().numpy()[0]
    # img2 = ((img2 + 1) / 2 * 255).permute(0, 2, 3, 1).cpu().numpy()[0]

    mse_value = ((img1 - img2)**2).mean()
    if mse_value == 0:
        result = float('inf')
    else:
        result = 20. * np.log10(255. / np.sqrt(mse_value))

    return result


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--work-dir')

    return parser.parse_args()


def draw_points(img, points):
    # points = []
    for idx, point in enumerate(points):
        if idx % 2 == 0:
            # draw a red circle at the handle point
            cv2.circle(img, tuple(point), 10, (255, 0, 0), -1)
        else:
            # draw a blue circle at the handle point
            cv2.circle(img, tuple(point), 10, (0, 0, 255), -1)
        # points.append(tuple(point))
        # draw an arrow from handle point to target point
        if len(points) == 2:
            cv2.arrowedLine(img,
                            points[0],
                            points[1], (255, 255, 255),
                            4,
                            tipLength=0.5)
            # points = []
    return img if isinstance(img, np.ndarray) else np.array(img)


def main():
    args = get_args()
    config_path = args.config
    config = Config.fromfile(config_path)

    # prepare hyper parameters
    lora_path = config.lora

    n_pix_step = 40
    lam = 0.1
    inversion_strength = 0.75
    latent_lr = 0.01
    start_step = 0
    start_layer = 10

    model_path = 'runwayml/stable-diffusion-v1-5'
    vae_path = 'default'

    prompt = config.prompt
    points = config.points

    src_image_path = config.src_path
    tar_image_path = config.tar_path

    source_image = np.array(Image.open(src_image_path))
    target_image = np.array(Image.open(tar_image_path))

    image_with_clicks = draw_points(np.array(Image.open(src_image_path)),
                                    points)
    mask = np.array(Image.open(config.mask))

    # 2. run drag
    out_image = _run_drag(source_image,
                          image_with_clicks,
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
                          save_dir="./results")

    os.makedirs(args.work_dir, exist_ok=True)
    Image.fromarray(out_image).save(osp.join(args.work_dir, 'out.png'))
    # 3. run evaluation
    # out_image = np.array(out_image)  # [0, 255]
    mask = np.array(Image.open(config.mask))

    mask = mask[..., None]
    gen_fg = out_image * mask
    tar_fg = target_image * mask

    gen_bg = out_image * (1 - mask)
    tar_bg = source_image * (1 - mask)

    _lpips = LPIPS().cuda()

    fg_ssim = ssim(gen_fg, tar_fg)
    fg_psnr = psnr(gen_fg, tar_fg)

    bg_ssim = ssim(gen_bg, tar_bg)
    bg_psnr = psnr(gen_bg, tar_bg)

    def np_to_ten(img):
        # convert to [-1, 1]
        image = torch.from_numpy(img).float() / 127.5 - 1  # [-1, 1]
        image = rearrange(image, "h w c -> 1 c h w")
        image = image.cuda()
        return image

    fg_lpips = _lpips(np_to_ten(gen_fg), np_to_ten(tar_fg)).item()
    bg_lpips = _lpips(np_to_ten(gen_bg), np_to_ten(tar_bg)).item()

    # 4. dump results
    res_dict = dict(fg_ssim=fg_ssim,
                    bg_ssim=bg_ssim,
                    fg_psnr=fg_psnr,
                    bg_psnr=bg_psnr,
                    fg_lpips=fg_lpips,
                    bg_lpips=bg_lpips)
    res_dict_cfg = Config(res_dict)
    cfg_save_path = osp.join(args.work_dir, 'res.py')
    res_dict_cfg.dump(cfg_save_path)

    print(f'Results saved to {cfg_save_path}')


if __name__ == '__main__':
    main()
