import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pathlib import Path
import math
import os
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.paths import DATASET_DIR

from src.gol_adv_sys.DatasetManager import FixedDataset
from src.gol_adv_sys.DeviceManager import DeviceManager
from src.gol_adv_sys.utils.helper_functions import get_config_from_batch
from src.gol_adv_sys.utils import constants as constants


def gaussian_window(size, sigma):
    gauss = torch.Tensor([math.exp(-(x - size//2)**2/float(2*sigma**2)) for x in range(size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, size_average=False):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = (0.01*1)**2
    C2 = (0.03*1)**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def main():
    device_manager = DeviceManager()
    train_path = DATASET_DIR / f"{constants.dataset_name}_train.pt"
    dataset_train = FixedDataset(train_path)
    train_dataloader = DataLoader(dataset_train, batch_size=constants.bs, shuffle=True)

    config = None
    for batch in train_dataloader:
        config = batch
        break

    # Get configs from batch
    initial_config = get_config_from_batch(config, constants.CONFIG_NAMES["initial"], device_manager.default_device)
    final_config = get_config_from_batch(config, constants.CONFIG_NAMES["final"], device_manager.default_device)
    easy_metric = get_config_from_batch(config, constants.CONFIG_NAMES["metric_easy"], device_manager.default_device)
    medium_metric = get_config_from_batch(config, constants.CONFIG_NAMES["metric_medium"], device_manager.default_device)
    hard_metric = get_config_from_batch(config, constants.CONFIG_NAMES["metric_hard"], device_manager.default_device)

    zeros = torch.zeros(128, 1, 32, 32).to(device_manager.default_device)
    ones = torch.ones(128, 1, 32, 32).to(device_manager.default_device)

    # Calculate SSIM
    score = ssim(easy_metric, medium_metric)
    score_2 = ssim(easy_metric, hard_metric)
    score_3 = ssim(zeros, ones)
    score_4 = ssim(zeros, zeros)

    print("SSIM score easy-medium:\n", score)
    print("SSIM score easy-hard:\n", score_2)
    print("SSIM score zeros-ones:\n", score_3)
    print("SSIM score zeros-zeros:\n", score_4)

    return 0


if __name__ == "__main__":
    return_code = main()
    exit(return_code)

