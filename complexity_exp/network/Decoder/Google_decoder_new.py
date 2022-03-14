import random

import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn

from complexity_exp.utils.helper import load_config

cfg = load_config()

if torch.cuda.is_available():
    """benchmark设置为True，程序开始时花费额外时间，为整个网络的卷积层搜索最合适的卷积实现算法，
    进而实现网络的加速。要求输入维度和网络（特别是卷积层）结构设置不变，否则程序不断做优化"""
    cudnn.benchmark = True
    if cfg.train.seed is not None:
        np.random.seed(cfg.train.seed)  # Numpy module.
        random.seed(cfg.train.seed)  # Python random module.
        torch.manual_seed(cfg.train.seed)  # Sets the seed for generating random numbers.
        torch.cuda.manual_seed(cfg.train.seed)  # Sets the seed for generating random numbers for the current GPU.
        torch.cuda.manual_seed_all(cfg.train.seed)  # Sets the seed for generating random numbers on all GPUs.
        cudnn.deterministic = True  # 每次返回的卷积算法将是确定的，即默认算法


class RevealNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.r1 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.r2 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.r3 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU())
        self.r4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.r5 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.r6 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=5, padding=2),
            nn.ReLU())
        self.r7 = nn.Sequential(
            nn.Conv2d(150, 3, kernel_size=1, padding=0))

    def forward(self, x):
        r1 = self.r1(x)
        r2 = self.r2(x)
        r3 = self.r3(x)
        x = torch.cat((r1, r2, r3), 1)
        r4 = self.r4(x)
        r5 = self.r5(x)
        r6 = self.r6(x)
        x = torch.cat((r4, r5, r6), 1)
        x = self.r7(x)
        return x


class Google_decoderNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.s3 = RevealNet()

    def forward(self, source):
        output = self.s3(source)
        #print("decoder_output", output.shape)
        return output
