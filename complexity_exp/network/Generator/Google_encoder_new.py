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


class PrepNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.p1 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU())

        self.p2 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())

        self.p3 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU())

        self.p4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=3, padding=1),
            nn.ReLU())

        self.p5 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())

        self.p6 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=5, padding=2),
            nn.ReLU())

    def forward(self, x):
        # print("input x:", x.shape)
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        x = torch.cat((p1, p2, p3), 1)

        p4 = self.p4(x)
        p5 = self.p5(x)
        p6 = self.p6(x)

        x = torch.cat((p4, p5, p6), 1)
        # print("output x:", x.shape)
        return x


class HidingNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.h1 = nn.Sequential(
            nn.Conv2d(153, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU())

        self.h2 = nn.Sequential(
            nn.Conv2d(153, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())

        self.h3 = nn.Sequential(
            nn.Conv2d(153, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU())

        self.h4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=3, padding=1),
            nn.ReLU())

        self.h5 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())

        self.h6 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=5, padding=2),
            nn.ReLU())

        self.h7 = nn.Sequential(
            nn.Conv2d(150, 3, kernel_size=1, padding=0))

    def forward(self, x):
        h1 = self.h1(x)
        h2 = self.h2(x)
        h3 = self.h3(x)

        x = torch.cat((h1, h2, h3), 1)

        h4 = self.h4(x)
        h5 = self.h5(x)
        h6 = self.h6(x)

        x = torch.cat((h4, h5, h6), 1)
        x = self.h7(x)
        return x


class Google_encoderNet(nn.Module):       #编码解码器拆开是否区别？
    def __init__(self):
        super().__init__()
        self.s1 = PrepNet()
        self.s2 = HidingNet()

    def forward(self, cover, secret):
        # print("cover:{} secret:{}".format(cover.shape, secret.shape))
        x1 = self.s1(secret)
        # print("x1:",x1.shape)
        x = torch.cat((x1, cover), 1)
        output = self.s2(x)
        #print("encoder_output", output.shape)
        return output
