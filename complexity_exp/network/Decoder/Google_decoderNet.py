import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.backends import cudnn

from utils.helper import load_config

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


class Google_decoderNet(nn.Module):

    def __init__(self):
        super(Google_decoderNet, self).__init__()
        self.define_decoder()

    def define_decoder(self):
        self.conv1 = nn.Conv2d(3, 50, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 10, kernel_size=4, padding=2)
        self.conv3 = nn.Conv2d(3, 5, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(65, 50, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(65, 10, kernel_size=4, padding=2)
        self.conv6 = nn.Conv2d(65, 5, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(65, 50, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(65, 10, kernel_size=4, padding=2)
        self.conv9 = nn.Conv2d(65, 5, kernel_size=5, padding=2)
        self.conv10 = nn.Conv2d(65, 50, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(65, 10, kernel_size=4, padding=2)
        self.conv12 = nn.Conv2d(65, 5, kernel_size=5, padding=2)
        self.conv13 = nn.Conv2d(65, 50, kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(65, 10, kernel_size=4, padding=2)
        self.conv15 = nn.Conv2d(65, 5, kernel_size=5, padding=2)
        self.conv16 = nn.Conv2d(65, 3, kernel_size=3, padding=1)  # 3改1

    def forward(self, encoder_output):
        x = encoder_output
        # print("encoder_output")
        # print(encoder_output.shape)
        pad = (0, -1, 0, -1)
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        # print(x2.shape)
        x2 = F.pad(x2, pad, 'constant', 0)
        x3 = F.relu(self.conv3(x))
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        x4 = torch.cat([x1, x2, x3], 1)
        x1 = F.relu(self.conv4(x4))
        x2 = F.relu(self.conv5(x4))
        x2 = F.pad(x2, pad, 'constant', 0)
        x3 = F.relu(self.conv6(x4))
        x4 = torch.cat([x1, x2, x3], 1)
        x1 = F.relu(self.conv7(x4))
        x2 = F.relu(self.conv8(x4))
        x2 = F.pad(x2, pad, 'constant', 0)
        x3 = F.relu(self.conv9(x4))
        x4 = torch.cat([x1, x2, x3], 1)
        x1 = F.relu(self.conv10(x4))
        x2 = F.relu(self.conv11(x4))
        x2 = F.pad(x2, pad, 'constant', 0)
        x3 = F.relu(self.conv12(x4))
        x4 = torch.cat([x1, x2, x3], 1)
        x1 = F.relu(self.conv13(x4))
        x2 = F.relu(self.conv14(x4))
        x2 = F.pad(x2, pad, 'constant', 0)
        x3 = F.relu(self.conv15(x4))
        x4 = torch.cat([x1, x2, x3], 1)
        output = F.relu(self.conv16(x4))
        # output = self.conv16(x4)
        # print("output")
        # print(output.shape)
        return output
