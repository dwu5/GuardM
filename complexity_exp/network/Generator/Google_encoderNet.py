import torch
from torch import nn
import torch.nn.functional as F

from utils.helper import load_config

cfg = load_config()
img_size = cfg.train.dataloader.resize

class Google_encoderNet(nn.Module):

    def __init__(self):
        super(Google_encoderNet, self).__init__()
        self.define_encoder()

    def define_encoder(self):
        # Preparation Network
        self.pad = (0, 1, 0, 1)
        self.conv1 = nn.Conv2d(3, 50, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 10, kernel_size=4, padding=1)
        self.conv3 = nn.Conv2d(3, 5, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(65, 50, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(65, 10, kernel_size=4, padding=1)
        self.conv6 = nn.Conv2d(65, 5, kernel_size=5, padding=2)
        # Hidden network
        self.conv7 = nn.Conv2d(68, 50, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(68, 10, kernel_size=4, padding=1)
        self.conv9 = nn.Conv2d(68, 5, kernel_size=5, padding=2)
        self.conv10 = nn.Conv2d(65, 50, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(65, 10, kernel_size=4, padding=1)
        self.conv12 = nn.Conv2d(65, 5, kernel_size=5, padding=2)
        self.conv13 = nn.Conv2d(65, 50, kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(65, 10, kernel_size=4, padding=1)
        self.conv15 = nn.Conv2d(65, 5, kernel_size=5, padding=2)
        self.conv16 = nn.Conv2d(65, 50, kernel_size=3, padding=1)
        self.conv17 = nn.Conv2d(65, 10, kernel_size=4, padding=1)
        self.conv18 = nn.Conv2d(65, 5, kernel_size=5, padding=2)
        self.conv19 = nn.Conv2d(65, 50, kernel_size=3, padding=1)
        self.conv20 = nn.Conv2d(65, 10, kernel_size=4, padding=1)
        self.conv21 = nn.Conv2d(65, 5, kernel_size=5, padding=2)
        self.conv22 = nn.Conv2d(65, 3, kernel_size=3, padding=1)

    def forward(self, input, sec_img):

        source, payload = input, sec_img
        pad = (0, 1, 0, 1)
        y = source.reshape((-1, 3, img_size, img_size))
        x = payload.reshape((-1, 3, img_size, img_size))
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        # print("=========================秘密图像x")
        # print(x)
        # print("ecox2 pre")
        # print(x2.shape)
        x2 = F.pad(x2, pad, 'constant', 0)
        # print("ecox2 after")
        # print(x2.shape)
        x3 = F.relu(self.conv3(x))
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        x4 = torch.cat([x1, x2, x3], 1)
        # print("=========================第一层x4")
        # print(x4);
        x1 = F.relu(self.conv4(x4))
        x2 = F.relu(self.conv5(x4))
        x2 = F.pad(x2, pad, 'constant', 0)
        x3 = F.relu(self.conv6(x4))
        x4 = torch.cat([x1, x2, x3], 1)
        x4 = torch.cat([y, x4], 1)  # 第三次
        # print("=========================第二层x4")
        # print(x4);
        x1 = F.relu(self.conv7(x4))
        x2 = F.relu(self.conv8(x4))
        x2 = F.pad(x2, pad, 'constant', 0)
        x3 = F.relu(self.conv9(x4))
        x4 = torch.cat([x1, x2, x3], 1)  # 4
        x1 = F.relu(self.conv10(x4))
        x2 = F.relu(self.conv11(x4))
        x2 = F.pad(x2, pad, 'constant', 0)
        x3 = F.relu(self.conv12(x4))
        x4 = torch.cat([x1, x2, x3], 1)  # 5
        # print("=========================第三层x4")
        # print(x4);
        x1 = F.relu(self.conv13(x4))
        x2 = F.relu(self.conv14(x4))
        x2 = F.pad(x2, pad, 'constant', 0)
        x3 = F.relu(self.conv15(x4))
        x4 = torch.cat([x1, x2, x3], 1)  # 6
        # print("=========================第四层x4")
        # print(x4);
        x1 = F.relu(self.conv16(x4))
        x2 = F.relu(self.conv17(x4))
        x2 = F.pad(x2, pad, 'constant', 0)
        x3 = F.relu(self.conv18(x4))
        x4 = torch.cat([x1, x2, x3], 1)  # 7
        # print("=========================第五层x4")
        # print(x4);
        x1 = F.relu(self.conv19(x4))
        x2 = F.relu(self.conv20(x4))
        x2 = F.pad(x2, pad, 'constant', 0)
        x3 = F.relu(self.conv21(x4))
        x4 = torch.cat([x1, x2, x3], 1)  # 8
        # print("=========================第六层x4")
        # print(x4);
        output = F.relu(self.conv22(x4))
        # print("=========================output")
        # print(x4);
        # print(output.shape)
        return output