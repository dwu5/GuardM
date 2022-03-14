import torch
from torch import nn
import torch.nn.functional as F


class encoderNet(nn.Module):
    """
    定义encoder时，payload和source的网络层是交错融合的
    """
    def __init__(self):
        super(encoderNet, self).__init__()
        self.define_encoder()

    def define_encoder(self):
        # layer1
        self.encoder_payload_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.encoder_source_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)

        # layer2
        self.encoder_payload_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.encoder_source_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.encoder_source_21 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        # self.encoder_bn2 = nn.BatchNorm2d(32)

        # layer3
        self.encoder_payload_3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.encoder_source_3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        # layer4
        self.encoder_payload_4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.encoder_source_4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.encoder_source_41 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        # self.encoder_bn4 = nn.BatchNorm2d(32)

        # layer5
        self.encoder_payload_5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.encoder_source_5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        # layer6
        self.encoder_payload_6 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.encoder_source_6 = nn.Conv2d(192, 128, kernel_size=3, padding=1)
        self.encoder_source_61 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.encoder_source_62 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        # self.encoder_bn6 = nn.BatchNorm2d(32)

        # layer7
        self.encoder_payload_7 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.encoder_source_7 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        # layer8
        self.encoder_payload_8 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.encoder_source_8 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.encoder_source_81 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.encoder_source_82 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        # self.encoder_bn8 = nn.BatchNorm2d(32)

        # layer9
        self.encoder_source_9 = nn.Conv2d(32, 16, kernel_size=1)

        # layer10
        self.encoder_source_10 = nn.Conv2d(16, 8, kernel_size=1)

        # layer11
        self.encoder_source_11 = nn.Conv2d(8, 3, kernel_size=1)


    def forward(self, input, sec_img):  # 传入source_image 和 payload的元组，再解包

        source, payload = input, sec_img # source：torch.Size([batch_size, 32, 32, 3]) payload：torch.Size([batach_size, 32, 32])
        s = source.reshape((-1, 3, 32, 32))
        p = payload.reshape((-1, 1, 32, 32))

        # layer1
        p = F.relu(self.encoder_payload_1(p))
        s = F.relu(self.encoder_source_1(s))

        # layer2
        p = F.relu(self.encoder_payload_2(p))
        s1 = torch.cat((s, p), 1)  # 64通道
        s = F.relu(self.encoder_source_2(s1))
        s = F.relu(self.encoder_source_21(s1))
        #         s = self.encoder_bn2(s)

        # layer3
        p = F.relu(self.encoder_payload_3(p))
        s = F.relu(self.encoder_source_3(s))

        # layer4
        p = F.relu(self.encoder_payload_4(p))
        s2 = torch.cat((s, p, s1), 1)  # 128通道
        s = F.relu(self.encoder_source_4(s2))
        s = F.relu(self.encoder_source_41(s))
        #         s = self.encoder_bn4(s)

        # layer5
        p = F.relu(self.encoder_payload_5(p))
        s = F.relu(self.encoder_source_5(s))

        # layer6
        p = F.relu(self.encoder_payload_6(p))
        s3 = torch.cat((s, p, s2), 1)  # 192通道
        s = F.relu(self.encoder_source_6(s3))
        s = F.relu(self.encoder_source_61(s))
        s = F.relu(self.encoder_source_62(s))
        #         s = self.encoder_bn6(s)

        # layer7
        p = F.relu(self.encoder_payload_7(p))
        s = F.relu(self.encoder_source_7(s))

        # layer8
        p = F.relu(self.encoder_payload_8(p))
        s4 = torch.cat((s, p, s3), 1)  # 256通道，torch.Size([batch_size, 256, 32, 32])
        s = F.relu(self.encoder_source_8(s4))
        s = F.relu(self.encoder_source_81(s))
        s = F.relu(self.encoder_source_82(s))
        #         s = self.encoder_bn8(s)

        # layer9
        s = F.relu(self.encoder_source_9(s))

        # layer10
        s = F.relu(self.encoder_source_10(s))

        # layer11
        encoder_output = self.encoder_source_11(s)  # torch.Size([batch_size, 3, 32, 32])

        return encoder_output  # 返回载秘图像

