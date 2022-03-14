# encoding: utf-8
import torch
import torch.nn as nn


class DiscriminatorNet(nn.Module):
    def __init__(self, nc=3, nhf=8, output_function=nn.Sigmoid):
        super(DiscriminatorNet, self).__init__()
        #self.key_pre = RevealPreKey()
        # input is (3+1) x 256 x 256
        self.main = nn.Sequential(
            nn.Conv2d(nc, nhf, 4, 2, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf*2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1),
            nn.BatchNorm2d(nhf*4),
            nn.ReLU(True),
            nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf*2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, nc, 4, 2, 1),
            nn.BatchNorm2d(nc),
            nn.ReLU(True)
            #output_function()
        )
        self.linear = nn.Sequential(
            nn.Linear(192, 1),
            output_function()
        )

        # nn.Sigmoid()
        #self.reveal_Message = UnetRevealMessage()

    def forward(self, input):
        #pkey = pkey.view(-1, 1, 32, 32)
        #pkey_feature = self.key_pre(pkey)

        #input_key = torch.cat([input, pkey_feature], dim=1)
        print("input in dis:", input.shape)
        ste_feature = self.main(input)
        print("ste_feature:", ste_feature.shape)
        out = ste_feature.view(ste_feature.shape[0], -1)
        print("out in dis:", out.shape)
        out = self.linear(out)
        print("out in linear:", out.shape)
        return out
