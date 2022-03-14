import torch
from torch import nn
import torch.nn.functional as F


class decoderNet(nn.Module):
    """
    在decoder时，payload和source的定义是分离的
    """
    def __init__(self):
        super(decoderNet, self).__init__()
        self.define_decoder()


    def define_decoder(self):
        self.decoder_layers1 = nn.Conv2d(3, 256, kernel_size=3, padding=1)
        self.decoder_layers2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        #         self.decoder_bn2 = nn.BatchNorm2d(64)

        self.decoder_layers3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder_layers4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        #         self.decoder_bn4 = nn.BatchNorm2d(32)

        self.decoder_layers5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        # payload_decoder
        self.decoder_payload1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.decoder_payload2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        self.decoder_payload3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.decoder_payload4 = nn.Conv2d(8, 8, kernel_size=3, padding=1)

        self.decoder_payload5 = nn.Conv2d(8, 3, kernel_size=3, padding=1)
        self.decoder_payload6 = nn.Conv2d(3, 1, kernel_size=3, padding=1)

        # source_decoder
        self.decoder_source1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.decoder_source2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)

        self.decoder_source3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.decoder_source4 = nn.Conv2d(8, 8, kernel_size=3, padding=1)

        self.decoder_source5 = nn.Conv2d(8, 3, kernel_size=3, padding=1)
        self.decoder_source6 = nn.Conv2d(3, 3, kernel_size=3, padding=1)


    def forward(self, encoder_output):


        d = encoder_output.reshape(-1, 3, 32, 32)  # torch.Size([batch_size, 3, 32, 32])
        # layer1
        d = F.relu(self.decoder_layers1(d))
        d = F.relu(self.decoder_layers2(d))
        #         d = self.decoder_bn2(d)

        # layer3
        d = F.relu(self.decoder_layers3(d))
        d = F.relu(self.decoder_layers4(d))
        #         d = self.decoder_bn4(d)

        init_d = F.relu(self.decoder_layers5(d))

        # ---------------- decoder_payload ----------------

        # layer 1 & 2
        d = F.relu(self.decoder_payload1(init_d))
        d = F.relu(self.decoder_payload2(d))
        # layer 3 & 4
        d = F.relu(self.decoder_payload3(d))
        d = F.relu(self.decoder_payload4(d))
        # layer 5 & 6
        d = F.relu(self.decoder_payload5(d))
        decoded_payload = self.decoder_payload6(d)

        return decoded_payload



