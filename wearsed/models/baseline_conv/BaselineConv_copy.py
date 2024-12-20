'''
Baseline model for WearSED using a U-Net
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineConv(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super(BaselineConv, self).__init__()

        # Encoder
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.enc_conv_1 = double_conv_block(in_channels, 64)
        self.enc_conv_2 = double_conv_block(64, 128)
        self.enc_conv_3 = double_conv_block(128, 256)
        self.enc_conv_4 = double_conv_block(256, 512)
        self.enc_conv_5 = double_conv_block(512, 1024)

        # Decoder
        self.up_transpose_1 = nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, output_padding=1)
        self.up_transpose_2 = nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_transpose_3 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_transpose_4 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.dec_conv_1 = double_conv_block(1024, 512)
        self.dec_conv_2 = double_conv_block(512, 256)
        self.dec_conv_3 = double_conv_block(256, 128)
        self.dec_conv_4 = double_conv_block(128, 64)

        # Output
        self.out = nn.Conv1d(in_channels=64, out_channels=out_channels, kernel_size=1) 

    def forward(self, x):  # [bs, 6, 600]

        bs, in_ch, seq_len = x.shape
        print(bs, in_ch, seq_len)
        hyp, spo2, pleth = x[:,0:1], x[:,1].unsqueeze(1), x[:,2:]
        print(hyp.shape, spo2.shape, pleth.shape)
        pleth = pleth.view((bs, 1, -1))
        print(hyp.shape, spo2.shape, pleth.shape)
        signal = torch.cat([hyp, spo2], dim=1)
        print('##', signal.shape)

        print(x.shape)
        print(x[0].shape)

        # Encoding
        enc_1 = self.enc_conv_1(x)                     # [bs, 64,   600]
        enc_2 = self.enc_conv_2(self.max_pool(enc_1))  # [bs, 128,  300]
        enc_3 = self.enc_conv_3(self.max_pool(enc_2))  # [bs, 256,  150]
        enc_4 = self.enc_conv_4(self.max_pool(enc_3))  # [bs, 512,  75]
        x     = self.enc_conv_5(self.max_pool(enc_4))  # [bs, 1024, 37]

        # Decoding
        x = self.up_transpose_1(x)    # [bs, 512,  75]
        x = torch.cat([enc_4, x], 1)  # [bs, 1024, 75]
        x = self.dec_conv_1(x)        # [bs, 512,  75]

        x = self.up_transpose_2(x)    # [bs, 256, 150]
        x = torch.cat([enc_3, x], 1)  # [bs, 512, 150]
        x = self.dec_conv_2(x)        # [bs, 256, 150]

        x = self.up_transpose_3(x)    # [bs, 128, 300]
        x = torch.cat([enc_2, x], 1)  # [bs, 256, 300]
        x = self.dec_conv_3(x)        # [bs, 128, 300]

        x = self.up_transpose_4(x)    # [bs, 64,  600]
        x = torch.cat([enc_1, x], 1)  # [bs, 128, 600]
        x = self.dec_conv_4(x)        # [bs, 64,  600]

        # Output
        out = self.out(x).squeeze(1)  # [bs, 600] 
        return out


def double_conv_block(in_channels, out_channels):
    block = nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    return block