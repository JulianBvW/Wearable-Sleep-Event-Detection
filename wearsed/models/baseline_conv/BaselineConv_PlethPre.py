'''
Baseline model for WearSED using a U-Net (Using Pleth Preprocessing instead of whole signal)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineConv(nn.Module):
    def __init__(self, pleth_statistical=True, pleth_vae_latents=True, out_channels=1):
        super(BaselineConv, self).__init__()
        in_channels = 2 + pleth_statistical*5 + pleth_vae_latents*8

        # Encoder
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.enc_conv_1 = double_conv_block(in_channels, 8)
        self.enc_conv_2 = double_conv_block(8, 16)
        self.enc_conv_3 = double_conv_block(16, 32)
        self.enc_conv_4 = double_conv_block(32, 64)

        # Decoder
        self.up_transpose_1 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.up_transpose_2 = nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.up_transpose_3 = nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=2, stride=2)
        self.dec_conv_1 = double_conv_block(64, 32)
        self.dec_conv_2 = double_conv_block(32, 16)
        self.dec_conv_3 = double_conv_block(16, 8)

        # Output
        self.out = nn.Conv1d(in_channels=8, out_channels=out_channels, kernel_size=1) 

    def forward(self, x):                              # [bs, in_channels, 600]

        # Encoding
        enc_1 = self.enc_conv_1(x)                     # [bs, 8,   600]
        enc_2 = self.enc_conv_2(self.max_pool(enc_1))  # [bs, 16,  300]
        enc_3 = self.enc_conv_3(self.max_pool(enc_2))  # [bs, 32,  150]
        x = self.enc_conv_4(self.max_pool(enc_3))      # [bs, 64,  75]

        # Decoding
        x = self.up_transpose_1(x)                     # [bs, 32, 150]
        x = torch.cat([enc_3, x], 1)                   # [bs, 64, 150]
        x = self.dec_conv_1(x)                         # [bs, 32, 150]

        x = self.up_transpose_2(x)                     # [bs, 16, 300]
        x = torch.cat([enc_2, x], 1)                   # [bs, 32, 300]
        x = self.dec_conv_2(x)                         # [bs, 16, 300]

        x = self.up_transpose_3(x)                     # [bs,  8, 600]
        x = torch.cat([enc_1, x], 1)                   # [bs, 16, 600]
        x = self.dec_conv_3(x)                         # [bs,  8, 600]

        # Output
        out = self.out(x).squeeze(1)                   # [bs, 600]
        return out


def double_conv_block(in_channels, out_channels, kernel_size=3):
    block = nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True)
    )
    return block