'''
Attention U-Net model for WearSED
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1, use_attention=['gates', 'bottleneck']):
        super(AttentionUNet, self).__init__()

        self.use_attention = use_attention

        # PPG festure extraction
        self.fe_pool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.fe_conv_1 = double_conv_block(1, 4, kernel_size=5)
        self.fe_conv_2 = double_conv_block(4, 8, kernel_size=5)
        self.fe_conv_3 = double_conv_block(8, 16)
        self.fe_conv_4 = double_conv_block(16, 8)
        self.fe_conv_5 = double_conv_block(8, 4)

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

        # Attention Gates
        if 'gates' in self.use_attention:
            self.attn_1 = AttentionGate(32)
            self.attn_2 = AttentionGate(16)
            self.attn_3 = AttentionGate(8)

        # Self-Attention layers
        if 'bottleneck' in self.use_attention:
            self.self_attn_bottleneck = SelfAttention(64)

        # Output
        self.out = nn.Conv1d(in_channels=8, out_channels=out_channels, kernel_size=1)

    def feature_extract_ppg(self, x):

        # Input
        bs, in_ch, seq_len = x.shape
        hyp   = x[:,0].view((bs, 1, -1))               # [bs, 1, 600]
        spo2  = x[:,1].view((bs, 1, -1))               # [bs, 1, 600]
        pleth = x[:,2:].view((bs, 1, -1))              # [bs, 1, 256*600]

        # PPG feature extraction
        pleth = self.fe_conv_1(pleth)                  # [bs,  4, 256*600]
        pleth = self.fe_conv_2(self.fe_pool(pleth))    # [bs,  8,  64*600]
        pleth = self.fe_conv_3(self.fe_pool(pleth))    # [bs, 16,  16*600]
        pleth = self.fe_conv_4(self.fe_pool(pleth))    # [bs,  8,   4*600]
        pleth = self.fe_conv_5(self.fe_pool(pleth))    # [bs,  4,   1*600]

        # Concatinate
        return torch.cat([hyp, pleth], dim=1)          # [bs, 5, 600]
    
    def encode(self, x):

        enc_1 = self.enc_conv_1(x)                     # [bs, 8,   600]
        enc_2 = self.enc_conv_2(self.max_pool(enc_1))  # [bs, 16,  300]
        enc_3 = self.enc_conv_3(self.max_pool(enc_2))  # [bs, 32,  150]
        x = self.enc_conv_4(self.max_pool(enc_3))      # [bs, 64,  75]

        return x, [enc_1, enc_2, enc_3]

    def bottleneck(self, x):
        
        if 'bottleneck' in self.use_attention:
            x = self.self_attn_bottleneck(x)

        return x
    
    def decode(self, x, encs):
        enc_1, enc_2, enc_3 = encs

        x = self.up_transpose_1(x)                     # [bs, 32, 150]
        if 'gates' in self.use_attention:
            enc_3 = self.attn_1(x, enc_3)
        x = torch.cat([enc_3, x], 1)                   # [bs, 64, 150]
        x = self.dec_conv_1(x)                         # [bs, 32, 150]

        x = self.up_transpose_2(x)                     # [bs, 16, 300]
        if 'gates' in self.use_attention:
            enc_2 = self.attn_2(x, enc_2)
        x = torch.cat([enc_2, x], 1)                   # [bs, 32, 300]
        x = self.dec_conv_2(x)                         # [bs, 16, 300]

        x = self.up_transpose_3(x)                     # [bs,  8, 600]
        if 'gates' in self.use_attention:
            enc_1 = self.attn_3(x, enc_1)
        x = torch.cat([enc_1, x], 1)                   # [bs, 16, 600]
        x = self.dec_conv_3(x)                         # [bs,  8, 600]

        return x

    def forward(self, x):                              # [bs, 2+256, 600]

        x = self.feature_extract_ppg(x)
        x, encs = self.encode(x)
        x = self.bottleneck(x)
        x = self.decode(x, encs)

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


class AttentionGate(nn.Module):
    def __init__(self, channels):
        super(AttentionGate, self).__init__()

        self.W_x = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, padding=0),
            nn.BatchNorm1d(channels)
        )

        self.W_enc = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, padding=0),
            nn.BatchNorm1d(channels)
        )

        self.relu = nn.ReLU(inplace=True)

        self.out = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x, enc):
        proj_x   = self.W_x(x)
        proj_enc = self.W_enc(enc)

        gate = self.relu(proj_x + proj_enc)
        gate = self.out(gate)

        return enc * gate


class SelfAttention(nn.Module):
    def __init__(self, dims):
        super(SelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dims, num_heads=1, batch_first=True)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)     # nn.MultiheadAttention uses [bs, seq_len, channels]
        x, _ = self.attn(x, x, x)  # Use x three times for self-attention
        return x.permute(0, 2, 1)  # Convert back to [bs, channels, seq_length]