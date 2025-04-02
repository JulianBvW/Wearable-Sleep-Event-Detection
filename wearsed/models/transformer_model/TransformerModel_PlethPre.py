'''
Transformer model for WearSED
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer


class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        in_channels = 1 + 1 + 5 + 8  # Hypno + SpO2 + Statistical PPG + VAE PPG

        # PPG
        self.se_attention = SE_Block()

        # Transformer
        self.dim_upscale = nn.Conv1d(in_channels=in_channels, out_channels=in_channels*8, kernel_size=3, padding=1)
        self.positional_encoding = Summer(PositionalEncoding1D(in_channels*8))
        encoder_layer = nn.TransformerEncoderLayer(d_model=in_channels*8, nhead=4, batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.dim_downscale = nn.Conv1d(in_channels=in_channels*8, out_channels=in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

        # Output
        self.out = nn.Conv1d(in_channels=in_channels, out_channels=1, kernel_size=3, padding=1)

    def transformer(self, x):                 # [bs,  15, 600]
        x = self.relu(self.dim_upscale(x))    # [bs, 120, 600]
        x = x.permute(0, 2, 1)                # [bs, 600, 120] - Transformer needs format [bs, seq_len, channels]
        x = self.positional_encoding(x)       # [bs, 600, 120]
        x = self.transformer_enc(x)           # [bs, 600, 120]
        x = x.permute(0, 2, 1)                # [bs, 120, 600] - Permute back to [bs, channels, seq_len]
        x = self.relu(self.dim_downscale(x))  # [bs,  15, 600]
        return x

    def forward(self, x):

        # Input
        bs, in_ch, seq_len = x.shape            # [bs, 15, 600]
        hyp  = x[:,0].view((bs, 1, -1))         # [bs,  1, 600]
        spo2 = x[:,1].view((bs, 1, -1))         # [bs,  1, 600]
        ppg  = x[:,2:].view((bs, 13, -1))       # [bs, 13, 600]

        # Preprocess PPG
        ppg = self.se_attention(ppg)            # [bs, 13, 600]

        # Transformer
        x = torch.cat([hyp, spo2, ppg], dim=1)  # [bs, 15, 600]
        x = self.transformer(x)                 # [bs, 15, 600]

        # Output
        out = self.out(x).squeeze(1)            # [bs, 600]
        return out

class SE_Block(nn.Module):
    def __init__(self, in_channels=13, bottleneck_channels=6):
        super(SE_Block, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, bottleneck_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, in_ch, seq_len = x.shape               # [bs, 13, 600]
        y = self.squeeze(x).view(bs, in_ch)        # [bs, 13]
        y = self.excitation(y).view(bs, in_ch, 1)  # [bs, 13, 1]
        return x * y                               # [bs, 13, 600]
