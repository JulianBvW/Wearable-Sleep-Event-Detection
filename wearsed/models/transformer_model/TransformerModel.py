'''
Transformer model for WearSED
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncodingPermute1D, Summer


class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()

        # PPG
        self.feature_extract_ppg = MultiScaleCNN()
        self.se_attention = SE_Block()

        # Transformer
        self.dim_upscale = nn.Conv1d(in_channels=10, out_channels=10*8, kernel_size=3, padding=1)
        self.positional_encoding = Summer(PositionalEncodingPermute1D(10))
        encoder_layer = nn.TransformerEncoderLayer(d_model=10*8, nhead=4, batch_first=True)
        self.transformer_enc = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.dim_downscale = nn.Conv1d(in_channels=10*8, out_channels=10, kernel_size=3, padding=1)

        # Output
        self.out = nn.Conv1d(in_channels=10, out_channels=1, kernel_size=1)  # TODO Different kernel?

    def transformer(self, x):
        x = self.dim_upscale(x)      # [bs,  80, 600]
        x = x.permute(0, 2, 1)       # [bs, 600,  80] - Transformer needs format [bs, seq_len, channels]
        x = self.transformer_enc(x)  # [bs, 600,  80]
        x = x.permute(0, 2, 1)       # [bs,  80, 600] - Permute back to [bs, channels, seq_len]
        x = self.dim_downscale(x)    # [bs,  10, 600]
        return x

    def forward(self, x):

        # Input
        bs, in_ch, seq_len = x.shape
        hyp  = x[:,0].view((bs, 1, -1))                # [bs, 1, 600]
        spo2 = x[:,1].view((bs, 1, -1))                # [bs, 1, 600]
        ppg  = x[:,2:].view((bs, 1, -1))               # [bs, 1, 256*600]

        # Preprocess PPG
        ppg = self.feature_extract_ppg(ppg)            # [bs, 8, 600]
        ppg = self.se_attention(ppg)  # TODO optional  # [bs, 8, 600]

        # Transformer
        x = torch.cat([hyp, spo2, ppg], dim=1)         # [bs, 10, 600]
        x = self.positional_encoding(x)                # [bs, 10, 600]
        x = self.transformer(x)                        # [bs, 10, 600]

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

class MultiScaleCNN(nn.Module):
    def __init__(self):
        super(MultiScaleCNN, self).__init__()

        self.max_pool = nn.MaxPool1d(kernel_size=4, stride=4)

        self.small_conv_1 = double_conv_block(1, 4, kernel_size=5)
        self.small_conv_2 = double_conv_block(4, 8, kernel_size=5)
        self.small_conv_3 = double_conv_block(8, 16)
        self.small_conv_4 = double_conv_block(16, 8)
        self.small_conv_5 = double_conv_block(8, 4)

        self.large_conv_1 = double_conv_block(1, 4, kernel_size=15)
        self.large_conv_2 = double_conv_block(4, 8, kernel_size=15)
        self.large_conv_3 = double_conv_block(8, 16, kernel_size=15)
        self.large_conv_4 = double_conv_block(16, 8, kernel_size=15)
        self.large_conv_5 = double_conv_block(8, 4, kernel_size=15)

    def forward(self, x):
        x_small = self.small_conv_1(x)                         # [bs,  4, 256*600]
        x_small = self.small_conv_2(self.max_pool(x_small))    # [bs,  8,  64*600]
        x_small = self.small_conv_3(self.max_pool(x_small))    # [bs, 16,  16*600]
        x_small = self.small_conv_4(self.max_pool(x_small))    # [bs,  8,   4*600]
        x_small = self.small_conv_5(self.max_pool(x_small))    # [bs,  4,   1*600]

        x_large = self.large_conv_1(x)                         # [bs,  4, 256*600]
        x_large = self.large_conv_2(self.max_pool(x_large))    # [bs,  8,  64*600]
        x_large = self.large_conv_3(self.max_pool(x_large))    # [bs, 16,  16*600]
        x_large = self.large_conv_4(self.max_pool(x_large))    # [bs,  8,   4*600]
        x_large = self.large_conv_5(self.max_pool(x_large))    # [bs,  4,   1*600]

        return torch.cat([x_small, x_large], dim=1)            # [bs,  8,   1*600]
    
class SE_Block(nn.Module):
    def __init__(self, in_channels=8, bottleneck_channels=4):
        super(SE_Block, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, bottleneck_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, in_ch, seq_len = x.shape               # [bs, 8, 600]
        y = self.squeeze(x).view(bs, in_ch)        # [bs, 8]
        y = self.excitation(y).view(bs, in_ch, 1)  # [bs, 8, 1]
        return x * y                               # [bs, 8, 600]
