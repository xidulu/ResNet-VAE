import torch
from torch import nn
import torch.nn.functional as F
from .utils import SNConv2d, Attention
from torch.nn import Conv2d
from functools import partial

class BasicBlockDec(nn.Module):
    def __init__(self, in_channel, out_channel, upsample):
        super().__init__()
        self.in_channel, self.out_channel = in_channel, out_channel
        self.hidden_channel = out_channel
        self.conv1 = Conv2d(in_channel, self.hidden_channel, kernel_size=3, padding=1)
        self.conv2 = Conv2d(self.hidden_channel, out_channel, kernel_size=3, padding=1)
        self.upsample = partial(F.interpolate, scale_factor=2) if upsample else None
        self.learnable_sc = True if (in_channel != out_channel) or upsample else False
        if self.learnable_sc:
            self.conv_sc = Conv2d(in_channel, out_channel,
                                    kernel_size=1, padding=0)
        self.act = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def shortcut(self, x):
        if self.upsample:
            x = self.upsample(x)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return x

    def forward(self, x):
        h = self.act(self.bn1(x))
        if self.upsample:
            h = self.upsample(h)
        h = self.conv1(h)
        h = self.act(self.bn2(h))
        h = self.conv2(h)
        return h + self.shortcut(x)
        

class Decoder(nn.Module):
    def __init__(self, latent_dim, in_channels, out_channels, upsample, attention,
                 bottom_width=4, observed_channel=3):
        super().__init__()
        self.blocks = []
        self.bottom_width = bottom_width
        for i in range(len(out_channels)):
            dec_block = BasicBlockDec(
                in_channels[i],
                out_channels[i],
                upsample[i]
            )
            self.blocks += [dec_block]
            if attention[i]:
                self.blocks += [Attention(out_channels[i])]
        
        self.blocks = nn.ModuleList(self.blocks)
        self.linear = nn.Linear(latent_dim, in_channels[0] * self.bottom_width ** 2)
        self.activation = nn.ReLU()
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(out_channels[-1]),
            self.activation,
            nn.Conv2d(out_channels[-1], observed_channel, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        h = self.linear(x)
        h = h.view(h.shape[0], -1, self.bottom_width, self.bottom_width)
        for block in self.blocks:
            h = block(h)
        return self.output_layer(h)
