import torch
from torch import nn
import torch.nn.functional as F
from .utils import SNConv2d, Attention
from torch.nn import Conv2d


class BasicBlockEnc(nn.Module):
    def __init__(self, in_channel, out_channel, downsample, preactivation):
        super().__init__()
        self.in_channel, self.out_channel = in_channel, out_channel
        self.hidden_channel = out_channel
        self.conv1 = SNConv2d(in_channel, self.hidden_channel, kernel_size=3, padding=1)
        self.conv2 = SNConv2d(self.hidden_channel, out_channel, kernel_size=3, padding=1)
        self.downsample = nn.AvgPool2d(2) if downsample else None
        self.preactivation = preactivation
        self.learnable_sc = True if (in_channel != out_channel) or downsample else False
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        if self.learnable_sc:
            self.conv_sc = SNConv2d(in_channel, out_channel,
                                    kernel_size=1, padding=0)
        self.act = nn.ReLU()

    def shrotcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x):
        if self.preactivation:
            h = F.relu(self.bn1(x))
        else:
            h = x
        h = self.conv1(h)
        h  = self.conv2(self.act(self.bn2(h)))
        if self.downsample:
            h = self.downsample(h)
        return h + self.shrotcut(x)


class Encoder(nn.Module):
    def __init__(self, latent_dim, in_channels, out_channels, downsample, attention):
        super().__init__()
        self.blocks = []
        for i in range(len(out_channels)):
            enc_block = BasicBlockEnc(
                in_channels[i],
                out_channels[i],
                downsample[i],
                i > 0
            )
            self.blocks += [enc_block]
            if attention[i]:
                self.blocks += [Attention(out_channels[i])]

        self.blocks = nn.ModuleList(self.blocks)
        self.linear = nn.Linear(out_channels[-1], 2 * latent_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        h = x
        for block in self.blocks:
            h = block(h)
        h = torch.sum(self.activation(h), [2, 3])
        out = self.linear(h)
        return out
