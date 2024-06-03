# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn

from rave.pqmf import CachedPQMF as PQMF

from functools import partial
from typing import Callable, Optional, Sequence, Union

import cached_conv as cc
import numpy as np
import torch
import gin
import torch.nn as nn
from torch.nn.utils import weight_norm
from torchaudio.transforms import Spectrogram

def normalization(module: nn.Module, mode: str = 'identity'):
    if mode == 'identity':
        return module
    elif mode == 'weight_norm':
        return weight_norm(module)
    else:
        raise Exception(f'Normalization mode {mode} not supported')


class Residual(nn.Module):

    def __init__(self, module, cumulative_delay=0):
        super().__init__()
        additional_delay = module.cumulative_delay
        self.aligned = cc.AlignBranches(
            module,
            nn.Identity(),
            delays=[additional_delay, 0],
        )
        self.cumulative_delay = additional_delay + cumulative_delay

    def forward(self, x):
        x_net, x_res = self.aligned(x)
        return x_net + x_res


class ResidualLayer(nn.Module):

    def __init__(
        self,
        dim,
        kernel_size,
        dilations,
        cumulative_delay=0,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2)):
        super().__init__()
        net = []
        cd = 0
        for d in dilations:
            net.append(activation(dim))
            net.append(
                normalization(
                    cc.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        dilation=d,
                        padding=cc.get_padding(kernel_size, dilation=d),
                        cumulative_delay=cd,
                    )))
            cd = net[-1].cumulative_delay
        self.net = Residual(
            cc.CachedSequential(*net),
            cumulative_delay=cumulative_delay,
        )
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        return self.net(x)


class DilatedUnit(nn.Module):

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        dilation: int,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2)
    ) -> None:
        super().__init__()
        net = [
            activation(dim),
            normalization(
                cc.Conv1d(dim,
                          dim,
                          kernel_size=kernel_size,
                          dilation=dilation,
                          padding=cc.get_padding(
                              kernel_size,
                              dilation=dilation,
                          ))),
            activation(dim),
            normalization(cc.Conv1d(dim, dim, kernel_size=1)),
        ]

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = net[1].cumulative_delay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):

    def __init__(self,
                 dim,
                 kernel_size,
                 dilations_list,
                 cumulative_delay=0) -> None:
        super().__init__()
        layers = []
        cd = 0

        for dilations in dilations_list:
            layers.append(
                ResidualLayer(
                    dim,
                    kernel_size,
                    dilations,
                    cumulative_delay=cd,
                ))
            cd = layers[-1].cumulative_delay

        self.net = cc.CachedSequential(
            *layers,
            cumulative_delay=cumulative_delay,
        )
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        return self.net(x)

@gin.configurable
class ResidualStack(nn.Module):

    def __init__(self,
                 dim,
                 kernel_sizes,
                 dilations_list,
                 cumulative_delay=0) -> None:
        super().__init__()
        blocks = []
        for k in kernel_sizes:
            blocks.append(ResidualBlock(dim, k, dilations_list))
        self.net = cc.AlignBranches(*blocks, cumulative_delay=cumulative_delay)
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        x = self.net(x)
        x = torch.stack(x, 0).sum(0)
        return x


class UpsampleLayer(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        ratio,
        cumulative_delay=0,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2)):
        super().__init__()
        net = [activation(in_dim)]
        if ratio > 1:
            net.append(
                normalization(
                    cc.ConvTranspose1d(in_dim,
                                       out_dim,
                                       2 * ratio,
                                       stride=ratio,
                                       padding=ratio // 2)))
        else:
            net.append(
                normalization(
                    cc.Conv1d(in_dim, out_dim, 3, padding=cc.get_padding(3))))

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay + cumulative_delay * ratio

    def forward(self, x):
        return self.net(x)


def normalize_dilations(dilations: Union[Sequence[int],
                                         Sequence[Sequence[int]]],
                        ratios: Sequence[int]):
    if isinstance(dilations[0], int):
        dilations = [dilations for _ in ratios]
    return dilations

class SpeakerRAVE(nn.Module):

    def __init__(self, activation = lambda dim: nn.LeakyReLU(.2)):
        super().__init__()

        #self.pqmf = PQMF(100, 16)
        kernel_size = 3

        self.in_layer = normalization(
                cc.Conv1d(
                    16,
                    128,
                    kernel_size=kernel_size * 2 + 1,
                    padding=cc.get_padding(kernel_size * 2 + 1),
                ))

        r = 4
        num_channels = 128
        out_channels = 256
        d = 1

        self.layer2 = torch.nn.Sequential(Residual(
            DilatedUnit(dim=num_channels,
                        kernel_size=kernel_size,
                        dilation=d)),
            activation(num_channels),
            normalization(cc.Conv1d(num_channels,
                                    out_channels,
                                    kernel_size=2*r,
                                    stride=r,
                                    padding=cc.get_padding(2*r, r))))

        r = 4
        num_channels = 256
        out_channels = 256
        d = 3
        
        self.layer3 = torch.nn.Sequential(Residual(
            DilatedUnit(dim=num_channels,
                        kernel_size=kernel_size,
                        dilation=d)),
            activation(num_channels),
            normalization(cc.Conv1d(num_channels,
                                    out_channels,
                                    kernel_size=2*r,
                                    stride=r,
                                    padding=cc.get_padding(2*r, r))))

        r = 2
        num_channels = 256
        out_channels = 256
        d = 5
        
        self.layer4 = torch.nn.Sequential(Residual(
            DilatedUnit(dim=num_channels,
                        kernel_size=kernel_size,
                        dilation=d)),
            activation(num_channels),
            normalization(cc.Conv1d(num_channels,
                                    out_channels,
                                    kernel_size=2*r,
                                    stride=r,
                                    padding=cc.get_padding(2*r, r))))
    
        self.cat_layer = normalization(cc.Conv1d(out_channels,
                                                 out_channels,
                                                 kernel_size=1,
                                                 padding=cc.get_padding(1)))

        self.out_layer = normalization(cc.Conv1d(out_channels * 3,
                                                 768,
                                                 kernel_size=kernel_size,
                                                 padding=cc.get_padding(kernel_size)))

        self.activation = activation(768)

        attention_projection = 768
        attn_input = attention_projection * 3
        attn_output = attention_projection

        self.attention = nn.Sequential(
            nn.Conv1d(attn_input, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            cc.Conv1d(128, attn_output, kernel_size=1),
            nn.Softmax(dim=2),
        )

        self.bn5 = nn.BatchNorm1d(attention_projection*2)

        self.fc6 = nn.Linear(attention_projection*2, 256)
        self.bn6 = nn.BatchNorm1d(256)
        
        #network = [in_layer, layer2, layer3, layer4, cat_layer, out_layer]
        #self.network = cc.CachedSequential(*network)

        self.mp2 = torch.nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        #x = self.pqmf(x.unsqueeze(1))
        x = self.in_layer(x)
        x1 = self.layer2(x)
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)
        x4 = self.cat_layer(x3)

        x = torch.cat((self.mp2(x2), x3, x4), dim=1)
        
        x = self.out_layer(x)
        x = self.activation(x)

        t = x.size()[-1]

        global_x = torch.cat((x,
                              torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                              torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4, max=1e4)).repeat(1, 1, t)),
                              dim=1)

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4, max=1e4))

        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)

        return x