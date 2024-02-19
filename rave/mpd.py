import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

class DiscriminatorP(nn.Module):
    def __init__(self, period):
        super(DiscriminatorP, self).__init__()

        self.LRELU_SLOPE = 0.2
        self.period = period
        self.use_spectral_norm = False

        kernel_size = 5
        stride = 3
        norm_f = weight_norm if self.use_spectral_norm == False else spectral_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 64, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            norm_f(nn.Conv2d(64, 128, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            norm_f(nn.Conv2d(128, 256, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            norm_f(nn.Conv2d(256, 512, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1,padding=(kernel_size // 2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # Convert 1D time domain signal to 2D.
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad

        # Reshape to match the given period in one dimension.
        x = x.view(b, c, t // self.period, self.period)

        # Pass reshaped signal through convolutional disciminator network.
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)

        # fmap is list of discriminator's layer-wise feature map outputs
        # (for feature matching loss in HiFi-GAN; unused in UnivNet).
        fmap.append(x)

        # x is 1D tensor of frame-wise discriminator scores (0-1).
        x = torch.flatten(x, 1, -1)

        return fmap, x


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()

        self.periods = [2, 3, 5, 7, 11, 17, 23, 37]

        # Initialize discriminators for prime numbered periods.
        self.discriminators = nn.ModuleList(
            [DiscriminatorP(period) for period in self.periods]
        )

    def forward(self, x):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x))

        return ret  # [(feat, score), (feat, score), (feat, score), (feat, score), (feat, score)]
