import torch
import torch.nn as nn

from .mpd import MultiPeriodDiscriminator
from .mrd import MultiResolutionDiscriminator
from omegaconf import OmegaConf

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.MRD = MultiResolutionDiscriminator()
        self.MPD = MultiPeriodDiscriminator()

    def forward(self, x):
        return self.MRD(x), self.MPD(x)

