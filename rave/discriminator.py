import torch
import torch.nn as nn

from .mpd import MultiPeriodDiscriminator
from .mrd import MultiResolutionDiscriminator
from omegaconf import OmegaConf

class Discriminator_new(nn.Module):
    def __init__(self):
        super(Discriminator_new, self).__init__()
        self.MRD = MultiResolutionDiscriminator()
        self.MPD = MultiPeriodDiscriminator()

    def forward(self, x):
        return self.MRD(x), self.MPD(x)

