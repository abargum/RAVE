from typing import Optional

import torch
import torchaudio
from torch import nn

from .rough_cqt import CQT2010v2


class ConstantQTransform(nn.Module):
    """Constant Q-Transform."""

    hop_length: int
    """The number of samples between adjacent frame."""

    fmin: float
    """The minimum frequency."""

    bins: int
    """The number of output bins."""

    bins_per_octave: int
    """The number of frequency bins per octave."""

    sample_rate: int
    """The sampling rate."""

    def __init__(
        self,
        hop_length: int,
        fmin: float,
        bins: int,
        bins_per_octave: int,
        sample_rate: int,
    ):
        """Initializer.
        Args:
            hop_length: The number of samples between adjacent frame.
            fmin: The minimum frequency.
            bins: The number of output bins.
            bins_per_octave: The number of frequency bins per octave.
            sample_rate: The sampling rate.
        """
        super().__init__()
        # unknown `hop_length`
        # , since linguistic information is 50fps, hop_length could be 441
        self.hop_length = hop_length
        # fmin=32.7(C0)
        self.fmin = fmin
        # bins=191, bins_per_octave=24
        # , fmax = 2 ** (bins / bins_per_octave) * fmin
        #        = 2 ** (191 / 24) * 32.7
        #        = 8132.89

        self.bins = bins
        self.bins_per_octave = bins_per_octave
        self.sample_rate = sample_rate

        self.cqt = CQT2010v2(
            sample_rate,
            hop_length,
            fmin,
            n_bins=bins,
            bins_per_octave=bins_per_octave,
            trainable=False,
            output_format="Magnitude",
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply CQT on inputs.
        Args:
            inputs: [torch.float32; [B, T]], input speech signal.
        Returns:
            [torch.float32; [B, bins, T / hop_length]], CQT magnitudes.
        """
        return self.cqt(inputs[:, None])