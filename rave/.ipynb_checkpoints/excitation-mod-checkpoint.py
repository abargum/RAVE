import numpy as np
import torch
import torchyin
from typing import Optional

# ------ FROM TORCH YIN --------
import numpy as np

def estimate(
    signal,
    sample_rate: float,
    pitch_min: float = 20,
    pitch_max: float = 20000,
    frame_stride: float = 0.01,
    threshold: float = 0.1,
) -> torch.Tensor:

    signal = torch.as_tensor(signal)

    # convert frequencies to samples, ensure windows can fit 2 whole periods
    tau_min = int(sample_rate / pitch_max)
    tau_max = int(sample_rate / pitch_min)
    frame_length = 2 * tau_max
    frame_stride = int(frame_stride * sample_rate)

    # compute the fundamental periods
    frames = _frame(signal, frame_length, frame_stride)
    cmdf = _diff(frames, tau_max)[..., tau_min:]
    tau = _search(cmdf, tau_max, threshold)

    # convert the periods to frequencies (if periodic) and output
    return torch.where(
        tau > 0,
        sample_rate / (tau + tau_min + 1).type(signal.dtype),
        torch.tensor(0, device=tau.device).type(signal.dtype),
    )


def _frame(signal: torch.Tensor, frame_length: int, frame_stride: int) -> torch.Tensor:
    # window the signal into overlapping frames, padding to at least 1 frame
    if signal.shape[-1] < frame_length:
        signal = torch.nn.functional.pad(signal, [0, frame_length - signal.shape[-1]])
    return signal.unfold(dimension=-1, size=frame_length, step=frame_stride)


def _diff(frames: torch.Tensor, tau_max: int) -> torch.Tensor:
    # compute the frame-wise autocorrelation using the FFT
    fft_size = 2 ** (-int(-np.log(frames.shape[-1]) // np.log(2)) + 1)
    fft = torch.fft.rfft(frames, fft_size, dim=-1)
    corr = torch.fft.irfft(fft * fft.conj())[..., :tau_max]

    # difference function (equation 6)
    sqrcs = torch.nn.functional.pad((frames * frames).cumsum(-1), [1, 0])
    corr_0 = sqrcs[..., -1:]
    corr_tau = sqrcs.flip(-1)[..., :tau_max] - sqrcs[..., :tau_max]
    diff = corr_0 + corr_tau - 2 * corr

    # cumulative mean normalized difference function (equation 8)
    return (
        diff[..., 1:]
        * torch.arange(1, diff.shape[-1], device=diff.device)
        / torch.maximum(
            diff[..., 1:].cumsum(-1),
            torch.tensor(1e-5, device=diff.device),
        )
    )

def _search(cmdf: torch.Tensor, tau_max: int, threshold: float) -> torch.Tensor:
    # mask all periods after the first cmdf below the threshold
    # if none are below threshold (argmax=0), this is a non-periodic frame
    first_below = (cmdf < threshold).int().argmax(-1, keepdim=True)
    first_below = torch.where(first_below > 0, first_below, tau_max)
    beyond_threshold = torch.arange(cmdf.shape[-1], device=cmdf.device) >= first_below

    # mask all periods with upward sloping cmdf to find the local minimum
    increasing_slope = torch.nn.functional.pad(cmdf.diff() >= 0.0, [0, 1], value=1)

    # find the first period satisfying both constraints
    return (beyond_threshold & increasing_slope).int().argmax(-1)
    

# ----------------------------------


def remove_above_nyquist(amplitudes, pitch, sampling_rate: int):
    n_harm = amplitudes.shape[-1]
    pitches = pitch * torch.arange(1, n_harm + 1).to(pitch)
    aa = (pitches < sampling_rate / 2).float() + 1e-7
    return amplitudes * aa

def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = torch.nn.functional.interpolate(signal, size=signal.shape[-1] * factor)
    return signal.permute(0, 2, 1)

def frame_rms_batched(input_signal, frame_length):

    if frame_length <= 0:
        raise ValueError("Frame length must be a positive integer.")

    batch_size, signal_length = input_signal.shape
    num_frames = signal_length // frame_length
    input_frames = input_signal.view(batch_size, num_frames, frame_length)

    rms_values = torch.sqrt(torch.mean(input_frames**2, dim=2))

    return rms_values

def get_rms_val(input_batch, excitation_batch, block_size, threshold=0.1, eps=10e-5):
    rms_in = upsample(frame_rms_batched(input_batch, block_size).unsqueeze(-1), block_size)
    rms_ex = upsample(frame_rms_batched(excitation_batch, block_size).unsqueeze(-1), block_size)
    rms_val = (rms_in + eps) / (rms_ex + eps)
    rms_val[rms_val < threshold] = 0
    return rms_val.squeeze(-1)

def get_pitch(x, block_size, fs=44100, pitch_min=60):
    desired_num_frames = x.shape[-1] / block_size
    tau_max = int(fs / pitch_min)
    frame_length = 2 * tau_max
    frame_stride = (x.shape[-1] - frame_length) / (desired_num_frames - 1) / fs
    return estimate(x, sample_rate=fs, pitch_min=pitch_min, pitch_max=800, frame_stride=frame_stride)

class ExcitationModule(torch.nn.Module):
    def __init__(self, fs, is_remove_above_nyquist=True):
        super().__init__()
        self.fs = fs
        self.is_remove_above_nyquist = is_remove_above_nyquist

        self.amplitudes = torch.nn.Parameter(
            1. / torch.arange(1, 150 + 1).float(), requires_grad=False)
        
        self.ratio = torch.nn.Parameter(torch.tensor([1.0]).float(), requires_grad=False)
        self.n_harmonics = len(self.amplitudes)
        
    def forward(self, pitch, initial_phase: Optional[torch.Tensor]=None):

        if initial_phase is None:
            initial_phase = torch.zeros(pitch.shape[0], 1, 1).to(pitch)
        
        noise_mask = (pitch == 0).detach().float()
        noise_mask *= torch.randn(noise_mask.shape).to(pitch)
        f0 = pitch.detach()

        # harmonic synth
        theta = torch.tensor(2 * np.pi * f0 / self.fs).float()
        phase  = theta.cumsum(dim=1) + initial_phase
        phases = phase * torch.arange(1, self.n_harmonics + 1).to(phase)

        # anti-aliasing
        amplitudes = self.amplitudes * self.ratio
        if self.is_remove_above_nyquist:
            amp = remove_above_nyquist(amplitudes.to(phase), f0, int(self.fs))
        else:
            amp = amplitudes.to(phase)
        
        # signal
        signal = (torch.sin(phases) * amp).sum(-1, keepdim=True)
        signal += noise_mask
        signal = signal.squeeze(-1) / torch.max(signal)
        
        # phase
        final_phase = phase[:, -1:, :] % (2 * np.pi)
 
        return signal, final_phase.detach()