import numpy as np
import torch
from typing import Optional
import math

# ------ FROM TORCH YIN --------
def estimate(
    signal,
    sample_rate: int = 44100,
    pitch_min: float = 20.0,
    pitch_max: float = 20000.0,
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
    # frames: n_frames, frame_length
    # compute the frame-wise autocorrelation using the FFT
    fft_size = int(2 ** (-int(-math.log(frames.shape[-1]) // math.log(2)) + 1))
    fft = torch.fft.rfft(frames, fft_size, dim=-1)
    corr = torch.fft.irfft(fft * fft.conj())[..., :tau_max]

    # difference function (equation 6)
    sqrcs = torch.nn.functional.pad((frames * frames).cumsum(-1), [1, 0])
    corr_0 = sqrcs[..., -1:]
    corr_tau = sqrcs.flip(-1)[..., :tau_max] - sqrcs[..., :tau_max]
    diff = corr_0 + corr_tau - 2 * corr

    #print(diff.device, torch.arange(1, diff.shape[-1]).device)

    # cumulative mean normalized difference function (equation 8)
    return (
        diff[..., 1:]
        * torch.arange(1, diff.shape[-1], device=diff.device)
        / torch.clamp(diff[..., 1:].cumsum(-1), min=1e-5)
    )

@torch.jit.script
def _search(cmdf: torch.Tensor, tau_max: int, threshold: float) -> torch.Tensor:
    # mask all periods after the first cmdf below the threshold
    # if none are below threshold (argmax=0), this is a non-periodic frame
    first_below = (cmdf < threshold).int().argmax(-1, keepdim=True)
    first_below = torch.where(first_below > 0, first_below, tau_max)
    beyond_threshold = torch.arange(cmdf.shape[-1], device=cmdf.device) >= first_below

    # mask all periods with upward sloping cmdf to find the local minimum
    increasing_slope = torch.nn.functional.pad(cmdf.diff() >= 0.0, [0, 1], value=1.0)

    # find the first period satisfying both constraints
    return (beyond_threshold & increasing_slope).int().argmax(-1)
    

# ----------------------------------

class ExcitationModule(torch.nn.Module):
    def __init__(self, fs, encoding_ratio=1024, rms_thresh = 0.1, is_remove_above_nyquist=True):
        super().__init__()
        
        self.fs = fs
        self.encoding_ratio = encoding_ratio
        self.rms_thresh = rms_thresh
        self.is_remove_above_nyquist = is_remove_above_nyquist

        self.amplitudes = torch.nn.Parameter(
            1. / torch.arange(1, 150 + 1).float(), requires_grad=False)
        
        self.ratio = torch.nn.Parameter(torch.tensor([1.0]).float(), requires_grad=False)
        self.n_harmonics = len(self.amplitudes)
        
    def forward(self, audio, pitch_mult, initial_phase: Optional[torch.Tensor]=None):

        pitch = self.get_pitch(audio, self.encoding_ratio).unsqueeze(-1) 
        pitch = self.upsample(pitch, self.encoding_ratio) * (pitch_mult.unsqueeze(-1).to(pitch))

        if initial_phase is None:
            initial_phase = torch.zeros(pitch.shape[0], 1, 1).to(pitch)
        
        noise_mask = (pitch == 0).detach().float()
        noise_mask *= torch.randn(noise_mask.shape).to(noise_mask)
        #noise_mask *= (100 * torch.normal(mean=0.0, std=0.003 ** 2.0, size=noise_mask.shape)).to(pitch)
        f0 = pitch.detach()

        # harmonic synth
        theta = (2 * np.pi * f0 / self.fs).float().clone().detach()
        phase  = theta.cumsum(dim=1) + initial_phase
        phases = phase * torch.arange(1, self.n_harmonics + 1).to(phase)

        # anti-aliasing
        amplitudes = self.amplitudes * self.ratio
        if self.is_remove_above_nyquist:
            amp = self.remove_above_nyquist(amplitudes.to(phase), f0, self.fs)
        else:
            amp = amplitudes.to(phase)
        
        # signal
        signal = (torch.sin(phases) * amp).sum(-1, keepdim=True)
        #rms_val = self.get_rms_val(audio, signal.squeeze(-1), self.encoding_ratio, self.rms_thresh).unsqueeze(-1)
        #signal += (noise_mask * rms_val)
        signal = signal.squeeze(-1)
        
        ex = signal
 
        return signal

    def remove_above_nyquist(self, amplitudes, pitch, sampling_rate: int):
        n_harm = amplitudes.shape[-1]
        pitches = pitch * torch.arange(1, n_harm + 1).to(pitch)
        aa = (pitches < sampling_rate / 2).float() + 1e-7
        return amplitudes * aa

    def upsample(self, signal, factor: int):
        signal = signal.permute(0, 2, 1)
        signal = torch.nn.functional.interpolate(signal, size=int(signal.shape[-1] * factor))
        return signal.permute(0, 2, 1)

    def frame_rms_batched(self, input_signal, frame_length: int):

        if frame_length <= 0:
            raise ValueError("Frame length must be a positive integer.")

        batch_size, signal_length = input_signal.shape
        num_frames = signal_length // frame_length
        input_frames = input_signal.view(batch_size, num_frames, frame_length)

        rms_values = torch.sqrt(torch.mean(input_frames**2, dim=2))

        return rms_values

    def get_rms_val(self, input_batch, excitation_batch, encoding_ratio: int, threshold: float = 0.1, eps: float = 10e-5):
        rms_in = self.upsample(self.frame_rms_batched(input_batch, encoding_ratio).unsqueeze(-1), encoding_ratio)
        rms_ex = self.upsample(self.frame_rms_batched(excitation_batch, encoding_ratio).unsqueeze(-1), encoding_ratio)
        rms_val = (rms_in + eps) / (rms_ex + eps)
        rms_val[rms_val < threshold] = 0
        return rms_val.squeeze(-1)

    def get_pitch(self, x, encoding_ratio: int = 1024, fs: int = 44100, pitch_min: float = 70.0):
        desired_num_frames = x.shape[-1] / encoding_ratio
        tau_max = int(fs / pitch_min)
        frame_length = 2 * tau_max
        frame_stride = (x.shape[-1] - frame_length) / (desired_num_frames - 1) / fs
        return estimate(x, sample_rate=fs, pitch_min=pitch_min, pitch_max=400.0, frame_stride=frame_stride)