import numpy as np
import torch
import torchyin

def remove_above_nyquist(amplitudes, pitch, sampling_rate):
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

def get_rms_val(input_batch, excitation_batch, block_size, eps=10e-5):
    rms_in = upsample(frame_rms_batched(input_batch, block_size).unsqueeze(-1), block_size)
    rms_ex = upsample(frame_rms_batched(excitation_batch, block_size).unsqueeze(-1), block_size)
    rms_val = (rms_in + eps) / (rms_ex + eps)
    return rms_val.squeeze(-1)

def get_pitch(x, block_size, fs=48000, pitch_min=60):
    desired_num_frames = x.shape[-1] / block_size
    tau_max = int(fs / pitch_min)
    frame_length = 2 * tau_max
    frame_stride = (x.shape[-1] - frame_length) / (desired_num_frames - 1) / fs
    return torchyin.estimate(x, sample_rate=fs, pitch_min=pitch_min, pitch_max=500, frame_stride=frame_stride)

class ExcitationModule(torch.nn.Module):
    def __init__(self, fs, block_size, is_remove_above_nyquist=True):
        super().__init__()
        self.fs = fs
        self.is_remove_above_nyquist = is_remove_above_nyquist
        self.register_buffer("block_size", torch.tensor(block_size))

        self.amplitudes = torch.nn.Parameter(
            1. / torch.arange(1, 150 + 1).float(), requires_grad=False)
        
        self.ratio = torch.nn.Parameter(torch.tensor([1.0]).float(), requires_grad=False)
        self.n_harmonics = len(self.amplitudes)
        
    def forward(self, pitch, initial_phase=None):

        if initial_phase is None:
            initial_phase = torch.zeros(pitch.shape[0], 1, 1).to(pitch)
        
        mask = (pitch == 0).detach().float()
        mask *= torch.randn(mask.shape).to(pitch)
        f0 = pitch.detach()

        # harmonic synth
        phase  = torch.cumsum(2 * np.pi * f0 / self.fs, axis=1) + initial_phase
        phases = phase * torch.arange(1, self.n_harmonics + 1).to(phase)

        # anti-aliasing
        amplitudes = self.amplitudes * self.ratio
        if self.is_remove_above_nyquist:
            amp = remove_above_nyquist(amplitudes.to(phase), f0, self.fs)
        else:
            amp = amplitudes.to(phase)
        
        # signal
        signal = (torch.sin(phases) * amp).sum(-1, keepdim=True)
        signal += mask
        signal = signal.squeeze(-1) / torch.max(signal)
        
        # phase
        final_phase = phase[:, -1:, :] % (2 * np.pi)
 
        return signal, final_phase.detach()