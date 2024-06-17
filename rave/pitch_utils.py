import librosa
import numpy as np
import math
import torch
from scipy.fft import dct, idct
import json
import os
import argparse
import torch.nn.functional as F

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

def get_pitch(x, block_size: int, fs: int=44100, pitch_min: float=70.0, pitch_max: float=400.0):
    desired_num_frames = x.shape[-1] / block_size
    tau_max = int(fs / pitch_min)
    frame_length = 2 * tau_max
    frame_stride = (x.shape[-1] - frame_length) / (desired_num_frames - 1) / fs
    f0 = estimate(x, sample_rate=fs, pitch_min=pitch_min, pitch_max=pitch_max, frame_stride=frame_stride)
    return f0 

def one_hot(a, num_classes: int):
    a_flat = a.reshape(-1)
    one_hot_flat = torch.eye(num_classes).to(a)
    one_hot_flat = one_hot_flat[a_flat]
    shape = a.shape
    one_hot_batched = one_hot_flat.reshape(shape[0], shape[1], num_classes)
    return one_hot_batched

def extract_utterance_log_f0(y, sr: int, frame_len_samples: int, voiced_prob_cutoff: float=0.2):
    f0 = get_pitch(y, frame_len_samples)
    f0[f0 == 0] = float('nan')
    log_f0 = torch.log(f0)
    return log_f0, f0

def quantize_f0_norm(y, f0_median, f0_std, fs: int, win_length: int, norm_mode: str ='abs'):
    desired_num_frames = int(y.shape[-1] / 1024)
    utt_log_f0, f0 = extract_utterance_log_f0(y, fs, win_length)
    #utt_log_f0 = utt_log_f0[~torch.isnan(utt_log_f0)]
    
    #if utt_log_f0.nelement() == 0:
    #    utt_log_f0 = torch.zeros(128).to(y)
        
    #utt_log_f0 = F.interpolate(utt_log_f0.unsqueeze(0).unsqueeze(0), size=desired_num_frames, mode='linear')[0, 0, :]
    if norm_mode == 'abs':
        log_f0_norm = (utt_log_f0 - torch.log(torch.tensor([40]).to(y))) / (torch.log(torch.tensor([400]).to(y) - torch.log(torch.tensor([40]).to(y))))
    else:
        log_f0_norm = ((utt_log_f0 - f0_median) / f0_std) / 4.0
    return log_f0_norm, f0

def get_f0_norm(y, f0_median, f0_std, fs: int, win_length: int, mult: float=1.0, scale: float=0.0, num_f0_bins: int=256):
    log_f0_norm, f0 = quantize_f0_norm(y, f0_median, f0_std, fs, win_length)
    log_f0_norm += 0.5
    log_f0_norm = log_f0_norm * mult + scale
    bins = torch.linspace(0, 1, num_f0_bins+1).to(y)
    f0_one_hot_idxs = torch.bucketize(log_f0_norm, bins, right=True) - 1
    f0_one_hot = one_hot(f0_one_hot_idxs, num_f0_bins+1)
    return f0_one_hot, log_f0_norm

def extract_f0_median_std(wav, fs: int, win_length: int):
    log_f0_vals, _ = extract_utterance_log_f0(wav, fs, win_length)
    log_f0_vals = log_f0_vals[~torch.isnan(log_f0_vals)]
    log_f0_median = torch.median(log_f0_vals)
    log_f0_std = torch.std(log_f0_vals)
    return log_f0_median, log_f0_std

def calculate_stats(root_folder):
    stats_dict = {}
    
    # Iterate over each subfolder in the root folder
    for subdir in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir)
        print("Calculating stats for:", subdir)
        
        if os.path.isdir(subdir_path):
            medians = []
            stds = []
            
            # Iterate over each file in the subfolder
            for filename in os.listdir(subdir_path):
                if filename.endswith('.flac'):
                    file_path = os.path.join(subdir_path, filename)

                    # Load the audio file
                    audio, fs = librosa.load(file_path, sr=44100, mono=True)
                    
                    # Calculate the length of the audio file in seconds
                    src_f0_median, src_f0_std = extract_f0_median_std(torch.tensor(audio),
                                                                      fs,
                                                                      512,
                                                                      100)
                    # Add the length to the list
                    if torch.isnan(src_f0_median) or torch.isnan(src_f0_std):
                        break
                    else:
                        medians.append(src_f0_median)
                        stds.append(src_f0_std)
            
            # Calculate the mean length of audio files in the current subfolder
            mean_median = float(np.mean(medians))
            mean_std = float(np.mean(stds))
            print("median:", mean_median, "mean_std", mean_std)
            m_s_dict = {'mean': mean_median, 'std': mean_std}
                
            stats_dict[subdir] = m_s_dict
                
    return stats_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str, help='Path to the root folder containing subfolders with audio files')
    
    args = parser.parse_args()
    root_folder = args.root_folder
    
    speaker_stats = calculate_stats(root_folder)
    
    # Print the dictionary to see the results
    print(speaker_stats)
    
    # Save the results to a file
    with open('speaker_stats.json', 'w') as json_file:
        json.dump(speaker_stats, json_file, indent=4)

if __name__ == '__main__':
    main()