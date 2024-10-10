import logging
import math
import os

logging.basicConfig(level=logging.INFO)
logging.info("library loading")
logging.info("DEBUG")
import torch
from absl import flags, app

torch.set_grad_enabled(False)

import cached_conv as cc
import gin
import nn_tilde
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from absl import flags

import rave
import rave.blocks
import rave.core
import rave.resampler

FLAGS = flags.FLAGS

flags.DEFINE_string('run',
                    default=None,
                    help='Path to the run to export',
                    required=True)
flags.DEFINE_bool('streaming',
                  default=False,
                  help='Enable the model streaming mode')
flags.DEFINE_float(
    'fidelity',
    default=.95,
    lower_bound=.1,
    upper_bound=.999,
    help='Fidelity to use during inference (Variational mode only)')
flags.DEFINE_bool(
    'stereo',
    default=False,
    help='Enable fake stereo mode (one encoding, double decoding')
flags.DEFINE_bool('ema_weights',
                  default=False,
                  help='Use ema weights if available')
flags.DEFINE_integer('sr',
                     default=None,
                     help='Optional resampling sample rate')


class ScriptedRAVE(nn_tilde.Module):

    def __init__(self,
                 pretrained: rave.RAVE,
                 stereo: bool,
                 fidelity: float = .95,
                 target_sr: bool = None) -> None:
        super().__init__()
        self.stereo = stereo

        self.pqmf = pretrained.pqmf
        self.encoder = pretrained.encoder
        self.decoder = pretrained.decoder
        self.speaker = pretrained.speaker

        self.sr = pretrained.sr

        self.resampler = None

        if target_sr is not None:
            if target_sr != self.sr:
                assert not target_sr % self.sr, "Incompatible target sampling rate"
                self.resampler = rave.resampler.Resampler(target_sr, self.sr)
                self.sr = target_sr

        self.full_latent_size = pretrained.latent_size

        self.is_using_adain = False
        for m in self.modules():
            if isinstance(m, rave.blocks.AdaptiveInstanceNormalization):
                self.is_using_adain = True
                break

        if self.is_using_adain and stereo:
            raise ValueError("Stereo mode not yet supported with AdaIN")

        self.register_attribute("learn_target", False)
        self.register_attribute("reset_target", False)
        self.register_attribute("learn_source", False)
        self.register_attribute("reset_source", False)

        self.register_buffer("latent_pca", pretrained.latent_pca)
        self.register_buffer("latent_mean", pretrained.latent_mean)
        self.register_buffer("fidelity", pretrained.fidelity)

        self.register_buffer("in_median", None)
        self.register_buffer("in_std", None)

        if isinstance(pretrained.encoder, rave.blocks.VariationalEncoder):
            self.latent_size = 320

        elif isinstance(pretrained.encoder, rave.blocks.DiscreteEncoder):
            self.latent_size = pretrained.encoder.num_quantizers

        elif isinstance(pretrained.encoder, rave.blocks.WasserteinEncoder):
            self.latent_size = pretrained.latent_size

        elif isinstance(pretrained.encoder, rave.blocks.SphericalEncoder):
            self.latent_size = pretrained.latent_size - 1

        else:
            raise ValueError(
                f'Encoder type {pretrained.encoder.__class__.__name__} not supported'
            )

        x_len = 2**14
        x = torch.zeros(1, 1, x_len)

        if self.resampler is not None:
            x = self.resampler.to_model_sampling_rate(x)

        x_m = x.clone() if self.pqmf is None else self.pqmf(x)

        z = self.encoder(x_m[:, :6, :])

        ratio_encode = x_len // z.shape[-1]

        channels = ["(L)", "(R)"] if stereo else ["(mono)"]

        self.fake_adain = rave.blocks.AdaptiveInstanceNormalization(0)

        """
        self.register_method(
            "encode",
            in_channels=1,
            in_ratio=1,
            out_channels=self.latent_size,
            out_ratio=ratio_encode,
            input_labels=['(signal) Input audio signal'],
            output_labels=[
                f'(signal) Latent dimension {i}'
                for i in range(self.latent_size)
            ],
        )
        self.register_method(
            "decode",
            in_channels=self.latent_size,
            in_ratio=ratio_encode,
            out_channels=2 if stereo else 1,
            out_ratio=1,
            input_labels=[
                f'(signal) Latent dimension {i}'
                for i in range(self.latent_size)
            ],
            output_labels=[
                f'(signal) Reconstructed audio signal {channel}'
                for channel in channels
            ],
        )
        """

        self.register_method(
            "forward",
            in_channels=1,
            in_ratio=1,
            out_channels=2 if stereo else 1,
            out_ratio=1,
            input_labels=['(signal) Input audio signal'],
            output_labels=[
                f'(signal) Reconstructed audio signal {channel}'
                for channel in channels
            ],
        )

    def post_process_latent(self, z):
        raise NotImplementedError

    def pre_process_latent(self, z):
        raise NotImplementedError

    def update_adain(self):
        for m in self.modules():
            if isinstance(m, rave.blocks.AdaptiveInstanceNormalization):
                m.learn_x.zero_()
                m.learn_y.zero_()

                if self.learn_target[0]:
                    m.learn_y.add_(1)
                if self.learn_source[0]:
                    m.learn_x.add_(1)

                if self.reset_target[0]:
                    m.reset_y()
                if self.reset_source[0]:
                    m.reset_x()

        self.reset_source = False,
        self.reset_target = False,

    def slice_windows(self, signal: torch.Tensor, frame_size: int, hop_size: int, window:str='none', pad:bool=True):
        """
        slice signal into overlapping frames
        pads end if (l_x - frame_size) % hop_size != 0
        Args:
            signal: [batch, n_samples]
            frame_size (int): size of frames
            hop_size (int): size between frames
        Returns:
            [batch, n_frames, frame_size]
        """
        _batch_dim, l_x = signal.shape
        remainder = (l_x - frame_size) % hop_size
        if pad:
            pad_len = 0 if (remainder == 0) else hop_size - remainder
            signal = F.pad(signal, (0, pad_len), 'constant')
        signal = signal[:, None, None, :] # adding dummy channel/height
        frames = F.unfold(signal, (1, frame_size), stride=(1, hop_size)) #batch, frame_size, n_frames
        frames = frames.permute(0, 2, 1) # batch, n_frames, frame_size
        if window == 'hamming':
            win = torch.hamming_window(frame_size)[None, None, :].to(frames.device)
            frames = frames * win
        return frames

    def yin_frame(self, audio_frame, sample_rate:int, pitch_min:float =50, pitch_max:float =2000, threshold:float=0.1):
        # audio_frame: (n_frames, frame_length)
        tau_min = int(sample_rate / pitch_max)
        tau_max = int(sample_rate / pitch_min)
        assert audio_frame.shape[-1] > tau_max
        
        cmdf = self._diff(audio_frame, tau_max)[..., tau_min:]
        tau = self._search(cmdf, tau_max, threshold)
    
        return torch.where(
                tau > 0,
                sample_rate / (tau + tau_min + 1).type(audio_frame.dtype),
                torch.tensor(0).type(audio_frame.dtype),
            )

    def estimate(
        self,
        signal,
        sample_rate: int = 44100,
        pitch_min: float = 20.0,
        pitch_max: float = 20000.0,
        frame_stride: float = 0.01,
        threshold:  float = 0.3,
    ) -> torch.Tensor:
    
        signal = torch.as_tensor(signal)
    
        # convert frequencies to samples, ensure windows can fit 2 whole periods
        tau_min = int(sample_rate / pitch_max)
        tau_max = int(sample_rate / pitch_min)
        frame_length = 2 * tau_max
        frame_stride = int(frame_stride * sample_rate)
    
        # compute the fundamental periods
        frames = self._frame(signal, frame_length, frame_stride)
        cmdf = self._diff(frames, tau_max)[..., tau_min:]
        tau = self._search(cmdf, tau_max, threshold)
    
        # convert the periods to frequencies (if periodic) and output
        return torch.where(
            tau > 0,
            sample_rate / (tau + tau_min + 1).type(signal.dtype),
            torch.tensor(0, device=tau.device).type(signal.dtype),
        )
    
    
    def _frame(self, signal: torch.Tensor, frame_length: int, frame_stride: int) -> torch.Tensor:
        # window the signal into overlapping frames, padding to at least 1 frame
        if signal.shape[-1] < frame_length:
            signal = torch.nn.functional.pad(signal, [0, frame_length - signal.shape[-1]])
        return signal.unfold(dimension=-1, size=frame_length, step=frame_stride)
    
    
    def _diff(self, frames: torch.Tensor, tau_max: int) -> torch.Tensor:
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
    
    def _search(self, cmdf: torch.Tensor, tau_max: int, threshold: float) -> torch.Tensor:
        # mask all periods after the first cmdf below the threshold
        # if none are below threshold (argmax=0), this is a non-periodic frame
        first_below = (cmdf < threshold).int().argmax(-1, keepdim=True)
        first_below = torch.where(first_below > 0, first_below, tau_max)
        beyond_threshold = torch.arange(cmdf.shape[-1], device=cmdf.device) >= first_below
    
        # mask all periods with upward sloping cmdf to find the local minimum
        increasing_slope = torch.nn.functional.pad(cmdf.diff() >= 0.0, [0, 1], value=1.0)
    
        # find the first period satisfying both constraints
        return (beyond_threshold & increasing_slope).int().argmax(-1)

    def get_pitch(self, x, block_size: int, fs: int=44100, pitch_min: float=50.0, pitch_max: float=500.0):
        desired_num_frames = x.shape[-1] / block_size
        tau_max = int(fs / pitch_min)
        frame_length = 2 * tau_max
        frame_stride = (x.shape[-1] - frame_length) / (desired_num_frames - 1) / fs
        f0 = self.estimate(x, sample_rate=fs, pitch_min=pitch_min, pitch_max=pitch_max, frame_stride=frame_stride)
        return f0 

    def extract_utterance_inference(self, y, sr: int, frame_len_samples: int):
        f0 = self.get_pitch(y, frame_len_samples)
        return f0

    def update_in_stats(self, f0_vals):
        current_median = torch.median(f0_vals)
        current_std = torch.std(f0_vals)

        if self.in_median is None:
            self.in_median = current_median
            self.in_std = current_std
        else:
            self.in_median = 0.1 * current_median + (1 - 0.1) * self.in_median
            self.in_std = 0.1 * current_std + (1 - 0.1) * self.in_std

    def encode(self, x):
        
        x = self.pqmf(x)
        z = self.encoder(x[:, :6, :])
        emb = self.speaker.repeat(z.shape[0], 1, z.shape[-1])
        z = torch.cat((z, emb), dim=1)
        
        return z

    def decode(self, z, f0, from_forward: bool = False):
        y, harm = self.decoder(z, f0, 512)
        y = self.pqmf.inverse(y)
        return y

    def forward(self, x):
        z = self.encode(x)
        
        #f0 = torch.ones(z.shape[0], z.shape[-1]) * 250

        f0 = self.extract_utterance_inference(x, self.sr, 512).squeeze(1)
        #f0[f0 == 0] = float('nan')
        f0 *= 1.7
        
        #f0_vals = f0[~torch.isnan(f0)]
        #self.update_in_stats(f0_vals)

        #print(self.in_median)

        #standardized_source_pitch = f0 - self.in_median / self.in_std
        #source_pitch = (standardized_source_pitch * 35.76 + 211.82)
        #source_pitch[torch.isnan(source_pitch)] = 0
        
        y = self.decode(z, f0)
        return y

    @torch.jit.export
    def get_learn_target(self) -> bool:
        return self.learn_target[0]

    @torch.jit.export
    def set_learn_target(self, learn_target: bool) -> int:
        self.learn_target = (learn_target, )
        return 0

    @torch.jit.export
    def get_learn_source(self) -> bool:
        return self.learn_source[0]

    @torch.jit.export
    def set_learn_source(self, learn_source: bool) -> int:
        self.learn_source = (learn_source, )
        return 0

    @torch.jit.export
    def get_reset_target(self) -> bool:
        return self.reset_target[0]

    @torch.jit.export
    def set_reset_target(self, reset_target: bool) -> int:
        self.reset_target = (reset_target, )
        return 0

    @torch.jit.export
    def get_reset_source(self) -> bool:
        return self.reset_source[0]

    @torch.jit.export
    def set_reset_source(self, reset_source: bool) -> int:
        self.reset_source = (reset_source, )
        return 0


class VariationalScriptedRAVE(ScriptedRAVE):

    def post_process_latent(self, z):
        z = self.encoder.reparametrize(z)[0]
        z = z - self.latent_mean.unsqueeze(-1)
        z = F.conv1d(z, self.latent_pca.unsqueeze(-1))
        z = z[:, :self.latent_size]
        return z

    def pre_process_latent(self, z):
        noise = torch.randn(
            z.shape[0],
            self.full_latent_size - self.latent_size,
            z.shape[-1],
        ).type_as(z)
        z = torch.cat([z, noise], 1)
        z = F.conv1d(z, self.latent_pca.T.unsqueeze(-1))
        z = z + self.latent_mean.unsqueeze(-1)
        return z


class DiscreteScriptedRAVE(ScriptedRAVE):

    def post_process_latent(self, z):
        z = self.encoder.rvq.encode(z)
        return z.float()

    def pre_process_latent(self, z):
        z = torch.clamp(z, 0,
                        self.encoder.rvq.layers[0].codebook_size - 1).long()
        z = self.encoder.rvq.decode(z)
        if self.encoder.noise_augmentation:
            noise = torch.randn(z.shape[0], self.encoder.noise_augmentation,
                                z.shape[-1]).type_as(z)
            z = torch.cat([z, noise], 1)
        return z


class WasserteinScriptedRAVE(ScriptedRAVE):

    def post_process_latent(self, z):
        return z

    def pre_process_latent(self, z):
        if self.encoder.noise_augmentation:
            noise = torch.randn(z.shape[0], self.encoder.noise_augmentation,
                                z.shape[-1]).type_as(z)
            z = torch.cat([z, noise], 1)
        return z


class SphericalScriptedRAVE(ScriptedRAVE):

    def post_process_latent(self, z):
        return rave.blocks.unit_norm_vector_to_angles(z)

    def pre_process_latent(self, z):
        return rave.blocks.angles_to_unit_norm_vector(z)


def main(argv):
    cc.use_cached_conv(True)

    logging.info("building rave")

    gin.parse_config_file(os.path.join(FLAGS.run, "config.gin"))
    checkpoint = rave.core.search_for_run(FLAGS.run)

    pretrained = rave.RAVE()
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint, map_location='cpu')
        if FLAGS.ema_weights and "EMA" in checkpoint["callbacks"]:
            pretrained.load_state_dict(
                checkpoint["callbacks"]["EMA"],
                strict=False,
            )
        else:
            pretrained.load_state_dict(
                checkpoint["state_dict"],
                strict=False,
            )
    else:
        print("No checkpoint found, RAVE will remain randomly initialized")
    pretrained.eval()

    if isinstance(pretrained.encoder, rave.blocks.VariationalEncoder):
        script_class = VariationalScriptedRAVE
    elif isinstance(pretrained.encoder, rave.blocks.DiscreteEncoder):
        script_class = DiscreteScriptedRAVE
    elif isinstance(pretrained.encoder, rave.blocks.WasserteinEncoder):
        script_class = WasserteinScriptedRAVE
    elif isinstance(pretrained.encoder, rave.blocks.SphericalEncoder):
        script_class = SphericalScriptedRAVE
    else:
        raise ValueError(f"Encoder type {type(pretrained.encoder)} "
                         "not supported for export.")

    logging.info("warmup pass")

    x = torch.zeros(1, 1, 2**14)
    pretrained(x)

    logging.info("optimize model")

    for m in pretrained.modules():
        if hasattr(m, "weight_g"):
            nn.utils.remove_weight_norm(m)
    logging.info("script model")
    
    scripted_rave = script_class(
        pretrained=pretrained,
        stereo=FLAGS.stereo,
        fidelity=FLAGS.fidelity,
        target_sr=FLAGS.sr,
    )

    logging.info("save model")
    model_name = os.path.basename(os.path.normpath(FLAGS.run))
    if FLAGS.streaming:
        model_name += "_streaming"
    if FLAGS.stereo:
        model_name += "_stereo"
    model_name += ".ts"

    scripted_rave.export_to_ts(os.path.join(FLAGS.run, model_name))

    logging.info(
        f"all good ! model exported to {os.path.join(FLAGS.run, model_name)}")


if __name__ == "__main__":
    app.run(main)