import logging
import math
import os

logging.basicConfig(level=logging.INFO)
logging.info("library loading")
logging.info("DEBUG")
import torch

import librosa

torch.set_grad_enabled(False)

import cached_conv as cc
import gin
import nn_tilde
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from absl import flags, app

import sys, os
sys.path.append(os.path.abspath('../'))
import rave
import rave.model
import rave.blocks
import rave.core
import rave.resampler
from rave.pitch_utils import get_f0_norm, get_f0_norm_fcpe, extract_f0_median_std_fcpe, extract_f0_median_std, extract_f0_median_std_inference, get_f0_norm_inference, slice_windows, yin_frame

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

        #self.pqmf_speaker = pretrained.pqmf
        self.speaker_encoder = pretrained.speaker_encoder
        
        #self.speaker_stat_dict = pretrained.global_speaker_dict
        #self.tar_id = 'p225'

        emb_audio, _ = librosa.load("../vctk-small/p225/p225_005_mic1.flac", sr=44100, mono=True)
        emb_audio2, _ = librosa.load("../vctk-small/p226/p226_005_mic1.flac", sr=44100, mono=True)
        emb_audio3, _ = librosa.load("../vctk-small/p227/p227_005_mic1.flac", sr=44100, mono=True)
        emb_audio4, _ = librosa.load("../vctk-small/p228/p228_005_mic1.flac", sr=44100, mono=True)
        
        self.emb_audio = torch.tensor(emb_audio[:131072]).unsqueeze(0).unsqueeze(1)
        self.emb_audio2 = torch.tensor(emb_audio2[:131072]).unsqueeze(0).unsqueeze(1)
        self.emb_audio3 = torch.tensor(emb_audio3[:131072]).unsqueeze(0).unsqueeze(1)
        self.emb_audio4 = torch.tensor(emb_audio4[:131072]).unsqueeze(0).unsqueeze(1)
        
        emb_audio_pqmf = self.pqmf(torch.tensor(self.emb_audio))
        emb_audio_pqmf2 = self.pqmf(torch.tensor(self.emb_audio2))
        emb_audio_pqmf3 = self.pqmf(torch.tensor(self.emb_audio3))
        emb_audio_pqmf4 = self.pqmf(torch.tensor(self.emb_audio4))
        
        self.speaker1 = self.speaker_encoder(emb_audio_pqmf).unsqueeze(2)
        self.speaker2 = self.speaker_encoder(emb_audio_pqmf2).unsqueeze(2)
        self.speaker3 = self.speaker_encoder(emb_audio_pqmf3).unsqueeze(2)
        self.speaker4 = self.speaker_encoder(emb_audio_pqmf4).unsqueeze(2)

        #self.register_attribute("speaker5", torch.zeros(self.speaker4.shape))
        self.speaker5 = torch.ones(self.speaker4.shape)
        
        self.target_emb = self.speaker1
        self.target_audio = self.emb_audio

        self.sr = pretrained.sr
        self.prev_f0 = torch.zeros(1)

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
        
        self.register_attribute("speaker", 0)
        self.register_attribute("record", False)

        self.register_buffer("latent_pca", pretrained.latent_pca)
        self.register_buffer("latent_mean", pretrained.latent_mean)
        self.register_buffer("fidelity", pretrained.fidelity)

        self.target_buffer_size = 131072
        self.register_buffer("buffer", torch.zeros(self.target_buffer_size))
        self.current_position = 0

        if isinstance(pretrained.encoder, rave.blocks.VariationalEncoder):
            #latent_size = max(
            #    np.argmax(pretrained.fidelity.numpy() > fidelity), 1)
            #latent_size = 2**math.ceil(math.log2(latent_size))
            self.latent_size = 64 + 256 #+ 1 #latent_size

        elif isinstance(pretrained.encoder, rave.blocks.DiscreteEncoder):
            self.latent_size = 128 #pretrained.encoder.num_quantizers

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
            "fill_buffer",
            in_channels=1,
            in_ratio=1,
            out_channels=1,
            out_ratio=1,
            input_labels=['(signal) Input audio signal'],
            output_labels=[
                f'(signal) Reconstructed audio signal {channel}'
                for channel in channels
            ],
        )

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
        """

        self.register_method(
                "myforward",
                in_channels=2,
                in_ratio=1,
                out_channels=2 if stereo else 1,
                out_ratio=1,
                input_labels=['(signal) Input audio signal', '(signal) Input audio signal'],
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

    """
    @torch.jit.export
    def fill_buffer(self, x):
        if self.record[0]:
            print("enter record")
            buffer = torch.zeros((x.shape[0], 1, self.target_buffer_size))
            current_position = 0
            chunk_length = x.shape[-1]
            
            while current_position < self.target_buffer_size:
                #print(x.shape, buffer[:, :, current_position:current_position + chunk_length].shape)
                buffer[:, :, current_position:current_position + chunk_length] = x
                current_position += chunk_length
    
            target_pqmf = self.pqmf_speaker(buffer)
            self.set_speaker5(self.speaker_encoder(target_pqmf).unsqueeze(2))
            self.set_record = False

        return x

    @torch.jit.export
    def fill_buffer(self, x):
        chunk_length = x.shape[-1]
        if self.current_position < self.target_buffer_size:
            self.buffer[self.current_position:self.current_position + chunk_length] = x[0, :, 0]
            self.current_position += chunk_length
        else:
            self.current_position = 0
    """
            
    """
    @torch.jit.export
    def encode(self, x):
        if self.is_using_adain:
            self.update_adain()
        
        if self.resampler is not None:
            x = self.resampler.to_model_sampling_rate(x)

        if self.pqmf is not None:
            x = self.pqmf(x)

        dummy_emb = self.speaker_encoder(x)

        z = self.encoder(x[: ,:6, :])
        emb = self.speaker1.repeat(z.shape[0], 1, z.shape[-1])
        z = torch.cat((z, emb), dim=1)
        
        return z

    @torch.jit.export
    def decode(self, z, from_forward: bool = False):
        if self.is_using_adain and not from_forward:
            self.update_adain()

        if self.stereo:
            z = torch.cat([z, z], 0)

        #z = self.pre_process_latent(z)
        y = self.decoder(z)

        if self.pqmf is not None:
            y = self.pqmf.inverse(y)

        if self.resampler is not None:
            y = self.resampler.from_model_sampling_rate(y)

        if self.stereo:
            y = torch.cat(y.chunk(2, 0), 1)

        return y

    def forward(self, x):
        return self.decode(self.encode(x), from_forward=True)
    """

    @torch.jit.export
    def myforward(self, x):

        if self.speaker[0] == 0:
            self.target_emb = self.speaker1
            self.target_audio = self.emb_audio
        elif self.speaker[0] == 1:
            self.target_emb = self.speaker2
            self.target_audio = self.emb_audio2
        elif self.speaker[0] == 2:
            self.target_emb = self.speaker3
            self.target_audio = self.emb_audio3
        elif self.speaker[0] == 3:
            self.target_emb = self.speaker4
            self.target_audio = self.emb_audio4
        else:
            self.target_emb = self.speaker5

        x_in = x[:, 0, :]
        pitch = x[:, 1, 0]
        

        #source_pitch = (torch.ones(source_pitch.shape) * 200) + pitch.unsqueeze(-1)

        #shift_amount = 13
        #shifted_arr = torch.zeros(source_pitch.shape)
        #shifted_arr[:, shift_amount:] = source_pitch[:, :-shift_amount]

        x_pqmf = x_in.unsqueeze(1)
        
        if self.resampler is not None:
            x_pqmf = self.resampler.to_model_sampling_rate(x_pqmf)

        if self.pqmf is not None:
            x_pqmf = self.pqmf(x_pqmf)
        
        z = self.encoder(x_pqmf[: , :6, :])
        emb = self.target_emb.repeat(z.shape[0], 1, z.shape[-1])
        z = torch.cat((z, emb), dim=1)
        
        if self.stereo:
            z = torch.cat([z, z], 0)

        """
        windows = slice_windows(x_in, 1024, 1024, pad=False)
        f0 = yin_frame(windows, 44100, 50.0, 500.0)

        if f0[0, 0] == 0:
            # use previous f0 if noisy
            f0[:, 0] = self.prev_f0
            # also assume silent if noisy
            # loudness[:, 0] = 0
        for i in range(1, f0.shape[1]):
            if f0[0, i] == 0:
                f0[:, i] = f0[:, i-1]
                # loudness[:, i] = 0

        self.prev_f0 = f0[:, -1]
        """

        mean_in, std_in, _, _ = extract_f0_median_std_inference(x_in, self.sr, 1024)
        mean_tar, std_tar, _, _ = extract_f0_median_std_inference(self.target_audio.squeeze(1), self.sr, 1024)

        f0_norm = get_f0_norm_inference(x_in, mean_in, std_in, self.sr, 1024, mult=1.0, norm_mode="none")
        
        f0_norm[f0_norm == 0] = float('nan')
        standardized_source_pitch = (f0_norm - mean_in) / std_in
        source_pitch = (standardized_source_pitch * std_tar + mean_tar)
        source_pitch = source_pitch + pitch.unsqueeze(-1)
        source_pitch[torch.isnan(source_pitch)] = 0
            
        y_multi, nsf = self.decoder(z, source_pitch.to(z))

        if self.pqmf is not None:
            y = self.pqmf.inverse(y_multi)

        if self.resampler is not None:
            y = self.resampler.from_model_sampling_rate(y)

        if self.stereo:
            y = torch.cat(y.chunk(2, 0), 1)

        y = y + nsf

        return y
    
    """
    @torch.jit.export
    def get_speaker5(self) -> torch.Tensor:
        return self.speaker5[0]

    @torch.jit.export
    def set_speaker5(self, speaker5: torch.Tensor) -> int:
        self.speaker5 = (speaker5, )
        return 0
    """
    
    @torch.jit.export
    def get_record(self) -> bool:
        return self.record[0]

    @torch.jit.export
    def set_record(self, record: bool) -> int:
        self.record = (record, )
        return 0
    
    @torch.jit.export
    def get_speaker(self) -> int:
        return self.speaker[0]

    @torch.jit.export
    def set_speaker(self, speaker: int) -> int:
        self.speaker = (speaker, )
        return 0
        
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
    cc.use_cached_conv(FLAGS.streaming)

    logging.info("building rave")

    gin.parse_config_file(os.path.join(FLAGS.run, "config.gin"))
    checkpoint = rave.core.search_for_run(FLAGS.run)
    print("loading checkpoint:", checkpoint)

    pretrained = rave.RAVE()

    with open("default.txt", "w") as file:
        for param, val in pretrained.speaker_encoder.state_dict().items():
            file.write(param + "\n")
            file.write(str(val) + "\n")
    
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

    with open("pretrained.txt", "w") as file:
        for param, val in pretrained.speaker_encoder.state_dict().items():
            file.write(param + "\n")
            file.write(str(val) + "\n")

    if isinstance(pretrained.encoder, rave.blocks.VariationalEncoder):
        script_class = ScriptedRAVE
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