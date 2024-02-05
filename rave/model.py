import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as wn
import numpy as np
import pytorch_lightning as pl
from .core import multiscale_stft, Loudness, mod_sigmoid
from .core import amp_to_impulse_response, fft_convolve, get_beta_kl_cyclic_annealed
from .pqmf import CachedPQMF as PQMF
from sklearn.decomposition import PCA
from einops import rearrange

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
import torch.nn.functional as F
from typing import Tuple
import math

from time import time

import cached_conv as cc

from .excitation import ExcitationModule, get_pitch, get_rms_val, upsample
from .stft_loss import MultiResolutionSTFTLoss

import wandb
import torchyin
from transformers import logging

from rave.discriminator import Discriminator

logging.set_verbosity_error()

class Profiler:
    def __init__(self):
        self.ticks = [[time(), None]]

    def tick(self, msg):
        self.ticks.append([time(), msg])

    def __repr__(self):
        rep = 80 * "=" + "\n"
        for i in range(1, len(self.ticks)):
            msg = self.ticks[i][1]
            ellapsed = self.ticks[i][0] - self.ticks[i - 1][0]
            rep += msg + f": {ellapsed*1000:.2f}ms\n"
        rep += 80 * "=" + "\n\n\n"
        return rep


class Residual(nn.Module):
    def __init__(self, module, cumulative_delay=0):
        super().__init__()
        additional_delay = module.cumulative_delay
        self.aligned = cc.AlignBranches(
            module,
            nn.Identity(),
            delays=[additional_delay, 0],
        )
        self.cumulative_delay = additional_delay + cumulative_delay

    def forward(self, x):
        x_net, x_res = self.aligned(x)
        return x_net + x_res


# ------------------------------------
# RESIDUAL STACK FOR DECODER
# ------------------------------------

class ResidualStack(nn.Module):
    def __init__(self,
                 dim,
                 kernel_size,
                 padding_mode,
                 cumulative_delay=0,
                 bias=False):
        super().__init__()
        net = []

        res_cum_delay = 0
        # SEQUENTIAL RESIDUALS
        for i in range(3):
            # RESIDUAL BLOCK
            seq = [nn.LeakyReLU(.2)]
            seq.append(
                wn(
                    cc.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        padding=cc.get_padding(
                            kernel_size,
                            dilation=3**i,
                            mode=padding_mode,
                        ),
                        dilation=3**i,
                        bias=bias,
                    )))

            seq.append(nn.LeakyReLU(.2))
            seq.append(
                wn(
                    cc.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        padding=cc.get_padding(kernel_size, mode=padding_mode),
                        bias=bias,
                        cumulative_delay=seq[-2].cumulative_delay,
                    )))

            res_net = cc.CachedSequential(*seq)

            net.append(Residual(res_net, cumulative_delay=res_cum_delay))
            res_cum_delay = net[-1].cumulative_delay

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay + cumulative_delay

    def forward(self, x):
        return self.net(x)


class UpsampleLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 ratio,
                 padding_mode,
                 cumulative_delay=0,
                 bias=False):
        super().__init__()
        net = [nn.LeakyReLU(.2)]
        if ratio > 1:
            net.append(
                wn(
                    cc.ConvTranspose1d(
                        in_dim,
                        out_dim,
                        2 * ratio,
                        stride=ratio,
                        padding=ratio // 2,
                        bias=bias,
                    )))
        else:
            net.append(
                wn(
                    cc.Conv1d(
                        in_dim,
                        out_dim,
                        3,
                        padding=cc.get_padding(3, mode=padding_mode),
                        bias=bias,
                    )))

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay + cumulative_delay * ratio

    def forward(self, x):
        return self.net(x)


class NoiseGenerator(nn.Module):
    def __init__(self, in_size, data_size, ratios, noise_bands, padding_mode):
        super().__init__()
        net = []
        channels = [in_size] * len(ratios) + [data_size * noise_bands]
        cum_delay = 0
        for i, r in enumerate(ratios):
            net.append(
                cc.Conv1d(
                    channels[i],
                    channels[i + 1],
                    3,
                    padding=cc.get_padding(3, r, mode=padding_mode),
                    stride=r,
                    cumulative_delay=cum_delay,
                ))
            cum_delay = net[-1].cumulative_delay
            if i != len(ratios) - 1:
                net.append(nn.LeakyReLU(.2))

        self.net = cc.CachedSequential(*net)
        self.data_size = data_size
        self.cumulative_delay = self.net.cumulative_delay * int(
            np.prod(ratios))

        self.register_buffer(
            "target_size",
            torch.tensor(np.prod(ratios)).long(),
        )

    def forward(self, x):
        amp = mod_sigmoid(self.net(x) - 5)
        amp = amp.permute(0, 2, 1)
        amp = amp.reshape(amp.shape[0], amp.shape[1], self.data_size, -1)

        ir = amp_to_impulse_response(amp, self.target_size)
        noise = torch.rand_like(ir) * 2 - 1

        noise = fft_convolve(noise, ir).permute(0, 2, 1, 3)
        noise = noise.reshape(noise.shape[0], noise.shape[1], -1)
        return noise
    
# ------------------------------------
# OLD GENERATOR
# ------------------------------------
    
class Generator_Old(nn.Module):
    def __init__(self,
                 latent_size,
                 capacity,
                 data_size,
                 ratios,
                 loud_stride,
                 use_noise,
                 noise_ratios,
                 noise_bands,
                 padding_mode,
                 bias=False):
        super().__init__()
        net = [
            wn(
                cc.Conv1d(
                    latent_size,
                    2**len(ratios) * capacity,
                    7,
                    padding=cc.get_padding(7, mode=padding_mode),
                    bias=bias,
                ))
        ]

        for i, r in enumerate(ratios):
            in_dim = 2**(len(ratios) - i) * capacity
            out_dim = 2**(len(ratios) - i - 1) * capacity

            net.append(
                UpsampleLayer(
                    in_dim,
                    out_dim,
                    r,
                    padding_mode,
                    cumulative_delay=net[-1].cumulative_delay,
                ))
            net.append(
                ResidualStack(
                    out_dim,
                    3,
                    padding_mode,
                    cumulative_delay=net[-1].cumulative_delay,
                ))

        self.net = cc.CachedSequential(*net)

        wave_gen = wn(
            cc.Conv1d(
                out_dim,
                data_size,
                7,
                padding=cc.get_padding(7, mode=padding_mode),
                bias=bias,
            ))

        loud_gen = wn(
            cc.Conv1d(
                out_dim,
                1,
                2 * loud_stride + 1,
                stride=loud_stride,
                padding=cc.get_padding(2 * loud_stride + 1,
                                       loud_stride,
                                       mode=padding_mode),
                bias=bias,
            ))

        branches = [wave_gen, loud_gen]

        if use_noise:
            noise_gen = NoiseGenerator(
                out_dim,
                data_size,
                noise_ratios,
                noise_bands,
                padding_mode=padding_mode,
            )
            branches.append(noise_gen)

        self.synth = cc.AlignBranches(
            *branches,
            cumulative_delay=self.net.cumulative_delay,
        )

        self.use_noise = use_noise
        self.loud_stride = loud_stride
        self.cumulative_delay = self.synth.cumulative_delay

    def forward(self, x, add_noise: bool = True):
        x = self.net(x)

        if self.use_noise:
            waveform, loudness, noise = self.synth(x)
        else:
            waveform, loudness = self.synth(x)
            noise = torch.zeros_like(waveform)

        loudness = loudness.repeat_interleave(self.loud_stride)
        loudness = loudness.reshape(x.shape[0], 1, -1)

        waveform = torch.tanh(waveform) * mod_sigmoid(loudness)

        if add_noise:
            waveform = waveform + noise

        return waveform


# ------------------------------------
# NEW GENERATOR
# ------------------------------------

class DownsampleLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 ratio,
                 padding_mode,
                 cumulative_delay=0,
                 bias=False):
        super().__init__()
        net = []
        if ratio > 1:
            net.append(
                wn(
                    cc.Conv1d(
                        in_dim,
                        out_dim,
                        kernel_size=3,
                        stride=ratio,
                        padding=cc.get_padding(3, mode=padding_mode),
                        bias=bias,
                    )))
        else:
            net.append(
                wn(
                    cc.Conv1d(
                        in_dim,
                        out_dim,
                        3,
                        padding=cc.get_padding(3, mode=padding_mode),
                        bias=bias,
                    )))

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay + cumulative_delay * ratio

    def forward(self, x):
        return self.net(x)


class FiLM(nn.Module):
    def __init__(
        self,
        cond_dim,  # dim of conditioning input
        batch_norm=True,
    ):
        super().__init__()
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = torch.nn.BatchNorm1d(cond_dim // 2, affine=False)

    def forward(self, x, cond):

        device = x.device
        self.bn.to(x.device)

        g, b = torch.chunk(cond, 2, dim=1)

        if self.batch_norm:
            x = self.bn(x)  # apply BatchNorm without affine
        x = (x * g) + b  # then apply conditional affine

        return x


class Generator(nn.Module):
    def __init__(self,
                 latent_size,
                 capacity,
                 data_size,
                 ratios_up,
                 ratios_down,
                 loud_stride,
                 use_noise,
                 noise_ratios,
                 noise_bands,
                 padding_mode,
                 bias=False):

        super().__init__()

        # UPSAMPLER
        upsample_net = [
            wn(
                cc.Conv1d(
                    latent_size,
                    2**len(ratios_up) * capacity,
                    7,
                    padding=cc.get_padding(7, mode=padding_mode),
                    bias=bias,
                ))
        ]

        for i, r in enumerate(ratios_up):
            in_dim = 2**(len(ratios_up) - i) * capacity
            out_dim = 2**(len(ratios_up) - i - 1) * capacity

            upsample_net.append(
                UpsampleLayer(
                    in_dim,
                    out_dim,
                    r,
                    padding_mode,
                    cumulative_delay=upsample_net[-1].cumulative_delay,
                ))
            upsample_net.append(
                ResidualStack(
                    out_dim,
                    3,
                    padding_mode,
                    cumulative_delay=upsample_net[-1].cumulative_delay,
                ))

        # DOWNSAMPLER
        downsample_net = []
        for i, r in enumerate(ratios_down):
            in_dim = 16
            out_dim_d = 16

            if i == 0:
                downsample_net.append(
                    DownsampleLayer(in_dim,
                                    out_dim_d,
                                    r,
                                    padding_mode,
                                    cumulative_delay=0))
            else:
                downsample_net.append(
                    DownsampleLayer(
                        in_dim,
                        out_dim_d,
                        r,
                        padding_mode,
                        cumulative_delay=downsample_net[-1].cumulative_delay,
                    ))

        downsample_conv = []
        for i, r in enumerate(ratios_down):
            if i == 0:
                downsample_conv.append(
                    wn(
                        cc.Conv1d(
                            16,
                            (2**(i) * capacity) * 2,
                            1,
                            padding=cc.get_padding(1, mode=padding_mode),
                            cumulative_delay=0,
                            bias=bias,
                        )))
            else:
                downsample_conv.append(
                    wn(
                        cc.Conv1d(
                            16,
                            (2**(i) * capacity) * 2,
                            1,
                            padding=cc.get_padding(1, mode=padding_mode),
                            cumulative_delay=downsample_conv[-1].
                            cumulative_delay,
                            bias=bias,
                        )))

        self.upsampler = cc.CachedSequential(*upsample_net)
        self.downsampler = cc.CachedSequential(*downsample_net)
        self.downsampler_conv = cc.CachedSequential(*downsample_conv)

        # -----------------------------------------------

        film_net = []
        film_net.append(FiLM(1024))
        film_net.append(FiLM(512))
        film_net.append(FiLM(256))
        film_net.append(FiLM(128))
        self.film_conditioning = film_net

        # -----------------------------------------------

        wave_gen = wn(
            cc.Conv1d(
                out_dim,
                data_size,
                7,
                padding=cc.get_padding(7, mode=padding_mode),
                bias=bias,
            ))

        loud_gen = wn(
            cc.Conv1d(
                out_dim,
                1,
                2 * loud_stride + 1,
                stride=loud_stride,
                padding=cc.get_padding(2 * loud_stride + 1,
                                       loud_stride,
                                       mode=padding_mode),
                bias=bias,
            ))

        branches = [wave_gen, loud_gen]

        if use_noise:
            noise_gen = NoiseGenerator(
                out_dim,
                data_size,
                noise_ratios,
                noise_bands,
                padding_mode=padding_mode,
            )
            branches.append(noise_gen)

        self.synth = cc.AlignBranches(
            *branches,
            cumulative_delay=self.upsampler.cumulative_delay,
        )

        self.use_noise = use_noise
        self.loud_stride = loud_stride
        self.cumulative_delay = self.synth.cumulative_delay

    def forward(self, x, ex, add_noise: bool = True):
        downsampled_layers = []
        for down_layer, conv_layer in zip(self.downsampler,
                                          self.downsampler_conv):
            ex = down_layer(ex)
            downsampled_layers.append(conv_layer(ex))

        for i, up_layer in enumerate(self.upsampler):
            if i % 2 != 0 or i == 0:
                x = up_layer(x)
            else:
                x = up_layer(x)
                x = self.film_conditioning[i // 2 - 1](
                    x, downsampled_layers[len(downsampled_layers) - (i // 2)])

        # -----------------------------------------------

        if self.use_noise:
            waveform, loudness, noise = self.synth(x)
        else:
            waveform, loudness = self.synth(x)
            noise = torch.zeros_like(waveform)

        loudness = loudness.repeat_interleave(self.loud_stride)
        loudness = loudness.reshape(x.shape[0], 1, -1)

        waveform = torch.tanh(waveform) * mod_sigmoid(loudness)

        if add_noise:
            waveform = waveform + noise

        return waveform

# ------------------------------------
# ENCODER
# ------------------------------------

class Encoder(nn.Module):
    def __init__(self,
                 data_size,
                 capacity,
                 latent_size,
                 ratios,
                 padding_mode,
                 bias=False):
        super().__init__()
        net = [
            cc.Conv1d(data_size,
                      capacity,
                      7,
                      padding=cc.get_padding(7, mode=padding_mode),
                      bias=bias)
        ]

        for i, r in enumerate(ratios):
            in_dim = 2**i * capacity
            out_dim = 2**(i + 1) * capacity

            net.append(nn.BatchNorm1d(in_dim))
            net.append(nn.LeakyReLU(.25))
            net.append(
                cc.Conv1d(
                    in_dim,
                    out_dim,
                    2 * r + 1,
                    padding=cc.get_padding(2 * r + 1, r, mode=padding_mode),
                    stride=r,
                    bias=bias,
                    cumulative_delay=net[-3].cumulative_delay,
                ))

        net.append(nn.LeakyReLU(.2))
        net.append(
            cc.Conv1d(
                out_dim,
                latent_size,
                5,
                padding=cc.get_padding(5, mode=padding_mode),
                groups=2,
                bias=bias,
                cumulative_delay=net[-2].cumulative_delay,
            ))
        
        #net.append(torch.nn.LayerNorm(16))

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        z = self.net(x)
        return z #torch.split(z, z.shape[1] // 2, 1)


# class Discriminator(nn.Module):
#     def __init__(self, in_size, capacity, multiplier, n_layers):
#         super().__init__()

#         net = [
#             wn(cc.Conv1d(in_size, capacity, 15, padding=cc.get_padding(15)))
#         ]
#         net.append(nn.LeakyReLU(.2))

#         for i in range(n_layers):
#             net.append(
#                 wn(
#                     cc.Conv1d(
#                         capacity * multiplier**i,
#                         min(1024, capacity * multiplier**(i + 1)),
#                         41,
#                         stride=multiplier,
#                         padding=cc.get_padding(41, multiplier),
#                         groups=multiplier**(i + 1),
#                     )))
#             net.append(nn.LeakyReLU(.2))

#         net.append(
#             wn(
#                 cc.Conv1d(
#                     min(1024, capacity * multiplier**(i + 1)),
#                     min(1024, capacity * multiplier**(i + 1)),
#                     5,
#                     padding=cc.get_padding(5),
#                 )))
#         net.append(nn.LeakyReLU(.2))
#         net.append(
#             wn(cc.Conv1d(min(1024, capacity * multiplier**(i + 1)), 1, 1)))
#         self.net = nn.ModuleList(net)

#     def forward(self, x):
#         feature = []
#         for layer in self.net:
#             x = layer(x)
#             if isinstance(layer, nn.Conv1d):
#                 feature.append(x)
#         return feature


# class StackDiscriminators(nn.Module):
#     def __init__(self, n_dis, *args, **kwargs):
#         super().__init__()
#         self.discriminators = nn.ModuleList(
#             [Discriminator(*args, **kwargs) for i in range(n_dis)], )

#     def forward(self, x):
#         features = []
#         for layer in self.discriminators:
#             features.append(layer(x))
#             x = nn.functional.avg_pool1d(x, 2)
#         return features
    

class CrossEntropyProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(16)
        self.lin1 = torch.nn.Linear(64, 100)
        self.lin2 = torch.nn.Linear(16, 102)
        #self.logistic_projection = torch.nn.Linear(32, 100)
        self.softmax = torch.nn.Softmax()
        
    def forward(self, x):
        z_for_CE = self.layer_norm(x)
        z_for_CE = self.lin1(torch.permute(z_for_CE, (0, 2, 1)))
        z_for_CE = self.lin2(torch.permute(z_for_CE, (0, 2, 1)))
        z_for_CE = self.softmax(z_for_CE)
        return z_for_CE #torch.permute(z_for_CE, (0, 2, 1))
        
    
class ContrastiveLoss(nn.Module):
    """Contrastive loss.
    Introduced in Kaizhi Qian et al., _Contentvec: An improved self-supervised speech representation by disentangling speakers_
    """
    def __init__(
        self,
        num_candidates: int,
        negative_samples_minimum_distance_to_positive: int,
        temperature: float,
    ) -> None:
        super().__init__()

        self.negative_samples_minimum_distance_to_positive = (
            negative_samples_minimum_distance_to_positive
        )
        self.temperature = temperature
        self.num_candidates = num_candidates

    def make_negative_sampling_mask(
        self, num_items: int, device: torch.device
    ) -> torch.Tensor:
        upper_triu_mask = torch.triu(
            torch.ones(num_items, num_items),
            self.negative_samples_minimum_distance_to_positive + 1,
        )
        all_mask = upper_triu_mask.T + upper_triu_mask
        random_values = all_mask * torch.rand(num_items, num_items)
        k_th_quant = torch.topk(random_values, min(self.num_candidates, num_items))[0][
            :, -1:
        ]
        random_mask = (random_values >= k_th_quant) * all_mask + torch.eye(num_items)
        return random_mask.to(device)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # [2, B, linguistic_hidden_channels, S], normalize for cosine similarity.
        cosine_similarity = F.normalize(torch.stack([x, y], dim=0), p=2, dim=2)
        # N
        num_tokens = cosine_similarity.shape[-1]
        # [B, N]
        positive = cosine_similarity.prod(dim=0).sum(dim=1) / self.temperature
        #positive = positive.exp()
        # [2, B, N, N]
        confusion_matrix = (torch.matmul(cosine_similarity.transpose(2, 3), cosine_similarity) / self.temperature)
        # [N, N]
        negative_sampling_mask = self.make_negative_sampling_mask(num_tokens, x.device)

        # [2, B, N, N(sum = candidates)], negative case
        masked_confusion_matrix = confusion_matrix.masked_fill(
            ~negative_sampling_mask.to(torch.bool), -np.inf
        )
        # [2, B, N], negative case
        negative = masked_confusion_matrix.exp().sum(dim=-1)
        # []
        contrastive_loss = (
            -torch.sum(positive / negative, dim=-1).log().sum(dim=0).mean()
        )
        
        mean_positive = positive.mean() * self.temperature
        mean_negative = (
            (confusion_matrix * negative_sampling_mask).sum(dim=-1)
            / self.num_candidates
        ).mean() * self.temperature

        return contrastive_loss, mean_positive, mean_negative


class RAVE(pl.LightningModule):
    def __init__(self,
                 data_size,
                 capacity,
                 latent_size,
                 ratios,
                 ratios_down,
                 bias,
                 loud_stride,
                 use_noise,
                 noise_ratios,
                 noise_bands,
                 d_capacity,
                 d_multiplier,
                 d_n_layers,
                 warmup,
                 mode,
                 block_size,
                 speaker_encoder,
                 contrastive_loss,
                 content_loss,
                 no_latency=False,
                 min_kl=1e-4,
                 max_kl=5e-1,
                 cropped_latent_size=0,
                 feature_match=True,
                 sr=24000):
        super().__init__()
        self.save_hyperparameters()

        if data_size == 1:
            self.pqmf = None
        else:
            self.pqmf = PQMF(70 if no_latency else 100, data_size)

        self.loudness = Loudness(sr, 512)

        encoder_out_size = cropped_latent_size if cropped_latent_size else latent_size

        self.encoder = Encoder(
            data_size,
            capacity,
            encoder_out_size,
            ratios,
            "causal" if no_latency else "centered",
            bias,
        )
        
        self.CE_projection = CrossEntropyProjection()

        if speaker_encoder == "RESNET":
            speaker_size = 512
        elif speaker_encoder == "ECAPA":
            speaker_size = 192
        else:
            speaker_size = 0
            
        self.speaker_projection = torch.nn.Linear(speaker_size, 256)

        new_latent_size = latent_size + 256
        self.decoder = Generator_Old(
            new_latent_size,
            capacity,
            data_size,
            ratios,
            loud_stride,
            use_noise,
            noise_ratios,
            noise_bands,
            "causal" if no_latency else "centered",
            bias,
        )

        # self.discriminator = StackDiscriminators(
        #     3,
        #     in_size=1,
        #     capacity=d_capacity,
        #     multiplier=d_multiplier,
        #     n_layers=d_n_layers,
        # )

        self.discriminator = Discriminator()

        self.idx = 0

        self.register_buffer("latent_pca", torch.eye(encoder_out_size))
        self.register_buffer("latent_mean", torch.zeros(encoder_out_size))
        self.register_buffer("fidelity", torch.zeros(encoder_out_size))

        self.latent_size = latent_size

        self.automatic_optimization = False

        self.warmup = warmup
        self.warmed_up = False
        self.sr = sr
        self.mode = mode

        self.min_kl = min_kl
        self.max_kl = max_kl
        self.cropped_latent_size = cropped_latent_size

        self.feature_match = feature_match

        self.register_buffer("saved_step", torch.tensor(0))

        self.block_size = block_size
        self.excitation_module = ExcitationModule(self.sr, self.block_size)
        
        self.contr_coeff = 1e-5
        self.contr_loss = ContrastiveLoss(num_candidates=15,
                                          negative_samples_minimum_distance_to_positive=10,
                                          temperature=0.1)
        
        self.contrastive = contrastive_loss
        self.content = content_loss

        resolutions = []
        for hop_length_ms, win_length_ms in eval("[(5, 25), (10, 50), (2, 10)]"):
            hop_length = int(0.001 * hop_length_ms * self.sr)
            win_length = int(0.001 * win_length_ms * self.sr)
            n_fft = int(math.pow(2, int(math.log2(win_length)) + 1))
            resolutions.append((n_fft, hop_length, win_length))

        self.stft_criterion = MultiResolutionSTFTLoss(self.device, resolutions).to(self.device)

    def configure_optimizers(self):
        
        #enc_p = list(self.encoder.parameters())
        #enc_p += list(self.CE_projection.parameters())
        
        gen_p = list(self.decoder.parameters())
        gen_p += list(self.encoder.parameters())
        gen_p += list(self.CE_projection.parameters())
        dis_p = list(self.discriminator.parameters())
        
        #enc_opt = torch.optim.Adam(enc_p, 1e-4, (.5, .9))
        gen_opt = torch.optim.Adam(gen_p, 1e-4, (.5, .9))
        dis_opt = torch.optim.Adam(dis_p, 1e-4, (.5, .9))

        return gen_opt, dis_opt
    
    def contrastive_loss_function(self, ling_1, ling_2, kappa=0.1, content_adj=10, candidates=15):
    
        ling_s = F.normalize(torch.stack([ling_1, ling_2], dim=0), p=2, dim=2)
        num_tokens = ling_s.shape[-1]
        pos = ling_s.prod(dim=0).sum(dim=1) / kappa
        confusion = torch.matmul(ling_s.transpose(2, 3), ling_s) / kappa

        placeholder = torch.zeros(num_tokens, device=self.device)
        mask = torch.stack([
            placeholder.scatter(
                0,
                (
                    torch.randperm(num_tokens - content_adj, device=self.device)[:candidates]
                    + i + content_adj // 2 + 1) % num_tokens,
                1.)
            for i in range(num_tokens)])

        masked = confusion.masked_fill(~mask.to(torch.bool), -np.inf)
        neg = torch.logsumexp(masked, dim=-1)

        contr_loss = -torch.logsumexp(pos - neg, dim=-1).sum(dim=0).mean()
        return contr_loss
    
    def update_warmup(self):
        self.contr_coeff = min(
            self.contr_coeff + 1e-5, 10)

    def lin_distance(self, x, y):
        return torch.norm(x - y) / torch.norm(x)

    def log_distance(self, x, y):
        return abs(torch.log(x + 1e-7) - torch.log(y + 1e-7)).mean()

    def distance(self, x, y):
        scales = [2048, 1024, 512, 256, 128]
        x = multiscale_stft(x, scales, .75)
        y = multiscale_stft(y, scales, .75)

        lin = sum(list(map(self.lin_distance, x, y)))
        log = sum(list(map(self.log_distance, x, y)))

        return lin + log

    def reparametrize(self, mean, scale):
        std = nn.functional.softplus(scale) + 1e-4
        var = std * std
        logvar = torch.log(var)

        z = torch.randn_like(mean) * std + mean

        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        if self.cropped_latent_size:
            noise = torch.randn(
                z.shape[0],
                self.latent_size - self.cropped_latent_size,
                z.shape[-1],
            ).to(z.device)
            z = torch.cat([z, noise], 1)
        return z, kl

    def adversarial_combine(self, score_real, score_fake, mode="hinge"):
        if mode == "hinge":
            loss_dis = torch.relu(1 - score_real) + torch.relu(1 + score_fake)
            loss_dis = loss_dis.mean()
            loss_gen = -score_fake.mean()
        elif mode == "square":
            loss_dis = (score_real - 1).pow(2) + score_fake.pow(2)
            loss_dis = loss_dis.mean()
            loss_gen = (score_fake - 1).pow(2).mean()
        elif mode == "nonsaturating":
            score_real = torch.clamp(torch.sigmoid(score_real), 1e-7, 1 - 1e-7)
            score_fake = torch.clamp(torch.sigmoid(score_fake), 1e-7, 1 - 1e-7)
            loss_dis = -(torch.log(score_real) +
                         torch.log(1 - score_fake)).mean()
            loss_gen = -torch.log(score_fake).mean()
        else:
            raise NotImplementedError
        return loss_dis, loss_gen

    def training_step(self, batch, batch_idx):
        p = Profiler()
        self.saved_step += 1

        gen_opt, dis_opt = self.optimizers()

        x = batch['data_clean']

        # SPEAKER EMBEDDING AND PITCH EXCITATION
        sp = batch['speaker_emb']
        sp = self.speaker_projection(sp)
        sp = torch.permute(sp.unsqueeze(1).repeat(1, 16, 1), (0, 2, 1))

        # --------------------------------------

        x_clean = x.unsqueeze(1)
        x_perturb =  batch['data_perturbed_1'].unsqueeze(1)
        
        if self.pqmf is not None:  # MULTIBAND DECOMPOSITION
            x_clean = self.pqmf(x_clean)
            x_perturb = self.pqmf(x_perturb)
            p.tick("pqmf")

        #if self.warmed_up:  # EVAL ENCODER
        #    self.encoder.eval()
        #    self.CE_projection.eval()

        # ENCODE INPUT
        #z_init_1, kl = self.reparametrize(*self.encoder(x_clean[:, :5, :]))
        z_init_1 = self.encoder(x_perturb)
        kl = 0
        p.tick("encode")

        #if self.warmed_up:  # FREEZE ENCODER
            #z_init_1 = z_init_1.detach()
           # kl = kl.detach()
            
        predicted_units = self.CE_projection(z_init_1)
 
        z = torch.cat((z_init_1, sp), 1)

        #z = torch.cat((z_init_1, sp), 1)
        
        #z_detached = z.detach()
        
        #if self.warmed_up:
        #    z = z.detach()

        # DECODE LATENT
        y_pqmf = self.decoder(z, add_noise=self.warmed_up)
        
        #if self.warmed_up:
        #    z = z.detach()

        # DECODE LATENT
        #y_pqmf = self.decoder(z, add_noise=self.warmed_up)
        p.tick("decode")
        
        # CONTENT OF RECONSTRUCTED (Y)
        if self.content:
            with torch.no_grad():
                #y_enc, _ = self.reparametrize(*self.encoder(y_pqmf[:, :5, :]))
                y_enc, _ = self.encoder(y_pqmf[:, :5, :])
            content_loss = torch.nn.functional.l1_loss(z_init_1, y_enc) * 7.5
        else:
            content_loss = 0

        # DISTANCE BETWEEN INPUT AND OUTPUT
        #distance = self.distance(x_clean, y_pqmf)
        #print(distance)
        p.tick("mb distance")
        
        if self.contrastive:
            if self.warmed_up:
                contrastrive_loss = 0
                mean_positive = 0
                mean_negative = 0
            else:
                contrastrive_loss, mean_positive, mean_negative = self.contr_loss(z_init_1, z_init_2)
        else:
            contrastrive_loss = 0
            mean_positive = 0
            mean_negative = 0

        if self.pqmf is not None:  # FULL BAND RECOMPOSITION
            x_clean = self.pqmf.inverse(x_clean)
            y = self.pqmf.inverse(y_pqmf)
            sc_loss, mag_loss = self.stft_criterion(y.squeeze(1), x_clean.squeeze(1))
            distance = (sc_loss + mag_loss) * 2.5
            #print(new_dis)
            #distance = distance + new_dis #self.distance(x_clean, y)
            p.tick("fb distance")

        loud_x = self.loudness(x_clean)
        loud_y = self.loudness(y)
        loud_dist = (loud_x - loud_y).pow(2).mean()
        distance = distance #+ loud_dist
        p.tick("loudness distance")

        CE_loss = torch.nn.functional.cross_entropy(predicted_units, batch['discrete_units_16k'].type(torch.int64))

        # MRD, MPD losses.
        res_fake, period_fake = self.discriminator(y)

        # Compute LSGAN loss for all frames.
        loss_adv = 0.0
        for (_, score_fake) in res_fake + period_fake:
            loss_adv += torch.mean(torch.pow(score_fake - 1.0, 2))

        # Average across frames.
        loss_adv = loss_adv / len(res_fake + period_fake)
        
        # Overall generator loss (L_G).
        loss_gen = loss_adv + distance + CE_loss

        # MRD, MPD losses.
        res_fake, period_fake = self.discriminator(y.detach())  # fake audio from generator
        res_real, period_real = self.discriminator(x_clean)  # real audio

        # Compute LSGAN loss for all frames.
        loss_dis = 0.0
        for (_, score_fake), (_, score_real) in zip(
            res_fake + period_fake, res_real + period_real
        ):
            loss_dis += torch.mean(torch.pow(score_real - 1.0, 2))
            loss_dis += torch.mean(torch.pow(score_fake, 2))

        # Compute average to get overall discriminator loss (L_D).
        loss_dis = loss_dis / len(res_fake + period_fake)

        p.tick("gen loss compose")

        if self.global_step % 2:
            dis_opt.zero_grad()
            loss_dis.backward()
            dis_opt.step()
        else:
            gen_opt.zero_grad()
            loss_gen.backward()
            gen_opt.step()
    
        p.tick("optimization")

        # LOGGING
        self.log("loss_dis", loss_dis)
        self.log("loss_gen", loss_gen)
        self.log("loud_dist", loud_dist)
        self.log("regularization", kl)
        #self.log("pred_true", pred_true.mean())
        #self.log("pred_fake", pred_fake.mean())
        self.log("distance", distance)
        #self.log("beta", beta)
        #self.log("feature_matching", feature_matching_distance),
        self.log("CE", CE_loss),
        #self.log("content_loss", content_loss)
        p.tick("log")

        wandb.log({
            "loss_dis": loss_dis,
            "loss_gen": loss_gen,
            "distance": distance,
            #"feature_matching": feature_matching_distance,
            "contrastive_loss": contrastrive_loss,
            "contrastive_coeff": self.contr_coeff,
            "mean_positive": mean_positive,
            "mean_negative": mean_negative,
            "content": content_loss,
            "CE": CE_loss
            #"content_loss": content_loss
        })
        
        self.update_warmup()

        if self.saved_step > self.warmup:
              self.warmed_up = True

        # print(p)

    def encode(self, x, sp):

        # SPEAKER EMBEDDING AND PITCH EXCITATION
        sp = self.speaker_projection(sp)
        sp = torch.permute(sp.unsqueeze(1).repeat(1, 16, 1), (0, 2, 1))
        
        x = x.unsqueeze(1)
        
        if self.pqmf is not None:
            x = self.pqmf(x)

        z = self.encoder(x)
        
        return z, torch.cat((z, sp), 1), sp

    def decode(self, z):
        y = self.decoder(z, add_noise=True)
        if self.pqmf is not None:
            y = self.pqmf.inverse(y)
        return y

    def validation_step(self, batch, batch_idx):

        x = batch['data_clean']

        # SPEAKER EMBEDDING AND PITCH EXCITATION
        sp = batch['speaker_emb']
        sp = self.speaker_projection(sp)
        sp = torch.permute(sp.unsqueeze(1).repeat(1, 16, 1), (0, 2, 1))

        # --------------------------------------

        x_clean = x.unsqueeze(1)
        
        if self.pqmf is not None:
            x_clean = self.pqmf(x_clean)

        #mean, scale = self.encoder(x_clean[:, :5, :])
        #z, _ = self.reparametrize(mean, scale)
        z = self.encoder(x_clean)

        z = torch.cat((z, sp), 1)
        y = self.decoder(z, add_noise=self.warmed_up)

        if self.pqmf is not None:
            x_clean = self.pqmf.inverse(x_clean)
            y = self.pqmf.inverse(y)

        distance = self.distance(x_clean, y)

        #if self.trainer is not None:
        self.log("validation", distance)
        wandb.log({"validation": distance})

        #FOR CONVERSION
        speaker_emb_avg = batch['speaker_id_avg']
        speaker_emb_avg = self.speaker_projection(speaker_emb_avg)
        speaker_emb_avg = torch.permute(speaker_emb_avg.unsqueeze(1).repeat(1, 16, 1), (0, 2, 1))

        input_index = 0
        target_index = 1

        input_conversion = x[input_index].unsqueeze(0).unsqueeze(0)
        target_conversion = x[target_index].unsqueeze(0).unsqueeze(0)
        target_embedding = speaker_emb_avg[target_index].unsqueeze(0)

        if self.pqmf is not None:
            input_conversion = self.pqmf(input_conversion)

        #mean, scale = self.encoder(input_conversion[:, :5, :])
        #z, _ = self.reparametrize(mean, scale)
        z = self.encoder(input_conversion)

        z = torch.cat((z, target_embedding), 1)
        converted = self.decoder(z, add_noise=self.warmed_up)

        if self.pqmf is not None:
            converted = self.pqmf.inverse(converted)
            input_conversion = self.pqmf.inverse(input_conversion)

        return (torch.cat([x_clean, y], -1), 0, 
                #torch.cat([x_clean, x_perturbed_1, x_perturbed_2], -1),
                torch.cat([input_conversion, target_conversion, converted], -1))

    def validation_epoch_end(self, out):
        if len(out) != 0:
            audio, z, converted = list(zip(*out))

            if self.saved_step > self.warmup:
                self.warmed_up = True
        
        """
        # LATENT SPACE ANALYSIS
        if not self.warmed_up:
            z = torch.cat(z, 0)
            z = rearrange(z, "b c t -> (b t) c")

            self.latent_mean.copy_(z.mean(0))
            z = z - self.latent_mean
            
            print(z.shape)

            pca = PCA(z.shape[-1]).fit(z.cpu().numpy())

            components = pca.components_
            components = torch.from_numpy(components).to(z)
            self.latent_pca.copy_(components)

            var = pca.explained_variance_ / np.sum(pca.explained_variance_)
            var = np.cumsum(var)

            self.fidelity.copy_(torch.from_numpy(var).to(self.fidelity))

            var_percent = [.8, .9, .95, .99]
            for p in var_percent:
                self.log(f"{p}%_manifold",
                         np.argmax(var > p).astype(np.float32))
        
        """
        if len(out) != 0:
            y = torch.cat(audio, 0)[:64].reshape(-1)
            self.logger.experiment.add_audio("audio_val", y,
                                         self.saved_step.item(), self.sr)

            wandb.log({f"audio_val_{self.saved_step.item():06d}":
                        wandb.Audio(y.detach().cpu().numpy(),
                        caption="audio",
                        sample_rate=self.sr)
           })

            convert = torch.cat(converted, 0)[:64].reshape(-1)
            wandb.log({
                f"audio_conv{self.saved_step.item():06d}":
           wandb.Audio(convert.detach().cpu().numpy(),
                        caption="audio",
                        sample_rate=self.sr)
           })

            self.idx += 1

        if len(out) == 0:
            self.idx += 1