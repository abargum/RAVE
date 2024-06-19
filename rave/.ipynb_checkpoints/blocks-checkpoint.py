from functools import partial
from typing import Callable, Optional, Sequence, Union, Tuple, List

import cached_conv as cc
import gin
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torchaudio.transforms import Spectrogram

from .core import amp_to_impulse_response, fft_convolve, mod_sigmoid

import torch.nn.utils.weight_norm as wn
import torch.nn.functional as F

@gin.configurable
def normalization(module: nn.Module, mode: str = 'identity'):
    if mode == 'identity':
        return module
    elif mode == 'weight_norm':
        return weight_norm(module)
    else:
        raise Exception(f'Normalization mode {mode} not supported')

class SampleNorm(nn.Module):

    def forward(self, x):
        return x / torch.norm(x, 2, 1, keepdim=True)


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


class ResidualLayer(nn.Module):

    def __init__(
        self,
        dim,
        kernel_size,
        dilations,
        cumulative_delay=0,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2)):
        super().__init__()
        net = []
        cd = 0
        for d in dilations:
            net.append(activation(dim))
            net.append(
                normalization(
                    cc.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        dilation=d,
                        padding=cc.get_padding(kernel_size, dilation=d),
                        cumulative_delay=cd,
                    )))
            cd = net[-1].cumulative_delay
        self.net = Residual(
            cc.CachedSequential(*net),
            cumulative_delay=cumulative_delay,
        )
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        return self.net(x)


class DilatedUnit(nn.Module):

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        dilation: int,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2)
    ) -> None:
        super().__init__()
        net = [
            activation(dim),
            normalization(
                cc.Conv1d(dim,
                          dim,
                          kernel_size=kernel_size,
                          dilation=dilation,
                          padding=cc.get_padding(
                              kernel_size,
                              dilation=dilation,
                          ))),
            activation(dim),
            normalization(cc.Conv1d(dim, dim, kernel_size=1)),
        ]

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = net[1].cumulative_delay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):

    def __init__(self,
                 dim,
                 kernel_size,
                 dilations_list,
                 cumulative_delay=0) -> None:
        super().__init__()
        layers = []
        cd = 0

        for dilations in dilations_list:
            layers.append(
                ResidualLayer(
                    dim,
                    kernel_size,
                    dilations,
                    cumulative_delay=cd,
                ))
            cd = layers[-1].cumulative_delay

        self.net = cc.CachedSequential(
            *layers,
            cumulative_delay=cumulative_delay,
        )
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        return self.net(x)


@gin.configurable
class ResidualStack(nn.Module):

    def __init__(self,
                 dim,
                 kernel_sizes,
                 dilations_list,
                 cumulative_delay=0) -> None:
        super().__init__()
        blocks = []
        for k in kernel_sizes:
            blocks.append(ResidualBlock(dim, k, dilations_list))
        self.net = cc.AlignBranches(*blocks, cumulative_delay=cumulative_delay)
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        x = self.net(x)
        x = torch.stack(x, 0).sum(0)
        return x


class UpsampleLayer(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        ratio,
        cumulative_delay=0,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2)):
        super().__init__()
        net = [activation(in_dim)]
        if ratio > 1:
            net.append(
                normalization(
                    cc.ConvTranspose1d(in_dim,
                                       out_dim,
                                       2 * ratio,
                                       stride=ratio,
                                       padding=ratio // 2)))
        else:
            net.append(
                normalization(
                    cc.Conv1d(in_dim, out_dim, 3, padding=cc.get_padding(3))))

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay + cumulative_delay * ratio

    def forward(self, x):
        return self.net(x)


@gin.configurable
class NoiseGenerator(nn.Module):

    def __init__(self, in_size, data_size, ratios, noise_bands):
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
                    padding=cc.get_padding(3, r),
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


class NoiseGeneratorV2(nn.Module):

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        data_size: int,
        ratios: int,
        noise_bands: int,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2),
    ):
        super().__init__()
        net = []
        channels = [in_size]
        channels.extend((len(ratios) - 1) * [hidden_size])
        channels.append(data_size * noise_bands)

        for i, r in enumerate(ratios):
            net.append(
                cc.Conv1d(
                    channels[i],
                    channels[i + 1],
                    2 * r,
                    padding=(r, 0),
                    stride=r,
                ))
            if i != len(ratios) - 1:
                net.append(activation(channels[i + 1]))

        self.net = nn.Sequential(*net)
        self.data_size = data_size

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


class GRU(nn.Module):

    def __init__(self, latent_size: int, num_layers: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=latent_size,
            hidden_size=latent_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.register_buffer("gru_state", torch.tensor(0))
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled: return x
        x = x.permute(0, 2, 1)
        x = self.gru(x)[0]
        x = x.permute(0, 2, 1)
        return x

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True


class Generator(nn.Module):

    def __init__(
        self,
        latent_size,
        capacity,
        data_size,
        ratios,
        loud_stride,
        use_noise,
        recurrent_layer: Optional[Callable[[], nn.Module]] = None,
    ):
        super().__init__()
        net = [
            normalization(
                cc.Conv1d(
                    latent_size,
                    2**len(ratios) * capacity,
                    7,
                    padding=cc.get_padding(7),
                ))
        ]

        if recurrent_layer is not None:
            net.append(
                recurrent_layer(
                    dim=2**len(ratios) * capacity,
                    cumulative_delay=net[0].cumulative_delay,
                ))

        for i, r in enumerate(ratios):
            in_dim = 2**(len(ratios) - i) * capacity
            out_dim = 2**(len(ratios) - i - 1) * capacity

            net.append(
                UpsampleLayer(
                    in_dim,
                    out_dim,
                    r,
                    cumulative_delay=net[-1].cumulative_delay,
                ))
            net.append(
                ResidualStack(out_dim,
                              cumulative_delay=net[-1].cumulative_delay))

        self.net = cc.CachedSequential(*net)

        wave_gen = normalization(
            cc.Conv1d(out_dim, data_size, 7, padding=cc.get_padding(7)))

        loud_gen = normalization(
            cc.Conv1d(
                out_dim,
                1,
                2 * loud_stride + 1,
                stride=loud_stride,
                padding=cc.get_padding(2 * loud_stride + 1, loud_stride),
            ))

        branches = [wave_gen, loud_gen]

        if use_noise:
            noise_gen = NoiseGenerator(out_dim, data_size)
            branches.append(noise_gen)

        self.synth = cc.AlignBranches(
            *branches,
            cumulative_delay=self.net.cumulative_delay,
        )

        self.use_noise = use_noise
        self.loud_stride = loud_stride
        self.cumulative_delay = self.synth.cumulative_delay

        self.register_buffer("warmed_up", torch.tensor(0))

    def set_warmed_up(self, state: bool):
        state = torch.tensor(int(state), device=self.warmed_up.device)
        self.warmed_up = state

    def forward(self, x):
        x = self.net(x)

        if self.use_noise:
            waveform, loudness, noise = self.synth(x)
        else:
            waveform, loudness = self.synth(x)
            noise = torch.zeros_like(waveform)

        if self.loud_stride != 1:
            loudness = loudness.repeat_interleave(self.loud_stride)
        loudness = loudness.reshape(x.shape[0], 1, -1)

        waveform = torch.tanh(waveform) * mod_sigmoid(loudness)

        if self.warmed_up and self.use_noise:
            waveform = waveform + noise

        return waveform


class Encoder(nn.Module):

    def __init__(
        self,
        data_size,
        capacity,
        latent_size,
        ratios,
        n_out,
        sample_norm,
        repeat_layers,
        recurrent_layer: Optional[Callable[[], nn.Module]] = None,
    ):
        super().__init__()
        net = [cc.Conv1d(data_size, capacity, 7, padding=cc.get_padding(7))]

        for i, r in enumerate(ratios):
            in_dim = 2**i * capacity
            out_dim = 2**(i + 1) * capacity

            if sample_norm:
                net.append(SampleNorm())
            else:
                net.append(nn.BatchNorm1d(in_dim))
            net.append(nn.LeakyReLU(.2))
            net.append(
                cc.Conv1d(
                    in_dim,
                    out_dim,
                    2 * r + 1,
                    padding=cc.get_padding(2 * r + 1, r),
                    stride=r,
                    cumulative_delay=net[-3].cumulative_delay,
                ))

            for i in range(repeat_layers - 1):
                if sample_norm:
                    net.append(SampleNorm())
                else:
                    net.append(nn.BatchNorm1d(out_dim))
                net.append(nn.LeakyReLU(.2))
                net.append(
                    cc.Conv1d(
                        out_dim,
                        out_dim,
                        3,
                        padding=cc.get_padding(3),
                        cumulative_delay=net[-3].cumulative_delay,
                    ))

        net.append(nn.LeakyReLU(.2))

        if recurrent_layer is not None:
            net.append(
                recurrent_layer(
                    dim=out_dim,
                    cumulative_delay=net[-2].cumulative_delay,
                ))
            net.append(nn.LeakyReLU(.2))

        net.append(
            cc.Conv1d(
                out_dim,
                latent_size * n_out,
                5,
                padding=cc.get_padding(5),
                groups=n_out,
                cumulative_delay=net[-2].cumulative_delay,
            ))

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        z = self.net(x)
        return z


def normalize_dilations(dilations: Union[Sequence[int],
                                         Sequence[Sequence[int]]],
                        ratios: Sequence[int]):
    if isinstance(dilations[0], int):
        dilations = [dilations for _ in ratios]
    return dilations


class EncoderV2(nn.Module):

    def __init__(
        self,
        data_size: int,
        capacity: int,
        ratios: Sequence[int],
        latent_size: int,
        n_out: int,
        kernel_size: int,
        dilations: Sequence[int],
        keep_dim: bool = False,
        recurrent_layer: Optional[Callable[[], nn.Module]] = None,
        spectrogram: Optional[Callable[[], Spectrogram]] = None,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2),
        adain: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        dilations_list = normalize_dilations(dilations, ratios)

        if spectrogram is not None:
            self.spectrogram = spectrogram()
        else:
            self.spectrogram = None

        net = [
            normalization(
                cc.Conv1d(
                    data_size,
                    capacity,
                    kernel_size=kernel_size * 2 + 1,
                    padding=cc.get_padding(kernel_size * 2 + 1),
                )),
        ]

        num_channels = capacity
        for r, dilations in zip(ratios, dilations_list):
            # ADD RESIDUAL DILATED UNITS
            for d in dilations:
                if adain is not None:
                    net.append(adain(dim=num_channels))
                net.append(
                    Residual(
                        DilatedUnit(
                            dim=num_channels,
                            kernel_size=kernel_size,
                            dilation=d,
                        )))

            # ADD DOWNSAMPLING UNIT
            net.append(activation(num_channels))

            if keep_dim:
                out_channels = num_channels * r
            else:
                out_channels = num_channels * 2
            net.append(
                normalization(
                    cc.Conv1d(
                        num_channels,
                        out_channels,
                        kernel_size=2 * r,
                        stride=r,
                        padding=cc.get_padding(2 * r, r),
                    )))

            num_channels = out_channels

        net.append(activation(num_channels))
        net.append(
            normalization(
                cc.Conv1d(
                    num_channels,
                    latent_size * n_out,
                    kernel_size=kernel_size,
                    padding=cc.get_padding(kernel_size),
                )))

        if recurrent_layer is not None:
            net.append(recurrent_layer(latent_size * n_out))

        self.net = cc.CachedSequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spectrogram is not None:
            x = self.spectrogram(x[:, 0])[..., :-1]
            x = torch.log1p(x)

        x = self.net(x)
        return x


class GeneratorV2(nn.Module):

    def __init__(
        self,
        data_size: int,
        capacity: int,
        ratios: Sequence[int],
        latent_size: int,
        kernel_size: int,
        dilations: Sequence[int],
        keep_dim: bool = False,
        recurrent_layer: Optional[Callable[[], nn.Module]] = None,
        amplitude_modulation: bool = False,
        noise_module: Optional[NoiseGeneratorV2] = None,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2),
        adain: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        dilations_list = normalize_dilations(dilations, ratios)[::-1]
        ratios = ratios[::-1]

        if keep_dim:
            num_channels = np.prod(ratios) * capacity
        else:
            num_channels = 2**len(ratios) * capacity

        net = []

        if recurrent_layer is not None:
            net.append(recurrent_layer(latent_size))

        net.append(
            normalization(
                cc.Conv1d(
                    latent_size,
                    num_channels,
                    kernel_size=kernel_size,
                    padding=cc.get_padding(kernel_size),
                )), )

        for r, dilations in zip(ratios, dilations_list):
            # ADD UPSAMPLING UNIT
            if keep_dim:
                out_channels = num_channels // r
            else:
                out_channels = num_channels // 2
            net.append(activation(num_channels))
            net.append(
                normalization(
                    cc.ConvTranspose1d(num_channels,
                                       out_channels,
                                       2 * r,
                                       stride=r,
                                       padding=r // 2)))

            num_channels = out_channels

            # ADD RESIDUAL DILATED UNITS
            for d in dilations:
                if adain is not None:
                    net.append(adain(num_channels))
                net.append(
                    Residual(
                        DilatedUnit(
                            dim=num_channels,
                            kernel_size=kernel_size,
                            dilation=d,
                        )))

        net.append(activation(num_channels))

        waveform_module = normalization(
            cc.Conv1d(
                num_channels,
                data_size * 2 if amplitude_modulation else data_size,
                kernel_size=kernel_size * 2 + 1,
                padding=cc.get_padding(kernel_size * 2 + 1),
            ))

        self.noise_module = None
        self.waveform_module = None

        if noise_module is not None:
            self.waveform_module = waveform_module
            self.noise_module = noise_module(out_channels)
        else:
            net.append(waveform_module)

        self.net = cc.CachedSequential(*net)

        self.amplitude_modulation = amplitude_modulation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        x = self.net(x)

        noise = 0.

        if self.noise_module is not None:
            noise = self.noise_module(x)
            x = self.waveform_module(x)

        if self.amplitude_modulation:
            x, amplitude = x.split(x.shape[1] // 2, 1)
            x = x * torch.sigmoid(amplitude)

        x = x + noise

        return torch.tanh(x)

    def set_warmed_up(self, state: bool):
        pass

class VariationalEncoder(nn.Module):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder()
        self.register_buffer("warmed_up", torch.tensor(0))

    def reparametrize(self, z):
        mean, scale = z.chunk(2, 1)
        std = nn.functional.softplus(scale) + 1e-4
        var = std * std
        logvar = torch.log(var)

        z = torch.randn_like(mean) * std + mean
        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        return z, kl

    def set_warmed_up(self, state: bool):
        state = torch.tensor(int(state), device=self.warmed_up.device)
        self.warmed_up = state

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)

        #if self.warmed_up:
        #    z = z.detach()
        return z


class WasserteinEncoder(nn.Module):

    def __init__(
        self,
        encoder_cls,
        noise_augmentation: int = 0,
    ):
        super().__init__()
        self.encoder = encoder_cls()
        self.register_buffer("warmed_up", torch.tensor(0))
        self.noise_augmentation = noise_augmentation

    def compute_mean_kernel(self, x, y):
        kernel_input = (x[:, None] - y[None]).pow(2).mean(2) / x.shape[-1]
        return torch.exp(-kernel_input).mean()

    def compute_mmd(self, x, y):
        x_kernel = self.compute_mean_kernel(x, x)
        y_kernel = self.compute_mean_kernel(y, y)
        xy_kernel = self.compute_mean_kernel(x, y)
        mmd = x_kernel + y_kernel - 2 * xy_kernel
        return mmd

    def reparametrize(self, z):
        z_reshaped = z.permute(0, 2, 1).reshape(-1, z.shape[1])
        reg = self.compute_mmd(z_reshaped, torch.randn_like(z_reshaped))

        if self.noise_augmentation:
            noise = torch.randn(z.shape[0], self.noise_augmentation,
                                z.shape[-1]).type_as(z)
            z = torch.cat([z, noise], 1)

        return z, reg.mean()

    def set_warmed_up(self, state: bool):
        state = torch.tensor(int(state), device=self.warmed_up.device)
        self.warmed_up = state

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        if self.warmed_up:
            z = z.detach()
        return z


class DiscreteEncoder(nn.Module):

    def __init__(self,
                 encoder_cls,
                 vq_cls,
                 num_quantizers,
                 noise_augmentation: int = 0):
        super().__init__()
        self.encoder = encoder_cls()
        self.rvq = vq_cls()
        self.num_quantizers = num_quantizers
        self.register_buffer("warmed_up", torch.tensor(0))
        self.register_buffer("enabled", torch.tensor(0))
        self.noise_augmentation = noise_augmentation

    @torch.jit.ignore
    def reparametrize(self, z):
        if self.enabled:
            z, diff, _ = self.rvq(z)
        else:
            diff = torch.zeros_like(z).mean()

        if self.noise_augmentation:
            noise = torch.randn(z.shape[0], self.noise_augmentation,
                                z.shape[-1]).type_as(z)
            z = torch.cat([z, noise], 1)

        return z, diff

    def set_warmed_up(self, state: bool):
        state = torch.tensor(int(state), device=self.warmed_up.device)
        self.warmed_up = state

    def forward(self, x):
        z = self.encoder(x)
        return z


class SphericalEncoder(nn.Module):

    def __init__(self, encoder_cls: Callable[[], nn.Module]) -> None:
        super().__init__()
        self.encoder = encoder_cls()

    def reparametrize(self, z):
        norm_z = z / torch.norm(z, p=2, dim=1, keepdim=True)
        reg = torch.zeros_like(z).mean()
        return norm_z, reg

    def set_warmed_up(self, state: bool):
        pass

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return z


class Snake(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (self.alpha + 1e-9).reciprocal() * (self.alpha *
                                                       x).sin().pow(2)


class AdaptiveInstanceNormalization(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("mean_x", torch.zeros(cc.MAX_BATCH_SIZE, dim, 1))
        self.register_buffer("std_x", torch.ones(cc.MAX_BATCH_SIZE, dim, 1))
        self.register_buffer("learn_x", torch.zeros(1))
        self.register_buffer("num_update_x", torch.zeros(1))

        self.register_buffer("mean_y", torch.zeros(cc.MAX_BATCH_SIZE, dim, 1))
        self.register_buffer("std_y", torch.ones(cc.MAX_BATCH_SIZE, dim, 1))
        self.register_buffer("learn_y", torch.zeros(1))
        self.register_buffer("num_update_y", torch.zeros(1))

    def update(self, target: torch.Tensor, source: torch.Tensor,
               num_updates: torch.Tensor) -> None:
        bs = source.shape[0]
        target[:bs] += (source - target[:bs]) / (num_updates + 1)

    def reset_x(self):
        self.mean_x.zero_()
        self.std_x.zero_().add_(1)
        self.num_update_x.zero_()

    def reset_y(self):
        self.mean_y.zero_()
        self.std_y.zero_().add_(1)
        self.num_update_y.zero_()

    def transfer(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]

        x = (x - self.mean_x[:bs]) / (self.std_x[:bs] + 1e-5)
        x = x * self.std_y[:bs] + self.mean_y[:bs]

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return x

        if self.learn_y:
            mean = x.mean(-1, keepdim=True)
            std = x.std(-1, keepdim=True)

            self.update(self.mean_y, mean, self.num_update_y)
            self.update(self.std_y, std, self.num_update_y)
            self.num_update_y += 1

            return x

        else:
            if self.learn_x:
                mean = x.mean(-1, keepdim=True)
                std = x.std(-1, keepdim=True)

                self.update(self.mean_x, mean, self.num_update_x)
                self.update(self.std_x, std, self.num_update_x)
                self.num_update_x += 1

            if self.num_update_x and self.num_update_y:
                x = self.transfer(x)

            return x


def leaky_relu(dim: int, alpha: float):
    return nn.LeakyReLU(alpha)


def unit_norm_vector_to_angles(x: torch.Tensor) -> torch.Tensor:
    norms = x.flip(1).pow(2)
    norms[:, 1] += norms[:, 0]
    norms = norms[:, 1:]
    norms = norms.cumsum(1).flip(1).sqrt()
    angles = torch.arccos(x[:, :-1] / norms)
    angles[:, -1] = torch.where(
        x[:, -1] >= 0,
        angles[:, -1],
        2 * np.pi - angles[:, -1],
    )
    angles[:, :-1] = angles[:, :-1] / np.pi
    angles[:, -1] = angles[:, -1] / (2 * np.pi)
    return 2 * (angles - .5)


def angles_to_unit_norm_vector(angles: torch.Tensor) -> torch.Tensor:
    angles = (angles / 2 + .5) % 1
    angles[:, :-1] = angles[:, :-1] * np.pi
    angles[:, -1] = angles[:, -1] * (2 * np.pi)
    cos = angles.cos()
    sin = angles.sin().cumprod(dim=1)
    cos = torch.cat([
        cos,
        torch.ones(cos.shape[0], 1, cos.shape[-1]).type_as(cos),
    ], 1)
    sin = torch.cat([
        torch.ones(sin.shape[0], 1, sin.shape[-1]).type_as(sin),
        sin,
    ], 1)
    return cos * sin


def wrap_around_value(x: torch.Tensor, value: float = 1) -> torch.Tensor:
    return (x + value) % (2 * value) - value

# --------- ADDED BLOCKS --------------

class Discriminator(nn.Module):
    def __init__(self, in_size, capacity, multiplier, n_layers):
        super().__init__()

        net = [
            wn(cc.Conv1d(in_size, capacity, 15, padding=cc.get_padding(15)))
        ]
        net.append(nn.LeakyReLU(.2))

        for i in range(n_layers):
            net.append(
                wn(
                    cc.Conv1d(
                        capacity * multiplier**i,
                        min(1024, capacity * multiplier**(i + 1)),
                        41,
                        stride=multiplier,
                        padding=cc.get_padding(41, multiplier),
                        groups=multiplier**(i + 1),
                    )))
            net.append(nn.LeakyReLU(.2))

        net.append(
            wn(
                cc.Conv1d(
                    min(1024, capacity * multiplier**(i + 1)),
                    min(1024, capacity * multiplier**(i + 1)),
                    5,
                    padding=cc.get_padding(5),
                )))
        net.append(nn.LeakyReLU(.2))
        net.append(
            wn(cc.Conv1d(min(1024, capacity * multiplier**(i + 1)), 1, 1)))
        self.net = nn.ModuleList(net)

    def forward(self, x):
        feature = []
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.Conv1d):
                feature.append(x)
        return feature

class StackDiscriminators(nn.Module):
    def __init__(self, n_dis, *args, **kwargs):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [Discriminator(*args, **kwargs) for i in range(n_dis)], )

    def forward(self, x):
        features = []
        for layer in self.discriminators:
            features.append(layer(x))
            x = nn.functional.avg_pool1d(x, 2)
        return features
             
class FiLM(nn.Module):
    def __init__(
        self,
        cond_dim,  # dim of conditioning input
        batch_norm=True,
    ):
        super().__init__()
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(cond_dim // 2, affine=False)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:

        g, b = torch.chunk(cond, 2, dim=1)

        if self.batch_norm:
            x = self.bn(x)  # apply BatchNorm without affine
            
        x = (x * g) + b  # then apply conditional affine

        return x
    
class GeneratorV2Pitch(nn.Module):

    def __init__(
        self,
        data_size: int,
        capacity: int,
        ratios: Sequence[int],
        ratios_ex: Sequence[int],
        channels_ex: int,
        latent_size: int,
        kernel_size: int,
        dilations: Sequence[int],
        keep_dim: bool = False,
        recurrent_layer: Optional[Callable[[], nn.Module]] = None,
        amplitude_modulation: bool = False,
        noise_module: Optional[NoiseGeneratorV2] = None,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2),
        adain: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        dilations_list = normalize_dilations(dilations, ratios)[::-1]
        ratios = ratios[::-1]

        if keep_dim:
            num_channels = np.prod(ratios) * capacity
        else:
            num_channels = 2**len(ratios) * capacity

        net = []

        if recurrent_layer is not None:
            net.append(recurrent_layer(latent_size))

        net.append(
            normalization(
                cc.Conv1d(
                    latent_size,
                    num_channels,
                    kernel_size=kernel_size,
                    padding=cc.get_padding(kernel_size),
                )), )

        for r, dilations in zip(ratios, dilations_list):
            # ADD UPSAMPLING UNIT
            if keep_dim:
                out_channels = num_channels // r
            else:
                out_channels = num_channels // 2
            net.append(activation(num_channels))
            net.append(
                normalization(
                    cc.ConvTranspose1d(num_channels,
                                       out_channels,
                                       2 * r,
                                       stride=r,
                                       padding=r // 2)))

            num_channels = out_channels

            # ADD RESIDUAL DILATED UNITS
            for d in dilations:
                if adain is not None:
                    net.append(adain(num_channels))
                net.append(
                    Residual(
                        DilatedUnit(
                            dim=num_channels,
                            kernel_size=kernel_size,
                            dilation=d,
                        )))

        net.append(activation(num_channels))

        waveform_module = normalization(
            cc.Conv1d(
                num_channels,
                data_size * 2 if amplitude_modulation else data_size,
                kernel_size=kernel_size * 2 + 1,
                padding=cc.get_padding(kernel_size * 2 + 1),
            ))

        self.noise_module = None
        self.waveform_module = None

        if noise_module is not None:
            self.waveform_module = waveform_module
            self.noise_module = noise_module(out_channels)
        else:
            net.append(waveform_module)

        self.net = cc.CachedSequential(*net)
        self.amplitude_modulation = amplitude_modulation

        self.ex_down1 = normalization(cc.Conv1d(
                        16,
                        16,
                        kernel_size=2 * 1,
                        stride=1,
                        padding=cc.get_padding(2 * 1, 1),
                    ))
        
        self.ex_down2 = normalization(cc.Conv1d(
                        16,
                        16,
                        kernel_size=2 * 4,
                        stride=4,
                        padding=cc.get_padding(2 * 4, 4),
                    ))

        self.ex_down3 = normalization(cc.Conv1d(
                        16,
                        16,
                        kernel_size=2 * 4,
                        stride=4,
                        padding=cc.get_padding(2 * 4, 4),
                    ))

        self.ex_down4 = normalization(cc.Conv1d(
                        16,
                        16,
                        kernel_size=2 * 2,
                        stride=2,
                        padding=cc.get_padding(2 * 2, 2),
                    ))

        
        self.c_conv1 = normalization(cc.Conv1d(16,
                                               128,
                                               kernel_size=1,
                                               stride=1,
                                               padding=cc.get_padding(1)))
        
        self.c_conv2 = normalization(cc.Conv1d(16,
                                               256,
                                               kernel_size=1,
                                               stride=1,
                                               padding=cc.get_padding(1)))

        self.c_conv3 = normalization(cc.Conv1d(16,
                                               512,
                                               kernel_size=1,
                                               stride=1,
                                               padding=cc.get_padding(1)))

        self.c_conv4 = normalization(cc.Conv1d(16,
                                               1024,
                                               kernel_size=1,
                                               stride=1,
                                               padding=cc.get_padding(1)))
        """
        
        ex_net = []
        for r in ratios_ex:
            # ADD DOWNSAMPLING UNIT
            ex_net.append(
                normalization(
                    cc.Conv1d(
                        channels_ex,
                        channels_ex,
                        kernel_size=2 * r,
                        stride=r,
                        padding=cc.get_padding(2 * r, r),
                    )))

        self.ex_net = cc.CachedSequential(*ex_net)
        conv_net = []

        for i in range(len(ratios_ex)):
            # ADD DOWNSAMPLING UNIT
            conv_net.append(
                normalization(
                    cc.Conv1d(
                        channels_ex,
                        2**(i+3) * channels_ex,
                        kernel_size=1,
                        stride=1,
                        padding=cc.get_padding(1),
                    )))
            
        self.conv_net = cc.CachedSequential(*conv_net)
        """

        self.film1 = FiLM(1024)
        self.film2 = FiLM(512)
        self.film3 = FiLM(256)
        self.film4 = FiLM(128)
        
        """
        film_list = [1024, 512, 256, 128]
        self.film_net = nn.ModuleList([FiLM(i) for i in film_list])
        """

    def forward(self, x: torch.Tensor, ex: torch.Tensor) -> torch.Tensor:
        """
        downsampled_layers = []
        for down_layer, conv_layer in zip(self.ex_net, self.conv_net):
            ex = down_layer(ex) 
            x_conv = conv_layer(ex)
            downsampled_layers.append(x_conv)

        downsampled_layers.reverse()
        """

        ex_down1 = self.ex_down1(ex)
        print(ex_down1.shape)
        ex_conv1 = self.c_conv1(ex_down1)

        ex_down2 = self.ex_down2(ex_down1)
        print(ex_down2.shape)
        ex_conv2 = self.c_conv2(ex_down2)
        
        ex_down3 = self.ex_down3(ex_down2)
        print(ex_down3.shape)
        ex_conv3 = self.c_conv3(ex_down3)

        ex_down4 = self.ex_down4(ex_down3)
        print(ex_down4.shape)
        ex_conv4 = self.c_conv4(ex_down4)
        
        index = 0

        print("INPUT:", x.shape)
        
        for i, layer in enumerate(self.net):
            if i % 5 != 0 or i == 0:
                x = layer(x)
                print(i, x.shape)
            elif i == 5:
                print(i, layer(x).shape, ex_conv4.shape)
                x = self.film1(layer(x), ex_conv4)
                index += 1
            elif i == 10:
                print(i, layer(x).shape, ex_conv3.shape)
                x = self.film2(layer(x), ex_conv3)
                index += 1
            elif i == 15:
                print(i, layer(x).shape, ex_conv2.shape)
                x = self.film3(layer(x), ex_conv2)
                index += 1
            elif i == 20:
                print(i, layer(x).shape, ex_conv1.shape)
                x = self.film4(layer(x), ex_conv1)
                index += 1

        noise = 0.

        if self.noise_module is not None:
            noise = self.noise_module(x)
            x = self.waveform_module(x)

        if self.amplitude_modulation:
            x, amplitude = x.split(x.shape[1] // 2, 1)
            x = x * torch.sigmoid(amplitude)

        x = x + noise
        x = torch.tanh(x)

        return x

    def set_warmed_up(self, state: bool):
        pass



# ---------------
class ResBlock(nn.Module):
    """Residual block,"""

    def __init__(self, in_channels: int, out_channels: int, kernels: int):
        """Initializer.
        Args:
            in_channels: size of the input channels.
            out_channels: size of the output channels.
            kernels: size of the convolutional kernels.
        """
        super().__init__()
        self.branch = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Conv1d(
                in_channels, out_channels, kernels, padding=kernels // 2
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(
                out_channels, out_channels, kernels, padding=kernels // 2
            ),
        )

        self.shortcut = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform the inputs.
        Args:
            inputs: [torch.float32; [B, in_channels, F, N]], input channels.
        Returns:
            [torch.float32; [B, out_channels, F // 2, N]], output channels.
        """
        # [B, out_channels, F, N]
        outputs = self.branch(inputs)
        # [B, out_channels, F, N]
        shortcut = self.shortcut(inputs)
        # [B, out_channels, F // 2, N]
        return F.avg_pool1d(outputs + shortcut, 2)


def exponential_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Exponential sigmoid.
    Args:
        x: [torch.float32; [...]], input tensors.
    Returns:
        sigmoid outputs.
    """
    return 2.0 * torch.sigmoid(x) ** np.log(10) + 1e-7


class PitchEncoder(nn.Module):
    """Pitch-encoder."""

    freq: int
    """Number of frequency bins."""
    min_pitch: float
    """The minimum predicted pitch."""
    max_pitch: float
    """The maximum predicted pitch."""
    prekernels: int
    """Size of the first convolutional kernels."""
    kernels: int
    """Size of the frequency-convolution kernels."""
    channels: int
    """Size of the channels."""
    blocks: int
    """Number of residual blocks."""
    gru_dim: int
    """Size of the GRU hidden states."""
    hidden_channels: int
    """Size of the hidden channels."""
    f0_bins: int
    """Size of the output f0-bins."""
    f0_activation: str
    """F0 activation function."""

    def __init__(
        self,
        freq: int,
        min_pitch: float,
        max_pitch: float,
        prekernels: int,
        kernels: int,
        channels: int,
        blocks: int,
        gru_dim: int,
        hidden_channels: int,
        f0_bins: int,
        f0_activation: str,
    ):
        """Initializer.
        Args:
            freq: Number of frequency bins.
            min_pitch: The minimum predicted pitch.
            max_pitch: The maximum predicted pitch.
            prekernels: Size of the first convolutional kernels.
            kernels: Size of the frequency-convolution kernels.
            channels: Size of the channels.
            blocks: Number of residual blocks.
            gru_dim: Size of the GRU hidden states.
            hidden_channels: Size of the hidden channels.
            f0_bins: Size of the output f0-bins.
            f0_activation: f0 activation function.
        """
        super().__init__()

        self.freq = freq
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.prekernels = prekernels
        self.kernels = kernels
        self.channels = channels
        self.blocks = blocks
        self.gru_dim = gru_dim
        self.hidden_channels = hidden_channels
        self.f0_bins = f0_bins
        self.f0_activation = f0_activation

        assert f0_activation in ["softmax", "sigmoid", "exp_sigmoid"]

        # prekernels=7
        self.preconv = nn.Conv1d(
            128, channels, prekernels, padding=prekernels // 2
        )
        # channels=128, kernels=3, blocks=2
        self.resblock = nn.Sequential(
            *[ResBlock(channels, channels, kernels) for _ in range(blocks)]
        )
        # unknown `gru`
        self.gru = nn.GRU(
            freq // (2 * blocks),
            gru_dim,
            batch_first=True,
            bidirectional=True,
        )
        # unknown `hidden_channels`
        # f0_bins=64
        self.proj = nn.Sequential(
            nn.Linear(gru_dim * 2, hidden_channels * 2),
            nn.ReLU(),
            nn.Linear(hidden_channels * 2, f0_bins + 2),
        )

        self.register_buffer(
            "pitch_bins",
            # linear space in log-scale
            torch.linspace(
                np.log(self.min_pitch),
                np.log(self.max_pitch),
                self.f0_bins,
            ).exp(),
        )

    def forward(
        self, cqt_slice: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the pitch from inputs.
        Args:
            cqt_slice: [torch.float32; [B, F, N]]
                The input tensor. A frequency-axis slice of a Constant-Q Transform.
        Returns:
            pitch: [torch.float32; [B, N]], predicted pitch, based on frequency bins.
            f0_logits: [torch.float32; [B, N, f0_bins]], predicted f0 activation weights,
                based on the pitch bins.
            p_amp, ap_amp: [torch.float32; [B, N]], amplitude values.
        """
        # B, _, N
        batch_size, _, timesteps = cqt_slice.shape
        # [B, C, F, N]
        cqt_slice = cqt_slice.permute(0, 2, 1)
        x = self.preconv(cqt_slice)
        # [B, C F // 4, N]
        x = self.resblock(x)
        # [B, N, C x F // 4]
        #x = x.permute(0, 3, 1, 2).reshape(batch_size, timesteps, -1)
        #x = x.permute(0, 2, 1)
        # [B, N, G x 2]
        x, _ = self.gru(x)
        # [B, N, f0_bins], [B, N, 1], [B, N, 1]
        f0_weights, p_amp, ap_amp = torch.split(
            self.proj(x), [self.f0_bins, 1, 1], dim=-1
        )

        # potentially apply activation function
        if self.f0_activation == "softmax":
            f0_weights = torch.softmax(f0_weights, dim=-1)
        elif self.f0_activation == "sigmoid":
            f0_weights = torch.sigmoid(f0_weights) / torch.sigmoid(f0_weights).sum(
                dim=-1, keepdim=True
            )
        elif self.f0_activation == "exp_sigmoid":
            f0_weights = exponential_sigmoid(f0_weights) / exponential_sigmoid(
                f0_weights
            ).sum(dim=-1, keepdim=True)

        # [B, N]
        pitch = (f0_weights * self.pitch_bins).sum(dim=-1)

        return (
            pitch,
            f0_weights,
            exponential_sigmoid(p_amp).squeeze(dim=-1),
            exponential_sigmoid(ap_amp).squeeze(dim=-1),
        )

class ConvGLU(nn.Module):
    """Dropout - Conv1d - GLU and conditional layer normalization."""

    def __init__(
        self,
        channels: int,
        kernels: int,
        dilations: int,
        dropout: float,
        conditionning_channels: Optional[int] = None,
    ):
        """Initializer.
        Args:
            channels: size of the input channels.
            kernels: size of the convolutional kernels.
            dilations: dilation rate of the convolution.
            dropout: dropout rate.
            conditionning_channels: size of the condition channels, if provided.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels * 2,
                kernels,
                dilation=dilations,
                padding=(kernels - 1) * dilations // 2,
            ),
            nn.GLU(dim=1),
        )

        self.conditionning_channels = conditionning_channels
        if self.conditionning_channels is not None:
            self.cond = nn.Conv1d(self.conditionning_channels, channels * 2, 1)

    def forward(
        self, inputs: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Transform the inputs with given conditions.
        Args:
            inputs: [torch.float32; [B, channels, T]], input channels.
            cond: [torch.float32; [B, cond, T]], if provided.
        Returns:
            [torch.float32; [B, channels, T]], transformed.
        """
        # [B, channels, T]
        x = inputs + self.conv(inputs)
        if cond is not None:
            assert self.cond is not None, "condition module does not exists"
            # [B, channels, T]
            x = F.instance_norm(x, use_input_stats=True)
            # [B, channels, T]
            weight, bias = self.cond(cond).chunk(2, dim=1)
            # [B, channels, T]
            x = x * weight + bias
        return x

class CondSequential(nn.Module):
    """Sequential pass with conditional inputs."""

    def __init__(self, modules: Sequence[nn.Module]):
        """Initializer.
        Args:
            modules: list of torch modules.
        """
        super().__init__()
        self.lists = nn.ModuleList(modules)

    # def forward(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    def forward(
        self, inputs: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Pass the inputs to modules.
        Args:
            inputs: arbitary input tensors.
            args, kwargs: positional, keyword arguments.
        Returns:
            output tensor.
        """
        x = inputs
        for module in self.lists:
            # x = module.forward(x, *args, **kwargs)
            x = module.forward(x, cond)
        return x


class FrameLevelSynthesizer(nn.Module):
    """Frame-level synthesizer."""

    def __init__(
        self,
        in_channels: int,
        kernels: int,
        dilations: List[int],
        blocks: int,
        leak: float,
        dropout_rate: float,
        timbre_embedding_channels: Optional[int],
    ):
        """Initializer.
        Args:
            in_channels: The size of the input channels.
            kernels: The size of the convolutional kernels.
            dilations: The dilation rates.
            blocks: The number of the ConvGLU blocks after dilated ConvGLU.
            leak: The negative slope of the leaky relu.
            dropout_rate: The dropout rate.
            timbre_embedding_channels: The size of the time-varying timbre embeddings.
        """
        super().__init__()

        self.in_channels = in_channels
        self.timbre_embedding_channels = timbre_embedding_channels
        self.kernels = kernels
        self.dilations = dilations
        self.blocks = blocks
        self.leak = leak
        self.dropout_rate = dropout_rate

        # channels=1024
        # unknown `leak`, `dropout`
        self.preconv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1),
            nn.LeakyReLU(leak),
            nn.Dropout(dropout_rate),
        )
        # kernels=3, dilations=[1, 3, 9, 27, 1, 3, 9, 27], blocks=2
        self.convglus = CondSequential(
            [
                ConvGLU(
                    in_channels,
                    kernels,
                    dilation,
                    dropout_rate,
                    conditionning_channels=timbre_embedding_channels,
                )
                for dilation in dilations
            ]
            + [
                ConvGLU(
                    in_channels,
                    1,
                    1,
                    dropout_rate,
                    conditionning_channels=timbre_embedding_channels,
                )
                for _ in range(blocks)
            ]
        )

        self.proj = nn.Conv1d(in_channels, in_channels, 1)

    def forward(
        self, inputs: torch.Tensor, timbre: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Synthesize in frame-level.
        Args:
            inputs [torch.float32; [B, channels, T]]:
                Input features.
            timbre [torch.float32; [B, embed, T]], Optional:
                Time-varying timbre embeddings.
        Returns;
            [torch.float32; [B, channels, T]], outputs.
        """
        # [B, channels, T]
        x: torch.Tensor = self.preconv(inputs)
        # [B, channels, T]
        x = self.convglus(x, timbre)
        # [B, channels, T]
        return self.proj(x)