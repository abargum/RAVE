import math
from time import time
from typing import Callable, Dict, Optional

import gin
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange
from sklearn.decomposition import PCA
from torchaudio.functional import resample
import torch.nn.functional as F

import wandb
import random

import rave.core

from . import blocks
from .my_discriminator import NewDiscriminator
from .stft_loss import MultiResolutionSTFTLoss
from .blocks import StackDiscriminators, PitchEncoder
from .core import load_speaker_statedict

from .excitation import ExcitationModule, Synthesizer
from .transform import ConstantQTransform
from .crepe import crepe_pitch_estimation
from .wavenet import WaveNet

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


class WarmupCallback(pl.Callback):

    def __init__(self) -> None:
        super().__init__()
        self.state = {'training_steps': 0}

    def on_train_batch_start(self, trainer, pl_module, batch,
                             batch_idx) -> None:
        if self.state['training_steps'] >= pl_module.warmup:
            pl_module.warmed_up = True
        self.state['training_steps'] += 1

    def state_dict(self):
        return self.state.copy()

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)


class QuantizeCallback(WarmupCallback):

    def on_train_batch_start(self, trainer, pl_module, batch,
                             batch_idx) -> None:

        if pl_module.warmup_quantize is None: return

        if self.state['training_steps'] >= pl_module.warmup_quantize:
            if isinstance(pl_module.encoder, blocks.DiscreteEncoder):
                pl_module.encoder.enabled = torch.tensor(1).type_as(
                    pl_module.encoder.enabled)
        self.state['training_steps'] += 1


@gin.configurable
class BetaWarmupCallback(pl.Callback):

    def __init__(self, initial_value: float, target_value: float,
                 warmup_len: int) -> None:
        super().__init__()
        self.state = {'training_steps': 0}
        self.warmup_len = warmup_len
        self.initial_value = initial_value
        self.target_value = target_value

    def on_train_batch_start(self, trainer, pl_module, batch,
                             batch_idx) -> None:
        self.state['training_steps'] += 1
        if self.state["training_steps"] >= self.warmup_len:
            pl_module.beta_factor = self.target_value
            return

        warmup_ratio = self.state["training_steps"] / self.warmup_len

        beta = math.log(self.initial_value) * (1 - warmup_ratio) + math.log(
            self.target_value) * warmup_ratio
        pl_module.beta_factor = math.exp(beta)

    def state_dict(self):
        return self.state.copy()

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

class CrossEntropyProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(128)
        self.proj = nn.Conv1d(64, 100, 1, bias=False)
        
    def forward(self, x):
        z_for_CE = self.layer_norm(x)
        z_for_CE = self.proj(z_for_CE)
        z_for_CE = F.interpolate(z_for_CE, 148)
        return z_for_CE

class RelativePitchDifferenceLoss(nn.Module):
    """Relative pitch difference loss.

    Introduced in Gfeller et al., 2020, _Spice: Self-supervised pitch estimation_
    """

    def __init__(
        self,
        cqt_center: int,
        cqt_shift_min: int,
        cqt_shift_max: int,
        pitch_freq: int,
        delta: float,
        sigma: float,
    ) -> None:
        super().__init__()
        self.cqt_center = cqt_center
        self.cqt_shift_min = cqt_shift_min
        self.cqt_shift_max = cqt_shift_max
        self.pitch_freq = pitch_freq
        self.delta = delta
        self.sigma = sigma

    def forward(
        self,
        cqt: torch.Tensor,
        pitch: torch.Tensor,
        pitch_encoder: PitchEncoder,
        bsize: int,
    ) -> torch.Tensor:
        
        random_transposition = torch.randint(
            self.cqt_shift_min,
            self.cqt_shift_max + 1,  # for inclusive range
            (bsize,),
            device=cqt.device,
        )
        # real start index
        start = random_transposition + self.cqt_center
        # sampled
        transposed_cqt = torch.stack(
            [cqt_[i : i + self.pitch_freq] for cqt_, i in zip(cqt, start)]
        )
        # [B, N]
        predicted_transposed_pitch, _, _, _ = pitch_encoder(transposed_cqt)

        # pitch consistency
        return F.huber_loss(
            predicted_transposed_pitch.log2()
            + self.sigma * random_transposition[:, None],
            pitch.log2(),
            delta=self.delta,
        )

class PitchSupervisionLoss(nn.Module):
    """Mean-square error loss between predicted pitches and ground truth pitches."""

    def __init__(self, sample_rate: int, cqt_hop_length: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.cqt_hop_length = cqt_hop_length

    def forward(
        self, audios: torch.Tensor, cqts: torch.Tensor, pred_pitches: torch.Tensor
    ) -> torch.Tensor:
        device = pred_pitches.device
        gt_pitches = []
        remove_pred_indices = []
        for idx, audio in enumerate(audios):
            gt_pitch = crepe_pitch_estimation(
                audio.unsqueeze(0),
                self.sample_rate,
                self.cqt_hop_length,
                device,
            )
            # handle crepe potential crash
            if gt_pitch is not None:
                gt_pitches.append(
                    crepe_pitch_estimation(
                        audio.unsqueeze(0),
                        self.sample_rate,
                        self.cqt_hop_length,
                        device,
                    )
                )
            else:
                remove_pred_indices.append(idx)
        gt_pitches = torch.cat(gt_pitches, dim=0).to(pred_pitches.device)[
            ..., : cqts.shape[-1]
        ]
        # delete in reverse order to not throw off subsequent indexes in list
        for idx in sorted(remove_pred_indices, reverse=True):
            del pred_pitches[indx]
        return nn.functional.mse_loss(pred_pitches, gt_pitches)


@gin.configurable
class RAVE(pl.LightningModule):

    def __init__(
        self,
        latent_size,
        sampling_rate,
        encoder,
        decoder,
        speaker_encoder,
        discriminator,
        phase_1_duration,
        gan_loss,
        valid_signal_crop,
        feature_matching_fun,
        num_skipped_features,
        audio_distance: Callable[[], nn.Module],
        multiband_audio_distance: Callable[[], nn.Module],
        weights: Dict[str, float],
        warmup_quantize: Optional[int] = None,
        pqmf: Optional[Callable[[], nn.Module]] = None,
        update_discriminator_every: int = 2,
        enable_pqmf_encode: bool = True,
        enable_pqmf_decode: bool = True,
        enable_training: bool = True,
    ):
        super().__init__()

        self.pqmf = None
        if pqmf is not None:
            self.pqmf = pqmf()

        self.encoder = encoder()
        self.decoder = decoder()

        #self.pqmf_speaker = pqmf()
        self.speaker_encoder = speaker_encoder()
        spk_state, pqmf_state = self.load_speaker_statedict("/home/jupyter-arbu/RAVE/rave/pretrained/model000000075.model")

        #ONLY LOAD PRETRAINED SPK_EMB WHEN TRAINING
        #(REMEMBER TO RESET THIS MANUALLY IN CONFIG WHEN EXPORTING)
        if enable_training:
            self.speaker_encoder.load_state_dict(spk_state)
            print("Loaded pretrained speaker encoder")
        else:
            print("Loaded my-trained speaker encoder")

        self.speaker_encoder.eval()

        # .... RAVE LOSS .... #
        #self.discriminator = discriminator()
        # ................... #

        # .... MY LOSS .... #
        
        self.new_discriminator = NewDiscriminator()
        self.discriminator = StackDiscriminators(
            3,
            in_size=1,
            capacity=16,
            multiplier=4,
            n_layers=4,
        )

        resolutions = []
        for hop_length_ms, win_length_ms in eval("[(5, 25), (10, 50), (2, 10)]"):
            hop_length = int(0.001 * hop_length_ms * 44100)
            win_length = int(0.001 * win_length_ms * 44100)
            n_fft = int(math.pow(2, int(math.log2(win_length)) + 1))
            resolutions.append((n_fft, hop_length, win_length))

        self.stft_criterion = MultiResolutionSTFTLoss(torch.device("cuda:2"), resolutions).cuda(2)
        
        # ............... #

        self.audio_distance = audio_distance()
        self.multiband_audio_distance = multiband_audio_distance()

        self.gan_loss = gan_loss

        self.register_buffer("latent_pca", torch.eye(latent_size))
        self.register_buffer("latent_mean", torch.zeros(latent_size))
        self.register_buffer("fidelity", torch.zeros(latent_size))

        self.latent_size = latent_size

        self.automatic_optimization = False

        # SCHEDULE
        self.warmup = phase_1_duration
        self.warmup_quantize = warmup_quantize
        self.weights = weights

        self.warmed_up = False

        # CONSTANTS
        self.sr = sampling_rate
        self.valid_signal_crop = valid_signal_crop
        self.feature_matching_fun = feature_matching_fun
        self.num_skipped_features = num_skipped_features
        self.update_discriminator_every = update_discriminator_every

        self.eval_number = 0
        self.beta_factor = 1.
        self.integrator = None

        self.enable_pqmf_encode = enable_pqmf_encode
        self.enable_pqmf_decode = enable_pqmf_decode

        self.register_buffer("receptive_field", torch.tensor([0, 0]).long())

        self.ce_projection = CrossEntropyProjection()
        self.discrete_units = torch.hub.load("bshall/hubert:main",f"hubert_discrete",
                                             trust_repo=True).to(torch.device("cuda:2"))

        self.excitation_module = ExcitationModule(44100) #Synthesizer(1024, 44100, 44100)
        #self.pitch_encoder = PitchEncoder(160, 50, 1000, 7, 3, 128, 2, 256, 256, 64, "sigmoid")
        #self.cqt_transform = ConstantQTransform(1024, 32.7, 190, 24, 44100)
        #self.p_loss = RelativePitchDifferenceLoss(15, -12, 12, 160, 0.125, 0.5)
        #self.p_loss = PitchSupervisionLoss(44100, 1024)
        self.wavenet = WaveNet(64, 16, 3, 2, 10, 3)

    def configure_optimizers(self):
        enc_p = list(self.encoder.parameters())
        enc_p += list(self.ce_projection.parameters())
        
        #gen_p = list(self.encoder.parameters())
        gen_p = list(self.decoder.parameters())
        gen_p += list(self.wavenet.parameters())
        #gen_p += list(self.speaker_encoder.parameters())
        #gen_p += list(self.pitch_encoder.parameters())
        
        dis_p = list(self.discriminator.parameters())
        dis_p += list(self.new_discriminator.parameters())

        enc_opt = torch.optim.Adam(enc_p, 1e-4, (.5, .9))
        gen_opt = torch.optim.Adam(gen_p, 1e-4, (.5, .9))
        
        dis_opt = torch.optim.Adam(dis_p, 1e-4, (.5, .9))

        return enc_opt, gen_opt, dis_opt

    def split_features(self, features):
        feature_real = []
        feature_fake = []
        for scale in features:
            true, fake = zip(*map(
                lambda x: torch.split(x, x.shape[0] // 2, 0),
                scale,
            ))
            feature_real.append(true)
            feature_fake.append(fake)
        return feature_real, feature_fake

    def load_speaker_statedict(self, path):
        loaded_state = torch.load(path, map_location="cuda:%d" % 2)
        
        newdict = {}
        pqmfdict = {}
        delete_list = []
        
        for name, param in loaded_state.items():
            new_name = name.replace("__S__.", "")
            
            if "pqmf" in new_name:
                new_name = new_name.replace("pqmf.", "")
                pqmfdict[new_name] = param
            else:
                newdict[new_name] = param
                
            delete_list.append(name)
        loaded_state.update(newdict)
        for name in delete_list:
            del loaded_state[name]
                
        return loaded_state, pqmfdict

    def training_step(self, batch, batch_idx):

        #num_frames = batch[0].shape[-1] // 1024
        #cqt = self.cqt_transform(batch[0])[:, :, :num_frames]
        #pitch, f0, p_amp, ap_amp = self.pitch_encoder(cqt[:, 15 : 15+160])
        #ex = self.excitation_module(pitch, p_amp, ap_amp)

        #pitch, _ = get_pitch(batch[0], 1024)
        ex = self.excitation_module(batch[0])

        x_resampled = resample(batch[0], self.sr, 16000)
        target_units = torch.zeros(x_resampled.shape[0], 148)
        
        for i, sequence in enumerate(x_resampled):
            target_units[i, :] = self.discrete_units.units(sequence.unsqueeze(0).unsqueeze(0))

        p = Profiler()
        enc_opt, gen_opt, dis_opt = self.optimizers()

        x = batch[0].unsqueeze(1)
        x_p = batch[1].unsqueeze(1)

        if self.pqmf is not None:
            x_multiband = self.pqmf(x)
            x_p_multiband = self.pqmf(x_p)
            #ex_multiband = self.pqmf(ex)
        else:
            x_multiband = x
            x_p_multiband = x_p
        p.tick('decompose')

        self.encoder.set_warmed_up(self.warmed_up)
        self.decoder.set_warmed_up(self.warmed_up)

        # ENCODE INPUT
        if self.enable_pqmf_encode:
            z_pre_reg = self.encoder(x_p_multiband[:, :6, :])
        else:
            z_pre_reg = self.encoder(x_p_multiband)

        projected_z = self.ce_projection(z_pre_reg)
        ce_loss = torch.nn.functional.cross_entropy(projected_z,
                                                    target_units.type(torch.int64).to(x.device))

        #pitch_loss = self.p_loss(cqt, pitch, self.pitch_encoder, batch[0].shape[0])
        #pitch_loss = self.p_loss(batch[0], cqt, pitch)

        #z, reg = self.encoder.reparametrize(z_pre_reg)[:2]
       
        with torch.no_grad():
            emb = self.speaker_encoder(x_multiband).unsqueeze(2)
        emb = emb.repeat(1, 1, z_pre_reg.shape[-1])

        p.tick('encode')

        z = torch.cat((z_pre_reg.detach(), emb), dim=1)

        # DECODE LATENT
        y_multiband = self.decoder(z)
        interp = torch.nn.functional.interpolate(
            y_multiband, size=ex.shape[-1], mode="linear"
        )

        y = self.wavenet(ex, interp).unsqueeze(1)

        p.tick('decode')

        if self.valid_signal_crop and self.receptive_field.sum():
            x_multiband = rave.core.valid_signal_crop(
                x_multiband,
                *self.receptive_field,
            )
            y_multiband = rave.core.valid_signal_crop(
                y_multiband,
                *self.receptive_field,
            )
        p.tick('crop')

        # DISTANCE BETWEEN INPUT AND OUTPUT
        distances = {}

        if self.pqmf is not None:

            # .... RAVE LOSS .... #
            #multiband_distance = self.multiband_audio_distance(
            #    x_multiband, y_multiband)
            #p.tick('mb distance')
            # ................... #

            x = self.pqmf.inverse(x_multiband)
            #y = self.pqmf.inverse(y_multiband)

            # .... MY LOSS .... #
            sc_loss, mag_loss = self.stft_criterion(y.squeeze(1), x.squeeze(1))
            distance = (sc_loss + mag_loss) * 2.5
            distances = distance
            # ................. #
            
            p.tick('recompose')

            # .... RAVE LOSS .... #
            #for k, v in multiband_distance.items():
            #    distances[f'multiband_{k}'] = v
            # ................... #
        else:
            x = x_multiband
            y = y_multiband

        # .... RAVE LOSS .... #
        #fullband_distance = self.audio_distance(x, y)
        #p.tick('fb distance')

        #for k, v in fullband_distance.items():
        #    distances[f'fullband_{k}'] = v
        # ................... #

        feature_matching_distance = 0.
        
        # .... RAVE LOSS .... #
        """
        if self.warmed_up:  # DISCRIMINATION
            xy = torch.cat([x, y], 0)
            features = self.discriminator(xy)

            feature_real, feature_fake = self.split_features(features)

            loss_dis = 0
            loss_adv = 0

            pred_real = 0
            pred_fake = 0

            for scale_real, scale_fake in zip(feature_real, feature_fake):

                current_feature_distance = sum(
                    map(
                        self.feature_matching_fun,
                        scale_real[self.num_skipped_features:],
                        scale_fake[self.num_skipped_features:],
                    )) / len(scale_real[self.num_skipped_features:])

                feature_matching_distance = feature_matching_distance + current_feature_distance
                
                feature_matching_distance = 0

                _dis, _adv = self.gan_loss(scale_real[-1], scale_fake[-1])

                pred_real = pred_real + scale_real[-1].mean()
                pred_fake = pred_fake + scale_fake[-1].mean()

                loss_dis = loss_dis + _dis
                loss_adv = loss_adv + _adv

            #feature_matching_distance = feature_matching_distance / len(
            #    feature_real)

        else:
            pred_real = torch.tensor(0.).to(x)
            pred_fake = torch.tensor(0.).to(x)
            loss_dis = torch.tensor(0.).to(x)
            loss_adv = torch.tensor(0.).to(x)
        """
        # ................... #
        # .... MY LOSS .... #
        if self.warmed_up:  # DISCRIMINATION
            
            loss_dis_lvc = 0
            loss_adv_lvc = 0
            loss_dis_rave = 0
            loss_adv_rave = 0

            pred_true = 0
            pred_fake = 0
            
            # MRD, MPD losses.
            res_fake, period_fake = self.new_discriminator(y)

            # Compute LSGAN loss for all frames.
            for (_, score_fake) in res_fake + period_fake:
                loss_adv_lvc += torch.mean(torch.pow(score_fake - 1.0, 2))

            # Average across frames.
            loss_adv_lvc = loss_adv_lvc / len(res_fake + period_fake)

            # MRD, MPD losses.
            res_fake, period_fake = self.new_discriminator(y.detach())  # fake audio from generator
            res_real, period_real = self.new_discriminator(x)  # real audio

            # Compute LSGAN loss for all frames.
            for (_, score_fake), (_, score_real) in zip(
                res_fake + period_fake, res_real + period_real
            ):
                loss_dis_lvc += torch.mean(torch.pow(score_real - 1.0, 2))
                loss_dis_lvc += torch.mean(torch.pow(score_fake, 2))

            # Compute average to get overall discriminator loss (L_D).
            loss_dis_lvc = loss_dis_lvc / len(res_fake + period_fake)
            
            feature_true = self.discriminator(x)
            feature_fake = self.discriminator(y)

            for scale_true, scale_fake in zip(feature_true, feature_fake):

                _dis = torch.relu(1 - scale_true[-1]) + torch.relu(1 + scale_fake[-1])
                _dis = _dis.mean()
                _adv = -scale_fake[-1].mean()

                pred_true = pred_true + scale_true[-1].mean()
                pred_fake = pred_fake + scale_fake[-1].mean()

                loss_dis_rave = loss_dis_rave + _dis
                loss_adv_rave = loss_adv_rave + _adv
        else:
            pred_true = torch.tensor(0.).to(x)
            pred_fake = torch.tensor(0.).to(x)
            loss_dis_lvc = torch.tensor(0.).to(x)
            loss_adv_lvc = torch.tensor(0.).to(x)
            loss_dis_rave = torch.tensor(0.).to(x)
            loss_adv_rave = torch.tensor(0.).to(x)

            
        loss_dis = loss_dis_lvc + loss_dis_rave * 0.1
        loss_adv = loss_adv_lvc + (loss_adv_rave) * 0.1
        # ................. #
        
        p.tick('discrimination')

        # COMPOSE GEN LOSS
        loss_gen = {}

         # .... RAVE LOSS .... #
        #loss_gen.update(distances)
        
         # .... MY LOSS .... #
        loss_gen['audio'] = distances
        #loss_gen['pitch'] = pitch_loss * 10
        
        p.tick('update loss gen dict')

        #if reg.item():
        #    loss_gen['regularization'] = reg * self.beta_factor

        if self.warmed_up:
            loss_gen['feature_matching'] = feature_matching_distance
            loss_gen['adversarial'] = loss_adv

        # OPTIMIZATION
        if not (batch_idx %
                self.update_discriminator_every) and self.warmed_up:
            dis_opt.zero_grad()
            loss_dis.backward()
            dis_opt.step()
            p.tick('dis opt')
        else:
            enc_opt.zero_grad()
            gen_opt.zero_grad()
            ce_loss.backward(retain_graph=True)
            loss_gen_value = 0.
            for k, v in loss_gen.items():
                loss_gen_value += v * self.weights.get(k, 1.)
            loss_gen_value.backward()
            enc_opt.step()
            gen_opt.step()

        # LOGGING
        self.log("beta_factor", self.beta_factor)

        if self.warmed_up:
            self.log("loss_dis", loss_dis)
            #self.log("pred_real", pred_real.mean())
            #self.log("pred_fake", pred_fake.mean())

        # .... MY LOSS .... #
        wandb.log({
            "stft": distances,
            "loss_dis": loss_dis,
            "loss_gen": loss_gen,
            "dis rave": loss_dis,
            "adv rave": loss_adv,
            "unit_loss": ce_loss,
            "dis lvc": loss_dis_lvc,
            "dis rave": loss_dis_rave,
            "adv lvc": loss_adv_lvc,
            "adv rave": loss_adv_rave
        })
        # ................. #
        # .... RAVE LOSS .... #
        #wandb.log({
        #    "loss_dis": loss_dis,
        #    "loss_gen": loss_gen,
        #    "unit_loss": ce_loss
        #})
        # ................... #

        self.log_dict(loss_gen)
        p.tick('logging')

    def encode(self, x):
        
        if self.pqmf is not None and self.enable_pqmf_encode:
            x = self.pqmf(x)

        z = self.encoder(x[:, :6, :])

        emb = self.speaker_encoder(x).unsqueeze(-1)
        emb = emb.repeat(z.shape[0], 1, z.shape[-1])
        
        z = torch.cat((z, emb), dim=1)
        return z

    def decode(self, z, x):

        ex = self.excitation_module(x.squeeze(1), torch.ones(1)).unsqueeze(1)

        ex_multiband = self.pqmf(ex)
        
        y = self.decoder(z, ex_multiband)
        if self.pqmf is not None and self.enable_pqmf_decode:
            y = self.pqmf.inverse(y)
        return y

    def forward(self, x):
        return self.decode(self.encode(x), x)

    def validation_step(self, batch, batch_idx):

        #num_frames = batch[0].shape[-1] // 1024
        #cqt = self.cqt_transform(batch[0])[:, :, :num_frames]
        #pitch, f0, p_amp, ap_amp = self.pitch_encoder(cqt[:, :160])
        #ex = self.excitation_module(pitch, p_amp, ap_amp)

        #pitch, pitch_stats = get_pitch(batch[0], 1024)
        #print(pitch.shape)
        ex = self.excitation_module(batch[0])

        input_ex = ex
        #print(input_ex.shape)
        
        x = batch[0].unsqueeze(1)

        if self.pqmf is not None:
            x_multiband = self.pqmf(x)
            #ex_multiband = self.pqmf(ex)

        if self.enable_pqmf_encode:
            z = self.encoder(x_multiband[:, :6, :])

        else:
            z = self.encoder(x)

        if isinstance(self.encoder, blocks.VariationalEncoder):
            mean = torch.split(z, z.shape[1] // 2, 1)[0]
        else:
            mean = None

        mean = None

        with torch.no_grad():
            emb = self.speaker_encoder(x_multiband).unsqueeze(2)

        emb = emb.repeat(1, 1, z.shape[-1])
        z = torch.cat((z.detach(), emb), dim=1)
        
        #z = self.encoder.reparametrize(z)[0]
        y = self.decoder(z)

        interp = torch.nn.functional.interpolate(
            y, size=ex.shape[-1], mode="linear"
        )

        y = self.wavenet(ex, interp).unsqueeze(1)

        if self.pqmf is not None:
            x = self.pqmf.inverse(x_multiband)
            #y = self.pqmf.inverse(y)

        distance = self.audio_distance(x, y)

        full_distance = sum(distance.values())

        if self.trainer is not None:
            self.log('validation', full_distance)

        #FOR LOGGING CONVERSION
        ids = batch[-1]
        unique_elements = list(set(ids))
        
        if len(unique_elements) <= 2:
            inp_ind = 0
            tar_ind = 1
        else:
            chosen_elements = random.sample(unique_elements, 2)
            inp_ind, tar_ind = [ids.index(element) for element in chosen_elements]

        inp = batch[0][inp_ind].unsqueeze(0)
        tar = batch[0][tar_ind].unsqueeze(0)

        #cqt_s = self.cqt_transform(inp)[:, :, :num_frames]
        #pitch_s, f0_s, p_amp_s, ap_amp_s = self.pitch_encoder(cqt_s[:, :160])

        #pitch_s, pitch_stats_s = get_pitch(inp, 1024)

        #cqt_t = self.cqt_transform(tar)[:, :, :num_frames]
        #pitch_t, f0_t, p_amp_t, ap_amp_t = self.pitch_encoder(cqt_t[:, :160])

        #pitch_t, pitch_stats_t = get_pitch(tar, 1024)

        #standardized_source_pitch = (pitch_s - pitch_stats_s.mean()) / pitch_stats_s.std()
        #source_pitch = (standardized_source_pitch * pitch_stats_t.std() + pitch_stats_t.mean())
        
        #ex = self.excitation_module(source_pitch, p_amp_s, ap_amp_s)
        ex = self.excitation_module(inp)

        if self.pqmf is not None:
            inp_multiband = self.pqmf(inp.unsqueeze(1))
            tar_multiband = self.pqmf(tar.unsqueeze(1))
            #ex_multiband = self.pqmf(ex)

        if self.enable_pqmf_encode:
            z = self.encoder(inp_multiband[:, :6, :])

        with torch.no_grad():
            tar_emb = self.speaker_encoder(tar_multiband).unsqueeze(2)

        tar_emb = tar_emb.repeat(1, 1, z.shape[-1])
        z = torch.cat((z.detach(), tar_emb), dim=1)

        y_conv = self.decoder(z)
        interp = torch.nn.functional.interpolate(
            y_conv, size=ex.shape[-1], mode="linear"
        )

        outp = self.wavenet(ex, interp).unsqueeze(1)
        
        #if self.pqmf is not None:
            #outp = self.pqmf.inverse(y_conv)

        return torch.cat([x, y], -1), mean, torch.cat([inp.unsqueeze(0), tar.unsqueeze(0), outp], -1), torch.cat([x, input_ex.unsqueeze(1)], -1)

    def validation_epoch_end(self, out):

        """
        if not self.receptive_field.sum():
            print("Computing receptive field for this configuration...")
            lrf, rrf = rave.core.get_rave_receptive_field(self)
            self.receptive_field[0] = lrf
            self.receptive_field[1] = rrf
            print(
                f"Receptive field: {1000*lrf/self.sr:.2f}ms <-- x --> {1000*rrf/self.sr:.2f}ms"
            )

        if not len(out): return
        """

        audio, z, conv, ex = list(zip(*out))
        audio = list(map(lambda x: x.cpu(), audio))
        conv = list(map(lambda x: x.cpu(), conv))
        ex = list(map(lambda x: x.cpu(), ex))
        
        # LATENT SPACE ANALYSIS
        """
        if not self.warmed_up and isinstance(self.encoder,
                                             blocks.VariationalEncoder):
            z = torch.cat(z, 0)
            z = rearrange(z, "b c t -> (b t) c")

            self.latent_mean.copy_(z.mean(0))
            z = z - self.latent_mean

            pca = PCA(z.shape[-1]).fit(z.cpu().numpy())

            components = pca.components_
            components = torch.from_numpy(components).to(z)
            self.latent_pca.copy_(components)

            var = pca.explained_variance_ / np.sum(pca.explained_variance_)
            var = np.cumsum(var)

            self.fidelity.copy_(torch.from_numpy(var).to(self.fidelity))

            var_percent = [.8, .9, .95, .99]
            for p in var_percent:
                self.log(
                    f"fidelity_{p}",
                    np.argmax(var > p).astype(np.float32),
                )
        """

        y = torch.cat(audio, 0)[:8].reshape(-1).numpy()

        if self.integrator is not None:
            y = self.integrator(y)
            
        wandb.log({
                f"audio_val_{self.eval_number}":
                wandb.Audio(y,
                            caption="audio",
                            sample_rate=self.sr)
            })

        conv_y = torch.cat(conv, 0)[:8].reshape(-1).numpy()

        if self.integrator is not None:
            conv_y = self.integrator(conv_y)
            
        wandb.log({
                f"audio_conv_{self.eval_number}":
                wandb.Audio(conv_y,
                            caption="audio",
                            sample_rate=self.sr)
            })

        ex_y = torch.cat(ex, 0)[:8].reshape(-1).numpy()

        if self.integrator is not None:
            ex_y = self.integrator(ex_y)
            
        wandb.log({
                f"audio_ex_{self.eval_number}":
                wandb.Audio(ex_y,
                            caption="audio",
                            sample_rate=self.sr)
            })

        self.logger.experiment.add_audio("audio_val", y, self.eval_number,
                                         self.sr)
        self.eval_number += 1

    def on_fit_start(self):
        tb = self.logger.experiment

        config = gin.operative_config_str()
        config = config.split('\n')
        config = ['```'] + config + ['```']
        config = '\n'.join(config)
        tb.add_text("config", config)

        model = str(self)
        model = model.split('\n')
        model = ['```'] + model + ['```']
        model = '\n'.join(model)
        tb.add_text("model", model)
