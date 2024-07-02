import base64
import logging
import math
import os
import subprocess
from random import random, randint
from typing import Dict, Iterable, Optional, Sequence

import gin
import lmdb
import numpy as np
import requests
import torch
import yaml
from scipy.signal import lfilter
from torch.utils import data
from tqdm import tqdm
from udls import AudioExample as AudioExampleWrapper
from udls import transforms
from udls.generated import AudioExample
from abc import ABC, abstractmethod
import pathlib
import gin

import wandb
from torchaudio.functional import resample
from .perturbation import wav_to_Sound, formant_and_pitch_shift, parametric_equalizer
from .core import fast_load, decoded_audio_duration

class Augmentation(ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def __call__(self, audio_data):
        pass

    @abstractmethod
    def sample(self, size, audio_length):
        pass

class Transform(object):
    def __call__(self, x: torch.Tensor):
        raise NotImplementedError
        
class FormantPitchShift(Transform):
    def __init__(self, sr):
        self.sr = sr

    def __call__(self, x: np.ndarray):
        sound = wav_to_Sound(x, sampling_frequency=self.sr)
        sound = formant_and_pitch_shift(sound).values
        return sound[0]

class PEQ(Transform):
    def __init__(self, sr):
        self.sr = sr

    def __call__(self, x: np.ndarray):
        return parametric_equalizer(x, self.sr)

"""Background noise from:
https://github.com/MWM-io/nansypp/blob/bbc6e390632bad56d5c0764074390a9f4a23fb86/src/data/preprocessing/augmentation.py"""
@gin.configurable
class RandomBackgroundNoise(Augmentation):
    def __init__(
        self,
        sample_rate: int,
        min_snr_db: int,  
        max_snr_db: int, 
        noise_scale: float,
        length_s: float,
        noise_dir: str,
    ) -> None:
        """
        Args:
            sample_rate: sample rate.
            noise_dir: directory containing noise audio samples
            min_snr_db: minimum source-to-noise-ratio in decibel used to generate signal scaling audio data
            max_snr_db: maximum source-to-noise-ratio in decibel used to generate signal scaling audio data
            noise_scale: noise signal weight
            augmentation_number: augmentation index used when composing
            length_s: segment length of audio data from dataset in seconds
        """
        self.sample_rate = sample_rate
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.noise_scale = noise_scale
        self.length_s = length_s
        self.length_f = int(sample_rate * length_s)

        if not os.path.exists(noise_dir):
            raise IOError(f"Noise directory `{noise_dir}` does not exist")
        # find all NPY files including in sub-folders:
        self.noise_files_list = list(pathlib.Path(noise_dir).rglob("*.npy"))
        if len(self.noise_files_list) == 0:
            raise IOError(
                f"No decoded .npy file found in the noise directory `{noise_dir}`"
            )
        self.noise_files_dict = {
            path: int(decoded_audio_duration(path, sample_rate) * sample_rate)
            for path in tqdm(self.noise_files_list)
        }

    def __call__(self, audio_data, noises=None):
        """Add random noise to the audio_data.
        Args:
            audio_data: [torch.float32; [B, T]], input tensor.
        Returns:
            [torch.float32; [B, T]], generated augmentation.
        """
        shape = audio_data.shape
        if len(audio_data.shape) == 1:
            audio_data = audio_data.reshape((1, -1))
        N, audio_length = audio_data.shape
        if noises is None:
            noises = self.sample(N, audio_length)
        noises_to_add = noises[:N, :audio_length]
        snr_db = randint(self.min_snr_db, self.max_snr_db)
        snr = math.exp(snr_db / 10)
        audio_power = np.linalg.norm(audio_data)
        noise_power = np.linalg.norm(noises_to_add)
        scale = (snr * noise_power / (audio_power + 1e-6)).reshape(-1, 1)
        result = (torch.tensor(scale) * torch.tensor(audio_data) + torch.tensor(self.noise_scale) * noises_to_add) / 2
        result = result.reshape(shape)
        return result.numpy()

    def sample(self, size, audio_length):
        file_indices = np.random.choice(len(self.noise_files_list), size, replace=False)
        return torch.vstack(
            [
                fast_load(
                    self.noise_files_list[file_idx],
                    audio_length,
                    np.random.randint(
                        0,
                        self.noise_files_dict[self.noise_files_list[file_idx]]
                        - audio_length,
                    ),
                )
                for file_idx in file_indices
            ]
        )


def get_derivator_integrator(sr: int):
    alpha = 1 / (1 + 1 / sr * 2 * np.pi * 10)
    derivator = ([.5, -.5], [1])
    integrator = ([alpha**2, -alpha**2], [1, -2 * alpha, alpha**2])

    return lambda x: lfilter(*derivator, x), lambda x: lfilter(*integrator, x)


class AudioDataset(data.Dataset):

    @property
    def env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(self._db_path, lock=False)
        return self._env

    @property
    def keys(self) -> Sequence[str]:
        if self._keys is None:
            with self.env.begin() as txn:
                self._keys = list(txn.cursor().iternext(values=False))
        return self._keys

    def __init__(self,
                 db_path: str,
                 audio_key: str = 'waveform',
                 transforms: Optional[transforms.Transform] = None) -> None:
        super().__init__()
        self._db_path = db_path
        self._audio_key = audio_key
        self._env = None
        self._keys = None
        self._transforms = transforms

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        with self.env.begin() as txn:
            ae = AudioExample.FromString(txn.get(self.keys[index]))

        buffer = ae.buffers[self._audio_key]
        assert buffer.precision == AudioExample.Precision.INT16

        audio = np.frombuffer(buffer.data, dtype=np.int16)
        audio = audio.astype(np.float32) / (2**15 - 1)

        if self._transforms is not None:
            audio = self._transforms(audio)

        return audio


class LazyAudioDataset(data.Dataset):

    @property
    def env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(self._db_path, lock=False)
        return self._env

    @property
    def keys(self) -> Sequence[str]:
        if self._keys is None:
            with self.env.begin() as txn:
                self._keys = list(txn.cursor().iternext(values=False))
        return self._keys

    def __init__(self,
                 db_path: str,
                 n_signal: int,
                 sampling_rate: int,
                 additive_noise: bool,
                 transforms: Optional[transforms.Transform] = None) -> None:
        super().__init__()
        self._db_path = db_path
        self._env = None
        self._keys = None
        self._transforms = transforms
        self._n_signal = n_signal
        self._sampling_rate = sampling_rate
               
        self.formant_pitch = FormantPitchShift(sampling_rate)
        self.peq = PEQ(sampling_rate)
        self.additive_noise = additive_noise
        
        if self.additive_noise:
            self.noise_module = RandomBackgroundNoise(sample_rate=sampling_rate,
                                                      min_snr_db=14,
                                                      max_snr_db=15,
                                                      noise_scale=0.8,
                                                      length_s=n_signal)
        
        self.parse_dataset()

    def parse_dataset(self):
        items = []
        for key in tqdm(self.keys, desc='Discovering dataset'):
            with self.env.begin() as txn:
                ae = AudioExample.FromString(txn.get(key))
            length = float(ae.metadata['length'])
            n_signal = int(math.floor(length * self._sampling_rate))
            n_chunks = n_signal // self._n_signal
            items.append(n_chunks)
        items = np.asarray(items)
        items = np.cumsum(items)
        self.items = items

    def __len__(self):
        return self.items[-1]

    def __getitem__(self, index):
        audio_id = np.where(index < self.items)[0][0]
        if audio_id:
            index -= self.items[audio_id - 1]

        key = self.keys[audio_id]

        with self.env.begin() as txn:
            ae = AudioExample.FromString(txn.get(key))

        audio = extract_audio(
            ae.metadata['path'],
            self._n_signal,
            self._sampling_rate,
            index * self._n_signal,
        )
        
        speaker_id = ae.metadata['path'].split('/')[-2]
      
        if self._transforms is not None:
            audio = self._transforms(audio)
            
        audio_p = self.formant_pitch(audio)
        audio_p = self.peq(audio_p)
        
        if self.additive_noise:
            noises = self.noise_module.sample(1, self._n_signal)
            audio_p = self.noise_module(audio_p, noises=noises)

        audio_p = (audio_p / np.max(audio_p)) * 0.8

        return (audio.astype(np.float32), audio_p.astype(np.float32), speaker_id)


class HTTPAudioDataset(data.Dataset):

    def __init__(self, db_path: str):
        super().__init__()
        self.db_path = db_path
        logging.info("starting remote dataset session")
        self.length = int(requests.get("/".join([db_path, "len"])).text)
        logging.info("connection established !")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        example = requests.get("/".join([
            self.db_path,
            "get",
            f"{index}",
        ])).text
        example = AudioExampleWrapper(base64.b64decode(example)).get("audio")
        return example.copy()


def normalize_signal(x: np.ndarray, max_gain_db: int = 30):
    peak = np.max(abs(x))
    if peak == 0: return x

    log_peak = 20 * np.log10(peak)
    log_gain = min(max_gain_db, -log_peak)
    gain = 10**(log_gain / 20)

    return x * gain


def get_dataset(db_path,
                sr,
                n_signal,
                additive_noise: bool,
                derivative: bool = False,
                normalize: bool = False):
    if db_path[:4] == "http":
        return HTTPAudioDataset(db_path=db_path)
    with open(os.path.join(db_path, 'metadata.yaml'), 'r') as metadata:
        metadata = yaml.safe_load(metadata)
    lazy = metadata['lazy']

    transform_list = [
        lambda x: x.astype(np.float32),
        transforms.RandomCrop(n_signal),
        transforms.RandomApply(
            lambda x: random_phase_mangle(x, 20, 2000, .99, sr),
            p=.8,
        ),
        transforms.Dequantize(16),
    ]

    if normalize:
        transform_list.append(normalize_signal)

    if derivative:
        transform_list.append(get_derivator_integrator(sr)[0])

    transform_list.append(lambda x: x.astype(np.float32))

    transform_list = transforms.Compose(transform_list)

    if lazy:
        return LazyAudioDataset(db_path, n_signal, sr, additive_noise, transform_list)
    else:
        return AudioDataset(
            db_path,
            transforms=transform_list,
        )


@gin.configurable
def split_dataset(dataset, percent, max_residual: Optional[int] = None):
    split1 = max((percent * len(dataset)) // 100, 1)
    split2 = len(dataset) - split1
    if max_residual is not None:
        split2 = min(max_residual, split2)
        split1 = len(dataset) - split2
    print(f'train set: {split1} examples')
    print(f'val set: {split2} examples')
    split1, split2 = data.random_split(
        dataset,
        [split1, split2],
        generator=torch.Generator().manual_seed(42),
    )
    return split1, split2


def random_angle(min_f=20, max_f=8000, sr=24000):
    min_f = np.log(min_f)
    max_f = np.log(max_f)
    rand = np.exp(random() * (max_f - min_f) + min_f)
    rand = 2 * np.pi * rand / sr
    return rand


def pole_to_z_filter(omega, amplitude=.9):
    z0 = amplitude * np.exp(1j * omega)
    a = [1, -2 * np.real(z0), abs(z0)**2]
    b = [abs(z0)**2, -2 * np.real(z0), 1]
    return b, a


def random_phase_mangle(x, min_f, max_f, amp, sr):
    angle = random_angle(min_f, max_f, sr)
    b, a = pole_to_z_filter(angle, amp)
    return lfilter(b, a, x)


def extract_audio(path: str, n_signal: int, sr: int,
                  start_sample: int) -> Iterable[np.ndarray]:
    start_sec = start_sample / sr
    length = n_signal / sr + 0.1
    process = subprocess.Popen(
        [
            'ffmpeg',
            '-v',
            'error',
            '-ss',
            str(start_sec),
            '-i',
            path,
            '-ar',
            str(sr),
            '-ac',
            '1',
            '-t',
            str(length),
            '-f',
            's16le',
            '-',
        ],
        stdout=subprocess.PIPE,
    )

    chunk = process.communicate()[0]

    chunk = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 2**15
    chunk = np.concatenate([chunk, np.zeros(n_signal)], -1)
    return chunk[:n_signal]
