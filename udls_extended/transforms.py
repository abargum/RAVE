from random import choice, randint, random

import librosa as li
import numpy as np
import torch

from .perturbation import wav_to_Sound, formant_and_pitch_shift, parametric_equalizer


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


class RandomApply(Transform):
    """
    Apply transform with probability p
    """

    def __init__(self, transform, p=.5):
        self.transform = transform
        self.p = p

    def __call__(self, x: np.ndarray):
        if random() < self.p:
            x = self.transform(x)
        return x


class Compose(Transform):
    """
    Apply a list of transform sequentially
    """

    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, x: np.ndarray):
        for elm in self.transform_list:
            x = elm(x)
        return x
    

class Perturb(Transform):
    """
    Apply a list of perturbations and transform sequentially
    """

    def __init__(self, transform_list, sr):
        #self.peq_1 = PEQ(sr)
        #self.p_and_f_1 = FormantPitchShift(sr)
        #self.peq_2 = PEQ(sr)
        #self.p_and_f_2 = FormantPitchShift(sr)

        self.transform_list = transform_list

    def __call__(self, x: np.ndarray):
        #x_p_1 = self.peq_1(x)
        #x_p_1 = self.p_and_f_1(x_p_1)

        #x_p_2 = self.peq_2(x)
        #x_p_2 = self.p_and_f_2(x_p_2)

        for elm in self.transform_list:
            x = elm(x)
        return x


class RandomChoice(Transform):
    """
    Randomly select a transform from transform list and apply it
    """

    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, x: np.ndarray):
        x = choice(self.transform_list)(x)
        return x


class RandomCrop(Transform):
    """
    Randomly crops signal to fit n_signal samples
    """

    def __init__(self, n_signal):
        self.n_signal = n_signal

    def __call__(self, x: np.ndarray):
        in_point = randint(0, len(x) - self.n_signal)

        x = x[in_point:in_point + self.n_signal]

        return x


class Dequantize(Transform):

    def __init__(self, bit_depth):
        self.bit_depth = bit_depth

    def __call__(self, x: np.ndarray):
        rand = np.random.rand(len(x)) 
        
        x += rand / 2**self.bit_depth

        return x