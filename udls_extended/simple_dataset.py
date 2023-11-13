from concurrent.futures import ProcessPoolExecutor, TimeoutError
from os import makedirs, path
from pathlib import Path

import librosa as li
import numpy as np
import torch
from scipy.io.wavfile import read as read_wav_file
from tqdm import tqdm

from .base_dataset import SimpleLMDBDataset

from .perturbation import perturb


def dummy_load(name):
    """
    Preprocess function that takes one audio path and load it into
    chunks of 2048 samples.
    """
    x = li.load(name, 16000)[0]
    if len(x) % 2048:
        x = x[:-(len(x) % 2048)]
    x = x.reshape(-1, 2048)
    if x.shape[0]:
        return x
    else:
        return None


def simple_audio_preprocess(sampling_rate, N):
    def preprocess(name):
        try:
            x, sr = li.load(name, sr=sampling_rate)
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(e)
            return None

        pad = (N - (len(x) % N)) % N
        x = np.pad(x, (0, pad))

        x = x.reshape(-1, N)
        return x.astype(np.float32)

    return preprocess


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sampling_rate,
        out_database_location,
        folder_list=None,
        file_list=None,
        preprocess_function=dummy_load,
        transforms=None,
        extension="*.wav,*.aif",
        map_size=1e11,
        split_percent=.2,
        split_set="train",
        seed=0,
    ):
        super().__init__()

        assert folder_list is not None or file_list is not None
        makedirs(out_database_location, exist_ok=True)

        self.env = SimpleLMDBDataset(out_database_location, map_size)

        self.folder_list = folder_list
        self.file_list = file_list

        self.preprocess_function = preprocess_function
        self.extension = extension

        self.transforms = transforms
        self.sampling_rate = sampling_rate

        # IF NO DATA INSIDE DATASET: PREPROCESS
        self.len = len(self.env)

        if self.len == 0:
            self._preprocess()
            self.len = len(self.env)

        if self.len == 0:
            raise Exception("No data found !")

        self.index = np.arange(self.len)
        np.random.seed(seed)
        np.random.shuffle(self.index)

        if split_set == "train":
            self.len = int(np.floor((1 - split_percent) * self.len))
            self.offset = 0

        elif split_set == "test":
            self.offset = int(np.floor((1 - split_percent) * self.len))
            self.len = self.len - self.offset

        elif split_set == "full":
            self.offset = 0

    def _preprocess(self):
        extension = self.extension.split(",")
        idx = 0
        wavs = []

        # POPULATE WAV LIST
        if self.folder_list is not None:
            for f, folder in enumerate(self.folder_list.split(",")):
                print("Recursive search in {}".format(folder))
                for ext in extension:
                    wavs.extend(list(Path(folder).rglob(ext)))

        else:
            with open(self.file_list, "r") as file_list:
                wavs = file_list.read().split("\n")

        loader = tqdm(wavs)
        for wav in loader:
            loader.set_description("{}".format(path.basename(wav)))
            output = self.preprocess_function(wav)
            if output is not None:
                for o in output:
                    perturbed_o = perturb(o, self.sampling_rate)[0]
                    self.env[idx] = {
                        'data_clean': o,
                        'data_perturbed': perturbed_o
                    }
                    idx += 1

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        data = self.env[self.index[index + self.offset]]

        data_clean = data['data_clean']
        data_perturbed = data['data_perturbed']

        if self.transforms is not None:
            data_clean = self.transforms(data_clean)
            data_perturbed = self.transforms(data_perturbed)

        return {'data_clean': data_clean, 'data_perturbed': data_perturbed}
