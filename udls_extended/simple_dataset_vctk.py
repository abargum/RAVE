### THIS SCRIPT BUILDS ON TOP OF THE UDLS DATA-READER
### INCORPORATING PERTURBATION AND SPEAKER EMBEDDINGS
### Inspiration https://github.com/wonjune-kang/lvc-vc and https://github.com/dhchoi99/NANSY 

from concurrent.futures import ProcessPoolExecutor, TimeoutError
from os import makedirs, path
from pathlib import Path

import librosa as li
import numpy as np
import torch
import os
from sklearn import mixture
from scipy.io.wavfile import read as read_wav_file
from tqdm import tqdm

from .base_dataset import SimpleLMDBDataset
from .ResNetSE34L import MainModel as ResNetModel
from .ecapa_tdnn import ECAPA_TDNN

import torchyin

def get_pitch(x, block_size, fs=48000, pitch_min=60):
    desired_num_frames = x.shape[-1] / block_size
    tau_max = int(fs / pitch_min)
    frame_length = 2 * tau_max
    frame_stride = (x.shape[-1] - frame_length) / (desired_num_frames - 1) / fs
    return torchyin.estimate(x, sample_rate=fs, pitch_min=pitch_min, pitch_max=800, frame_stride=frame_stride)

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


class SimpleDataset_VCTK(torch.utils.data.Dataset):
    def __init__(
        self,
        sampling_rate,
        speaker_model,
        device,
        out_database_location,
        folder_list=None,
        file_list=None,
        preprocess_function=dummy_load,
        transforms=None,
        extension="*.wav,*.aif,*.flac",
        map_size=1e11,
        split_percent=.2,
        split_set="train",
        seed=0,
    ):
        super().__init__()

        assert folder_list is not None or file_list is not None
        makedirs(out_database_location, exist_ok=True)

        self.env = SimpleLMDBDataset(out_database_location, map_size)
        
        self.speaker_model = speaker_model
        if speaker_model == "RESNET":
            self.speaker_encoder = self.load_resnet_encoder("speaker_embedding/resnet34sel_pretrained.pt", device)
        elif speaker_model == "ECAPA":
            self.speaker_encoder = self.load_ecapa_tdnn("speaker_embedding/ecapa_tdnn_pretrained.pt", device)
        else:
            print("PLEASE CHOOSE A SPEAKER ENCODER")

        self.folder_list = folder_list
        self.file_list = file_list

        self.preprocess_function = preprocess_function
        self.extension = extension

        self.transforms = transforms
        self.sampling_rate = sampling_rate
        self.device = device

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

    def load_resnet_encoder(self, checkpoint_path, device):
        model = ResNetModel(512).eval().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("loading speaker encoder")

        new_state_dict = {}
        for k, v in checkpoint.items():
            try:
                new_state_dict[k[6:]] = checkpoint[k]
            except KeyError:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        return model
    
    def load_ecapa_tdnn(self, checkpoint_path, device):
        ecapa_tdnn = ECAPA_TDNN(C=1024).eval().to(device)
        ecapa_checkpoint = torch.load(checkpoint_path, map_location=device)

        new_state_dict = {}
        for k, v in ecapa_checkpoint.items():
            if 'speaker_encoder' in k:
                key = k.replace('speaker_encoder.', '')
                new_state_dict[key] = ecapa_checkpoint[k]

        ecapa_tdnn.load_state_dict(new_state_dict)
        return ecapa_tdnn

    def _preprocess(self):
        extension = self.extension.split(",")
        idx = 0

        avg_speaker_embs = {}
        resnet_emb_gmms = {}
        f0_median = {}
        f0_std = {}
        f0_median_log = {}
        f0_std_log = {}

        # POPULATE WAV LIST
        if self.folder_list is not None:
            for f, folder in enumerate(self.folder_list.split(",")):
                print("SPEAKER EMBEDDINGS: Recursive search in {}".format(folder))
                if len(os.listdir(folder)) > 0:
                    for subfolder in os.listdir(folder):
                        if subfolder != '.DS_Store':
                            
                            wavs = []
                            utt_embeddings = []
                            all_f0_values = []
                            all_f0_values_log = []

                            speaker_id = subfolder
                            print("Calculating utterance embeddings for", speaker_id)
                            for ext in extension:
                                wavs.extend(list(Path(folder, speaker_id).rglob(ext)))

                            # CALCULATE UTTERANCE EMBEDDINGS
                            loader = tqdm(wavs)
                            for wav in loader:
                                loader.set_description("{}".format(path.basename(wav)))
                                output = self.preprocess_function(wav)
                                if output is not None:
                                    for o in output:
                                        if self.speaker_model == "RESNET":
                                            utt_emb = self.speaker_encoder(torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(torch.device(self.device)))
                                        else:
                                            utt_emb = self.speaker_encoder(torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(torch.device(self.device)), aug=False)
                                        utt_emb = utt_emb.detach().cpu().squeeze().numpy()
                                        utt_embeddings.append(utt_emb)

                                        #CALCULATE PITCH
                                        f0, voiced_flag, voiced_probs = li.pyin(
                                                o,
                                                fmin=li.note_to_hz('C2'),
                                                fmax=li.note_to_hz('C5'),
                                                sr=self.sampling_rate,
                                                frame_length=1024,
                                                hop_length=256,
                                                n_thresholds=5
                                            )

                                        f0[np.where(voiced_probs < 0.2)] = np.nan
                                        log_f0 = np.log(f0)

                                        f0 = f0[~np.isnan(f0)]
                                        log_f0 = log_f0[~np.isnan(log_f0)]

                                        all_f0_values.extend(f0.tolist())
                                        all_f0_values_log.extend(log_f0.tolist())                     

                            utt_embeddings = np.stack(utt_embeddings)
                            gmm_dvector = mixture.GaussianMixture(n_components=1, covariance_type="diag")
                            gmm_dvector.fit(utt_embeddings)

                            avg_speaker_embs[speaker_id] = np.mean(utt_embeddings, axis=0)
                            resnet_emb_gmms[speaker_id] = gmm_dvector

                            all_f0_values = np.array(all_f0_values)
                            all_f0_values_log = np.array(all_f0_values_log)

                            f0_median[speaker_id] = np.median(all_f0_values)
                            f0_std[speaker_id] = np.std(all_f0_values)
                            f0_median_log[speaker_id] = np.median(all_f0_values_log)
                            f0_std_log[speaker_id] = np.std(all_f0_values_log)

            wavs = []

            # DATA PROCESSING
            for f, folder in enumerate(self.folder_list.split(",")):
                print("DATA PROCESSING: Recursive search in {}".format(folder))
                if len(os.listdir(folder)) > 0:
                    for subfolder in os.listdir(folder):
                        for ext in extension:
                            wavs.extend(list(Path(folder).rglob(ext)))

            loader = tqdm(wavs)
            for wav in loader:
                speaker_id = str(wav).split("/")[1]
                loader.set_description("{}".format(path.basename(wav)))
                output = self.preprocess_function(wav)
                if output is not None:
                    for o in output:
                        speaker_emb, _ = resnet_emb_gmms[speaker_id].sample(1)
                        speaker_avg = avg_speaker_embs[speaker_id]
                        self.env[idx] = {
                            'data_clean': o,
                            'speaker_emb': speaker_emb[0],
                            'speaker_id': speaker_id,
                            'speaker_emb_avg': speaker_avg,
                            'f0_median': f0_median[speaker_id],
                            'f0_std': f0_std[speaker_id],
                            'f0_log_median': f0_median_log[speaker_id],
                            'f0_log_std': f0_std_log[speaker_id]
                        }

                        idx += 1

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        data = self.env[self.index[index + self.offset]]

        if self.transforms is not None:
            data_clean, data_perturbed_1 = self.transforms(data['data_clean'])

        return {
            'data_clean': data_clean.astype(np.float32),
            'data_perturbed_1': data_perturbed_1.astype(np.float32),
            'speaker_emb': data['speaker_emb'].astype(np.float32),
            'speaker_emb_avg': data['speaker_emb_avg'].astype(np.float32),
            'f0_median': data['f0_median'].astype(np.float32),
            'f0_std': data['f0_std'].astype(np.float32),
            'f0_log_median': data['f0_log_median'].astype(np.float32),
            'f0_log_std': data['f0_log_std'].astype(np.float32),
            'speaker_id': data['speaker_id'],
        }
