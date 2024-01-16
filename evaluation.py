import torch

torch.set_grad_enabled(False)

from tqdm import tqdm

from rave import RAVE
from rave.core import search_for_run
from rave.core import random_phase_mangle, EMAModelCheckPoint

from effortless_config import Config
from os import path, makedirs, environ
from pathlib import Path

import librosa as li

import GPUtil as gpu
import torchaudio
import matplotlib.pyplot as plt

import soundfile as sf
import wenet
from jiwer import wer, cer

import numpy as np
from torch.utils.data import DataLoader, random_split

from udls_extended import SimpleDataset_VCTK as SimpleDataset
from udls_extended.simple_dataset_vctk import simple_audio_preprocess

from udls_extended.transforms import Compose, RandomApply, Dequantize, RandomCrop, Perturb

class args(Config):
    CKPT = None  # PATH TO YOUR PRETRAINED CHECKPOINT
    PREPROCESSED = None
    WAV = None
    OUT="files-for-eval"
    TXT_FOLDER = "txt-files"
    
args.parse_args()

def get_txt_file(example):
    txt_path = path.join(args.TXT_FOLDER, example['file_name']) + ".txt" 
    txt_file = open(txt_path)
    content = txt_file.read()
    txt_file.close()
    return " " + content.replace(",", "").replace(".", "").upper()

def plot_spectrogram(file, with_db=True, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1, figsize=(8,3))
    specgram = np.abs(li.stft(np.transpose(file)))
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    if with_db:
        specgram = li.amplitude_to_db(specgram[0, :250, 50:250], ref=np.max)
    im = axs.imshow(specgram, origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.savefig(f'spectrograms/{title}.png', bbox_inches='tight')

# GPU DISCOVERY
CUDA = gpu.getAvailable(maxMemory=.05)
if len(CUDA):
    environ["CUDA_VISIBLE_DEVICES"] = str(CUDA[0])
    use_gpu = 1
elif torch.cuda.is_available():
    print("Cuda is available but no fully free GPU found.")
    print("Reconstruction may be slower due to concurrent processes.")
    use_gpu = 1
else:
    print("No GPU found.")
    use_gpu = 0

device = torch.device("cuda:0" if use_gpu else "cpu")

# LOAD TRAINING DATA FOR EMBEDDINGS
preprocess = lambda name: simple_audio_preprocess(
        48000,
        65536,
    )(name).astype(np.float16)

dataset = SimpleDataset(
        48000,
        "RESNET",
        torch.device('cuda:0'),
        args.PREPROCESSED,
        args.WAV,
        preprocess_function=preprocess,
        split_set="full",
        transforms=Perturb([
            lambda x, x_p_1, x_p_2: (x.astype(np.float32), x_p_1.astype(np.float32), x_p_2.astype(np.float32)),
        ],
        48000))

rave = RAVE.load_from_checkpoint(
    search_for_run(args.CKPT),
    strict=False,
).eval().to(device)

model = wenet.load_model('english')
wer_full = 0
cer_full = 0

max_length = 300

for i, example in enumerate(dataset):
    if i < max_length:
        
        target_text = get_txt_file(example)
        file_path = path.join(args.OUT, f"{example['speaker_id']}_{str(i)}.wav")

        length = len(example['data_clean'])
        x = example['data_clean'].reshape(int(length / 65536), -1)

        # ENCODE / DECODE
        embedding = torch.tensor(example['speaker_id_avg']).unsqueeze(0).to(device)
        x = torch.from_numpy(x).float().to(device)
        
        """
        y = x.reshape(-1, 1).cpu().numpy()
        plot_spectrogram(y, title="recon")
        
        zeros = torch.zeros(embedding.repeat(x.shape[0], 1).shape).to(device)
        z, z_cat, sp = rave.encode(x, zeros)
        y = rave.decode(z_cat)
        y = y.reshape(-1, 1).cpu().numpy()
        plot_spectrogram(y, title="no_speaker")
        """

        z, z_cat, sp = rave.encode(x, embedding.repeat(x.shape[0], 1))
        y = rave.decode(z_cat)
        y = y.reshape(-1, 1).cpu().numpy()
        
        """
        zeros = torch.zeros(z.shape).to(device)
        z_cat = torch.cat((zeros, sp), 1)
        y = rave.decode(z_cat)
        y = y.reshape(-1, 1).cpu().numpy()
        plot_spectrogram(y, title="no_cont")
        """

        sf.write(file_path, y, 48000)
        result = model.transcribe(file_path)
        transcribed_text = str(result['text']).replace("▁", " ")
        
        w_error = wer(target_text, transcribed_text)
        wer_full += w_error
        
        c_error = cer(target_text, transcribed_text)
        cer_full += c_error
        
        if i % 50 == 0:
            print("Example:", i)
            
        i += 1
    else:
        break
    
    #print(target_text)
    #print(transcribed_text)

print("WER in %:", (wer_full / max_length) * 100)
print("CER in %:", (cer_full / max_length) * 100)
