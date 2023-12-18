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

import soundfile as sf

import numpy as np
from torch.utils.data import DataLoader, random_split

from udls_extended import SimpleDataset_VCTK as SimpleDataset
from udls_extended import simple_audio_preprocess

from udls_extended.transforms import Compose, RandomApply, Dequantize, RandomCrop, Perturb

class args(Config):
    CKPT = None  # PATH TO YOUR PRETRAINED CHECKPOINT
    WAV_FOLDER = None  # PATH TO YOUR WAV FOLDER
    OUT = "./reconstruction/"


args.parse_args()

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
        2 * 65536,
    )(name).astype(np.float16)

dataset = SimpleDataset(
        48000,
        "RESNET",
        torch.device('cuda'),
        "data/test-pitch/rave",
        "wav48_silence_trimmed",
        preprocess_function=preprocess,
        split_set="full",
        transforms=Perturb([
            lambda x, x_p: (x.astype(np.float32), x_p.astype(np.float32)),
        ],
        48000),
        seed=123)

#for i in range(20):
#    print(i, dataset[i]['speaker_id'])

in_index = 3
target_index = 9
seen = True
pitches = [-4.0, 1.0, 4.0]

in_sig = dataset[in_index]['data_clean']
print("In ID:", dataset[in_index]['speaker_id'], "out ID:", dataset[target_index]['speaker_id'])
print("Out ID:", dataset[target_index]['speaker_emb_avg'].shape)

target = dataset[target_index]['data_clean']
embedding = torch.tensor(dataset[target_index]['speaker_emb_avg']).unsqueeze(0).to(device)

# LOAD RAVE
rave = RAVE.load_from_checkpoint(
    search_for_run(args.CKPT),
    strict=False,
).eval().to(device)

# COMPUTE LATENT COMPRESSION RATIO
x = torch.randn(1, 2**16).to(device)
z, z_cat = rave.encode(x, dataset[in_index], embedding, 1.0, seen)
ratio = x.shape[-1] // z.shape[-1]

# SEARCH FOR WAV FILES
audios = tqdm(list(Path(args.WAV_FOLDER).rglob("*.wav")))

# RECONSTRUCTION
makedirs(args.OUT, exist_ok=True)
for audio in audios:
    for pitch in pitches:
        audio_name = path.splitext(path.basename(audio))[0]
        audios.set_description(audio_name)

        # LOAD AUDIO TO TENSOR
        x, sr = li.load(audio, sr=rave.sr)
        #x = in_sig
        x = torch.from_numpy(x).reshape(1, -1).float().to(device)

        # PAD AUDIO
        n_sample = x.shape[-1]
        pad = (ratio - (n_sample % ratio)) % ratio
        x = torch.nn.functional.pad(x, (0, pad))
        x = x[:, :65536]

        #embed = torch.zeros(embedding.shape).to(device)

        # ENCODE / DECODE
        z, z_cat = rave.encode(x, dataset[in_index], embedding, pitch, seen)
        y = rave.decode(z_cat)
        y = y.reshape(-1).cpu().numpy()[:n_sample]

        # WRITE AUDIO
        sf.write(path.join(args.OUT, f"reconstruction_{audio_name}_{pitch}.wav"), y, 48000)
        sf.write(path.join(args.OUT, f"input_{audio_name}.wav"), x.reshape(-1).cpu().numpy(), 48000)
        sf.write(path.join(args.OUT, "target.wav"), target, 48000)
