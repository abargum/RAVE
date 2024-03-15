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

length = 65536
sr = 48000

# ---------------------------------------------------        
# GPU DISCOVERY
# ---------------------------------------------------
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

device = torch.device("cuda" if use_gpu else "cpu")

# ---------------------------------------------------
# LOAD TRAINING DATA FOR EMBEDDINGS
# ---------------------------------------------------
preprocess = lambda name: simple_audio_preprocess(
        sr,
        length,
    )(name).astype(np.float16)

dataset = SimpleDataset(
        sr,
        "RESNET",
        torch.device('cuda'),
        #REMEMBER TO CHANGE THIS
        'data/audio-eval-48k',
        # ----------------------
        'audio',
        preprocess_function=preprocess,
        split_set="full",
        transforms=Perturb([
            lambda x, x_p_1: (x.astype(np.float32), x_p_1.astype(np.float32)),
        ], sr))

#for i in range(20):
#    print(i, dataset[i]['speaker_id'])

target_index = 3
target = dataset[target_index]['data_clean']
embedding = torch.tensor(dataset[target_index]['speaker_id_avg']).unsqueeze(0).to(device)
print("Out ID:", dataset[target_index]['speaker_id'])

# LOAD RAVE
rave = RAVE.load_from_checkpoint(
    search_for_run(args.CKPT),
    strict=False,
).eval().to(device)

# SEARCH FOR WAV FILES
audios = tqdm(list(Path(args.WAV_FOLDER).rglob("*.wav")))

# RECONSTRUCTION
makedirs(args.OUT, exist_ok=True)
for audio in audios:
    audio_name = path.splitext(path.basename(audio))[0]
    audios.set_description(audio_name)

    # LOAD AUDIO TO TENSOR
    x, sr = li.load(audio, sr=rave.sr)
    
    val = int(np.ceil((len(x) / 65536)))
    N_pad = 65536 * val
    pad = (N_pad - (len(x) % N_pad)) % N_pad
    x = np.pad(x, (0, pad))
    x = x.reshape(-1, N_pad)
    
    x = torch.from_numpy(x).float().to(device)
    x = x.reshape(int(x.shape[-1] / 65536), -1)
    
    
    z, sp = rave.encode(x, embedding)
    y = rave.decode(z, sp)
    y = y.reshape(-1, 1).cpu().numpy()
    
    sf.write(path.join(args.OUT, f"reconstruction_{audio_name}.wav"), y, 48000)
    sf.write(path.join(args.OUT, f"input_{audio_name}.wav"), x.reshape(-1).cpu().numpy(), 48000)
    sf.write(path.join(args.OUT, "target.wav"), target, 48000)


    # PAD AUDIO
    #n_sample = x.shape[-1]
    #pad = (ratio - (n_sample % ratio)) % ratio
    
    
    #val = int(np.ceil((len(x) / 32768)))
    #N = 65536 * val
    #pad = (N - (len(x) % N)) % N
    #x = np.pad(x, (0, pad))
    #x = torch.tensor(x.reshape(val, -1)).to(device)
    #print(x.shape, N)
    
    #x = torch.nn.functional.pad(x, (0, pad))
    #x = x[:, 0:65536]

    # ENCODE / DECODE
    #z, z_cat = rave.encode(x, embedding.repeat(x.shape[0], 1))
    #y = rave.decode(z_cat)
    #y = y.reshape(-1).cpu().numpy()

    # WRITE AUDIO