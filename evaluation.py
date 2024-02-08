import torch

torch.set_grad_enabled(False)

from tqdm import tqdm

from rave import RAVE
from rave.core import search_for_run
from rave.core import random_phase_mangle, EMAModelCheckPoint

from effortless_config import Config
import os
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
from resemblyzer import preprocess_wav, VoiceEncoder
from speechmos import dnsmos
from sklearn.decomposition import PCA

torch.manual_seed(42)

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
        specgram = li.amplitude_to_db(specgram[0, :, :], ref=np.max)
    im = axs.imshow(specgram, origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.savefig(f'plots/{title}.png', bbox_inches='tight')
    
def get_entries_by_id(dataset, id_to_find):
    for index in range(len(dataset)):
        data_entry = dataset[index]
        if data_entry['speaker_id'] == id_to_find:
            return data_entry['speaker_id_avg'], data_entry['speaker_id'], data_entry['data_clean']

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

device = torch.device("cuda:0" if use_gpu else "cpu")

# ---------------------------------------------------
# LOAD TRAINING DATA FOR EMBEDDINGS
# ---------------------------------------------------
preprocess = lambda name: simple_audio_preprocess(
        16000,
        32768,
    )(name).astype(np.float16)

dataset = SimpleDataset(
        16000,
        "RESNET",
        torch.device('cuda:0'),
        args.PREPROCESSED,
        args.WAV,
        preprocess_function=preprocess,
        split_set="full",
        transforms=Perturb([
            lambda x, x_1: (x.astype(np.float32), x_1.astype(np.float32)),
        ], 16000))

dataset_embeddings = []
p225_entry = get_entries_by_id(dataset, 'p225')
dataset_embeddings.append(p225_entry)
p226_entry = get_entries_by_id(dataset, 'p226')
dataset_embeddings.append(p226_entry)
p227_entry = get_entries_by_id(dataset, 'p227')
dataset_embeddings.append(p227_entry)
p228_entry = get_entries_by_id(dataset, 'p228')
dataset_embeddings.append(p228_entry)

rave = RAVE.load_from_checkpoint(
    search_for_run(args.CKPT),
    strict=False,
).eval().to(device)

# ---------------------------------------------------
# PLOT SPEAKER EMBEDDINGS
# ---------------------------------------------------
speaker_E = []
speaker_ID = []

for i, entry in enumerate(dataset):
    if i < 250:
        speaker_E.append(entry['speaker_emb'])
        speaker_ID.append(entry['speaker_id'])
    else:
        break
    
pca = PCA(n_components=2)
components = pca.fit_transform(np.array(speaker_E))

fig, ax = plt.subplots()

# Create a dictionary to map unique speaker IDs to colors
unique_speakers = list(set(speaker_ID))
colors = plt.cm.get_cmap('tab10', len(unique_speakers))
color_dict = {speaker: colors(i) for i, speaker in enumerate(unique_speakers)}

# Scatter plot with different colors for each speaker ID
for i, speaker in enumerate(unique_speakers):
    indices = [j for j, s in enumerate(speaker_ID) if s == speaker]
    ax.scatter(components[indices, 0], components[indices, 1], label=speaker, color=color_dict[speaker])

# Display a legend outside the plot
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('plots/scatter_plot_embeddings.png', bbox_inches='tight')        

# ---------------------------------------------------
# LOAD UTIL MODELS
# ---------------------------------------------------
model = wenet.load_model('english')
resemblyzer = VoiceEncoder()
max_length = 300 #len(dataset)

step = 0
wer_full = 0
cer_full = 0
ssim = 0
ovrl_mos = 0
sig_mos = 0
bak_mos = 0

for i, example in enumerate(dataset):
    
    # ---------------------------------------------------
    # CREATE SPECTROGRAMS FOR 1 EXAMPLE
    # ---------------------------------------------------
    if i == 0:
        print("Creating Spectrograms..")
        length = len(example['data_clean'])
        x = example['data_clean'].reshape(int(length / 32768), -1)
        x = torch.from_numpy(x).float().to(device)
        embedding = torch.tensor(example['speaker_id_avg']).unsqueeze(0).to(device)
        
        y = x.reshape(-1, 1).cpu().numpy()
        plot_spectrogram(y, title="original")

        #NO SPEAKER INFO
        zeros = torch.zeros(embedding.repeat(x.shape[0], 1).shape).to(device)
        z, z_cat, sp = rave.encode(x, zeros)
        y = rave.decode(z_cat)
        y = y.reshape(-1, 1).cpu().numpy()
        plot_spectrogram(y, title="no_speaker")
        
        #NO CONTENT INFO
        zeros = torch.zeros(z.shape).to(device)
        z_cat = torch.cat((zeros, sp), 1)
        y = rave.decode(z_cat)
        y = y.reshape(-1, 1).cpu().numpy()
        plot_spectrogram(y, title="no_cont")
        
        print("Creating Conversions..")
    # ---------------------------------------------------
    # START CONVERSION AND OBJECTIVE METRICS
    # ---------------------------------------------------
    if i < max_length:
        
        target_text = get_txt_file(example)

        length = len(example['data_clean'])
        x = example['data_clean'].reshape(int(length / 32768), -1)
        x = torch.from_numpy(x).float().to(device)

        for speaker in dataset_embeddings:
            # ENCODE / DECODE
            if speaker[1] != example['speaker_id']:
                embedding = torch.tensor(speaker[0]).unsqueeze(0).to(device)
                
                file_path_conv = path.join(args.OUT, f"{example['speaker_id']}_{str(i)}_{speaker[1]}_conv.wav")

                z, z_cat, _ = rave.encode(x, embedding.repeat(x.shape[0], 1))
                y = rave.decode(z_cat)
                y = y.reshape(-1, 1).cpu().numpy()

                sf.write(file_path_conv, y, 16000)
                result = model.transcribe(file_path_conv)
                transcribed_text = str(result['text']).replace("▁", " ")

                w_error = wer(target_text, transcribed_text)
                wer_full += w_error

                c_error = cer(target_text, transcribed_text)
                cer_full += c_error
                
                #RESEMBLYZER 
                file_path_target = path.join(args.OUT, f"{example['speaker_id']}_{str(i)}_{speaker[1]}_target.wav")                
                sf.write(file_path_target, speaker[-1], 16000)
                
                conv = preprocess_wav(file_path_conv)
                target = preprocess_wav(file_path_target)
                
                conv_emb = resemblyzer.embed_utterance(conv)
                target_emb = resemblyzer.embed_utterance(target)

                cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                cos_dis = cos(torch.tensor(conv_emb), torch.tensor(target_emb))
                ssim += cos_dis.numpy()
                
                #DNS-MOS
                conv = conv / np.max(abs(conv))
                dnsmos_vals = dnsmos.run(conv, 16000)
                ovrl_mos += dnsmos_vals['ovrl_mos']
                sig_mos += dnsmos_vals['sig_mos']
                bak_mos += dnsmos_vals['bak_mos']
            
                step += 1
        
        if i % 50 == 0:
            print("Example:", i)
            
        i += 1
    else:
        break
    
    #print(target_text)
    #print(transcribed_text)

print("WER in %:", (wer_full / step) * 100)
print("CER in %:", (cer_full / step) * 100)
print("SSIM in %:", (ssim / step) * 100)
print("OVRL MOS:", (ovrl_mos / step))
print("SIG MOS:", (sig_mos / step))
print("BAK MOS:", (bak_mos / step))