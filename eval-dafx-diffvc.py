import torch
import sys

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
from rave.model import FiLM

from udls_extended.ResNetSE34L import MainModel as ResNetModel
import random
from diffvc import get_converison_from_diffvc
import time as timer

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
device = device = torch.device("cuda:0")

# ---------------------------------------------------        
# ARGS
# ---------------------------------------------------
torch.manual_seed(42)

class args(Config):
    CKPT = "pretrained/film-dec-48k-nofm-full/checkpoints/last-newest.ckpt"
    WAV_FOLDER = "vctk-source-unseen"
    OUT="dafx-evaluation/files-for-dafx-eval"
    
args.parse_args()

SAMPLE_LENGTH = 65536
SR = 48000


rave = RAVE.load_from_checkpoint(
    search_for_run(args.CKPT),
    strict=False,
).eval().to(device)

"""
# ---------------------------------------------------        
# SPEED
# ---------------------------------------------------
num_trials = 10
dummy = np.random.rand(22050)
sf.write("dummy_in.wav", dummy, 22050) 

time = 0
for i in range(num_trials):
    print(i)
    _, start, end = diff_conv = get_converison_from_diffvc("dummy_in.wav", "dummy_in.wav")
    time += (end - start)
    
synthesis_speed = time / num_trials

print("AVERAGE ITERATION -------- :", synthesis_speed)
print("SYNTHESIS SPEED -------- :", (22050 / synthesis_speed) / 1000, "kHz")
print("REAL TIME FACTOR -------- :", (time / num_trials))

dummy = torch.rand(1, 65536).to(device)

start_time = timer.time()
for _ in range(num_trials):
    z, sp = rave.encode(dummy, torch.rand(1, 512).to(device))
    y = rave.decode(z, sp)
end_time = timer.time()
    
synthesis_speed = (((end_time - start_time) / num_trials) * 48000) / 65536

print("AVERAGE ITERATION -------- :", synthesis_speed)
print("SYNTHESIS SPEED -------- :", (48000 / synthesis_speed) / 1000, "kHz")
print("REAL TIME FACTOR -------- :", ((end_time - start_time) / num_trials) / (65536/48000))
"""
# ---------------------------------------------------        
# FUNCTIONS
# ---------------------------------------------------
def pick_random_speaker(target_folder, speaker):
    folder_path = target_folder + "/" + speaker
    
    if not os.path.isdir(folder_path):
        print("Error: The provided path is not a directory.")
        return None

    files = os.listdir(folder_path)
    wav_files = [file for file in files if file.lower().endswith('.flac')]

    if not wav_files:
        print("Error: No WAV files found in the directory.")
        return None

    random_wav_file = random.choice(wav_files)
    return os.path.join(folder_path, random_wav_file)

def load_resnet_encoder(checkpoint_path, device):
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
    
def get_txt_file(audio_name):
    if '_mic2' in audio_name:
        audio_name = audio_name.replace('_mic2', '')
    elif '_mic1' in audio_name:
        audio_name = audio_name.replace('_mic1', '')
    audio_txt_path = path.join(args.WAV_FOLDER, audio_name) + ".txt" 
    txt_file = open(audio_txt_path)
    content = txt_file.read()
    txt_file.close()
    return " " + content.replace(',', "").replace('.', "").replace('"', "").upper()

# ---------------------------------------------------        
# LOAD TARGET EMBEDDINGS
# ---------------------------------------------------
preprocess = lambda name: simple_audio_preprocess(
        SR,
        SAMPLE_LENGTH,
    )(name).astype(np.float32)

extension = "*.wav,*.aif,*.flac".split(",")
speaker_encoder = load_resnet_encoder("speaker_embedding/resnet34sel_pretrained.pt", "cuda")

avg_speaker_embs_seen = {}
audio_seen = {}

avg_speaker_embs_unseen = {}
audio_unseen = {}

folder_list_seen = "dafx-evaluation/seen-targets-vctk"   
folder_list_unseen = "dafx-evaluation/unseen-targets-vctk" 

if folder_list_seen is not None:
    for f, folder in enumerate(folder_list_seen.split(",")):
        if len(os.listdir(folder)) > 0:
            for subfolder in os.listdir(folder):
                if "." not in subfolder:
                    wavs = []
                    utt_embeddings = []
                    speaker_id = subfolder
                    for ext in extension:
                        wavs.extend(list(Path(folder, speaker_id).rglob(ext)))

                    loader = tqdm(wavs)
                    for wav in loader:
                        loader.set_description("{}".format(path.basename(wav)))
                        output = preprocess(wav)
                        if output is not None:
                            for o in output:
                                utt_emb = speaker_encoder(torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(torch.device(device)))
                                utt_emb = utt_emb.detach().cpu().squeeze().numpy()
                                utt_embeddings.append(utt_emb)
                            avg_speaker_embs_seen[speaker_id] = np.mean(utt_embeddings, axis=0)
                            audio_seen[speaker_id] = o
                                 

if folder_list_unseen is not None:
    for f, folder in enumerate(folder_list_unseen.split(",")):
        if len(os.listdir(folder)) > 0:
            for subfolder in os.listdir(folder):
                if "." not in subfolder:
                    wavs = []
                    utt_embeddings = []
                    speaker_id = subfolder
                    for ext in extension:
                        wavs.extend(list(Path(folder, speaker_id).rglob(ext)))

                    loader = tqdm(wavs)
                    for wav in loader:
                        loader.set_description("{}".format(path.basename(wav)))
                        output = preprocess(wav)
                        if output is not None:
                            for o in output:
                                utt_emb = speaker_encoder(torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(torch.device(device)))
                                utt_emb = utt_emb.detach().cpu().squeeze().numpy()
                                utt_embeddings.append(utt_emb)
                            avg_speaker_embs_unseen[speaker_id] = np.mean(utt_embeddings, axis=0)
                            audio_unseen[speaker_id] = o

print("SEEN SPEAKERS:", avg_speaker_embs_seen.keys())
print("UNSEEN SPEAKERS:", avg_speaker_embs_unseen.keys())

# ---------------------------------------------------
# LOAD UTIL MODELS
# ---------------------------------------------------
model = wenet.load_model('english')
resemblyzer = VoiceEncoder()

step = 0
seen_wer_full = 0
seen_cer_full = 0
seen_ssim = 0
seen_ovrl_mos = 0
seen_sig_mos = 0
seen_bak_mos = 0

seen_wer_full_avc = 0
seen_cer_full_avc = 0
seen_ssim_avc = 0
seen_ovrl_mos_avc = 0
seen_sig_mos_avc = 0
seen_bak_mos_avc = 0

# ---------------------------------------------------        
# START CONVERSION
# ---------------------------------------------------

print("TRAINED SAMPLE RATE:", rave.sr, "DEFINED SAMPLE RATE:", SR)
print("CALCULATING METRICS FOR SEEN CONVERSIONS")

audios = tqdm(list(Path(args.WAV_FOLDER).rglob("*.flac")))

for audio in audios:
    audio_name = path.splitext(path.basename(audio))[0]
    audios.set_description(audio_name)
    target_text = get_txt_file(audio_name)

    # LOAD AUDIO TO SPLTTABLE TENSOR 
    x, sr = li.load(audio, sr=SR)
    
    val = int(np.ceil((len(x) / 65536)))
    N_pad = 65536 * val
    pad = (N_pad - (len(x) % N_pad)) % N_pad
    x = np.pad(x, (0, pad))
    x = x.reshape(-1, N_pad)
    x = torch.from_numpy(x).float().to(device)
    x = x.reshape(int(x.shape[-1] / 65536), -1)
    
    #print("TARGET: ", target_text)
    
    for speaker, embedding in avg_speaker_embs_seen.items():
        sp_embedding = torch.tensor(embedding).unsqueeze(0).to(device)
        
        file_path_conv = path.join(args.OUT, f"seen_{audio_name}_{speaker}_conv.wav")
        file_path_target = path.join(args.OUT, f"seen_{audio_name}_{speaker}_target.wav")
        target_audio = audio_seen[speaker]
        
        #convert
        z, sp = rave.encode(x, sp_embedding)
        y = rave.decode(z, sp)
        y = y.reshape(-1, 1).cpu().numpy()
        
        sf.write(file_path_conv, y, SR)             
        sf.write(file_path_target, target_audio, SR)
        
        #OTHER MODELS
        avc_path_target = pick_random_speaker(folder_list_seen, speaker)
        diff_conv = get_converison_from_diffvc(audio, avc_path_target)
        avc_path_conv = path.join("dafx-evaluation/diffvc", f"seen_{audio_name}_{speaker}_diff.wav")    
        sf.write(avc_path_conv, diff_conv.reshape(-1, 1).cpu().numpy(), 22050)  
        
        #OBJECTIVE METRICS (AUTOMATICALLY LOADED TO 16kHz)
        #1. Convert
        conv = preprocess_wav(file_path_conv)
        conv_avc = preprocess_wav(avc_path_conv)
        
        #2. Retrieve the target
        target = preprocess_wav(file_path_target)
        target_avc = preprocess_wav(avc_path_target)
        
        #2.Get transcriptions and calculate
        result = model.transcribe(file_path_conv)
        transcribed_text = str(result['text']).replace("▁", " ")
        
        result_avc = model.transcribe(avc_path_conv)
        transcribed_text_avc = str(result_avc['text']).replace("▁", " ")
        
        w_error = wer(target_text, transcribed_text)
        seen_wer_full += w_error
        w_error = wer(target_text, transcribed_text_avc)
        seen_wer_full_avc += w_error

        c_error = cer(target_text, transcribed_text)
        seen_cer_full += c_error
        c_error = cer(target_text, transcribed_text_avc)
        seen_cer_full_avc += c_error
                
        #3. RESEMBLYZER 
        conv_emb = resemblyzer.embed_utterance(conv)
        conv_emb_avc = resemblyzer.embed_utterance(conv_avc)
        
        target_emb = resemblyzer.embed_utterance(target)
        target_emb_avc = resemblyzer.embed_utterance(target_avc)

        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        cos_dis = cos(torch.tensor(conv_emb), torch.tensor(target_emb))
        seen_ssim += cos_dis.numpy()
        
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        cos_dis = cos(torch.tensor(conv_emb_avc), torch.tensor(target_emb_avc))
        seen_ssim_avc += cos_dis.numpy()
                
        #4. DNS-MOS
        #conv = conv / np.max(abs(conv))
        conv, _ = li.load(file_path_conv, sr=16000)
        dnsmos_vals = dnsmos.run(conv, sr=16000)
        seen_ovrl_mos += dnsmos_vals['ovrl_mos']
        seen_sig_mos += dnsmos_vals['sig_mos']
        seen_bak_mos += dnsmos_vals['bak_mos']
        
        #conv_avc = conv_avc / np.max(abs(conv_avc))
        conv_avc, _ = li.load(avc_path_conv, sr=16000)
        dnsmos_vals_avc = dnsmos.run(conv_avc, sr=16000)
        seen_ovrl_mos_avc += dnsmos_vals_avc['ovrl_mos']
        seen_sig_mos_avc += dnsmos_vals_avc['sig_mos']
        seen_bak_mos_avc += dnsmos_vals_avc['bak_mos']
            
        step += 1
        
seen_wer = (seen_wer_full / step) * 100
seen_cer = (seen_cer_full / step) * 100
seen_ssim = (seen_ssim / step) * 100
seen_ovrl_mos = (seen_ovrl_mos / step)
seen_sig_mos = (seen_sig_mos / step)
seen_bak_mos = (seen_bak_mos / step)

seen_wer_avc = (seen_wer_full_avc / step) * 100
seen_cer_avc = (seen_cer_full_avc / step) * 100
seen_ssim_avc = (seen_ssim_avc / step) * 100
seen_ovrl_mos_avc = (seen_ovrl_mos_avc / step)
seen_sig_mos_avc = (seen_sig_mos_avc / step)
seen_bak_mos_avc = (seen_bak_mos_avc / step)

print("\nCALCULATING METRICS FOR UNSEEN CONVERSIONS")

step = 0
unseen_wer_full = 0
unseen_cer_full = 0
unseen_ssim = 0
unseen_ovrl_mos = 0
unseen_sig_mos = 0
unseen_bak_mos = 0

unseen_wer_full_avc = 0
unseen_cer_full_avc = 0
unseen_ssim_avc = 0
unseen_ovrl_mos_avc = 0
unseen_sig_mos_avc = 0
unseen_bak_mos_avc = 0

audios = tqdm(list(Path(args.WAV_FOLDER).rglob("*.flac")))

for audio in audios:
    audio_name = path.splitext(path.basename(audio))[0]
    audios.set_description(audio_name)
    target_text = get_txt_file(audio_name)

    # LOAD AUDIO TO SPLTTABLE TENSOR 
    x, sr = li.load(audio, sr=SR)
    
    val = int(np.ceil((len(x) / 65536)))
    N_pad = 65536 * val
    pad = (N_pad - (len(x) % N_pad)) % N_pad
    x = np.pad(x, (0, pad))
    x = x.reshape(-1, N_pad)
    x = torch.from_numpy(x).float().to(device)
    x = x.reshape(int(x.shape[-1] / 65536), -1)
    
    for speaker, embedding in avg_speaker_embs_unseen.items():
        sp_embedding = torch.tensor(embedding).unsqueeze(0).to(device)
        
        file_path_conv = path.join(args.OUT, f"unseen_{audio_name}_{speaker}_conv.wav")
        file_path_target = path.join(args.OUT, f"unseen_{audio_name}_{speaker}_target.wav")
        target_audio = audio_unseen[speaker]
        
        #convert
        z, sp = rave.encode(x, sp_embedding)
        y = rave.decode(z, sp)
        y = y.reshape(-1, 1).cpu().numpy()
        
        sf.write(file_path_conv, y, SR)             
        sf.write(file_path_target, target_audio, SR)
        
        #OTHER MODELS
        avc_path_target = pick_random_speaker(folder_list_unseen, speaker)
        diff_conv = get_converison_from_diffvc(audio, avc_path_target)
        avc_path_conv = path.join("dafx-evaluation/diffvc", f"unseen_{audio_name}_{speaker}_diff.wav")    
        sf.write(avc_path_conv, diff_conv.reshape(-1, 1).cpu().numpy(), 22050)  
        
        #OBJECTIVE METRICS (AUTOMATICALLY LOADED TO 16kHz)
        #1. Convert
        conv = preprocess_wav(file_path_conv)
        conv_avc = preprocess_wav(avc_path_conv)
        
        #2. Retrieve the target
        target = preprocess_wav(file_path_target)
        target_avc = preprocess_wav(avc_path_target)
        
        #2.Get transcriptions and calculate
        result = model.transcribe(file_path_conv)
        transcribed_text = str(result['text']).replace("▁", " ")
        
        result_avc = model.transcribe(avc_path_conv)
        transcribed_text_avc = str(result_avc['text']).replace("▁", " ")
        
        w_error = wer(target_text, transcribed_text)
        unseen_wer_full += w_error
        w_error = wer(target_text, transcribed_text_avc)
        unseen_wer_full_avc += w_error

        c_error = cer(target_text, transcribed_text)
        unseen_cer_full += c_error
        c_error = cer(target_text, transcribed_text_avc)
        unseen_cer_full_avc += c_error
                
        #3. RESEMBLYZER 
        conv_emb = resemblyzer.embed_utterance(conv)
        conv_emb_avc = resemblyzer.embed_utterance(conv_avc)
        
        target_emb = resemblyzer.embed_utterance(target)
        target_emb_avc = resemblyzer.embed_utterance(target_avc)

        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        cos_dis = cos(torch.tensor(conv_emb), torch.tensor(target_emb))
        unseen_ssim += cos_dis.numpy()
        
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        cos_dis = cos(torch.tensor(conv_emb_avc), torch.tensor(target_emb_avc))
        unseen_ssim_avc += cos_dis.numpy()
                
        #4. DNS-MOS
        #conv = conv / np.max(abs(conv))
        conv, _ = li.load(file_path_conv, sr=16000)
        dnsmos_vals = dnsmos.run(conv, sr=16000)
        unseen_ovrl_mos += dnsmos_vals['ovrl_mos']
        unseen_sig_mos += dnsmos_vals['sig_mos']
        unseen_bak_mos += dnsmos_vals['bak_mos']
        
        #conv_avc = conv_avc / np.max(abs(conv_avc))
        conv_avc, _ = li.load(avc_path_conv, sr=16000)
        dnsmos_vals_avc = dnsmos.run(conv_avc, sr=16000)
        unseen_ovrl_mos_avc += dnsmos_vals_avc['ovrl_mos']
        unseen_sig_mos_avc += dnsmos_vals_avc['sig_mos']
        unseen_bak_mos_avc += dnsmos_vals_avc['bak_mos']
            
        step += 1
        
unseen_wer = (unseen_wer_full / step) * 100
unseen_cer = (unseen_cer_full / step) * 100
unseen_ssim = (unseen_ssim / step) * 100
unseen_ovrl_mos = (unseen_ovrl_mos / step)
unseen_sig_mos = (unseen_sig_mos / step)
unseen_bak_mos = (unseen_bak_mos / step)

unseen_wer_avc = (unseen_wer_full_avc / step) * 100
unseen_cer_avc = (unseen_cer_full_avc / step) * 100
unseen_ssim_avc = (unseen_ssim_avc / step) * 100
unseen_ovrl_mos_avc = (unseen_ovrl_mos_avc / step)
unseen_sig_mos_avc = (unseen_sig_mos_avc / step)
unseen_bak_mos_avc = (unseen_bak_mos_avc / step)

print("\nSEEN SPEAKERS - WER in %", " RAVE:", seen_wer, " DIFF-VC:", seen_wer_avc)
print("SEEN SPEAKERS - CER in %", " RAVE:", seen_cer, " DIFF-VC:", seen_cer_avc)
print("SEEN SPEAKERS - SSIM in %", " RAVE:", seen_ssim, " DIFF-VC:", seen_ssim_avc)
print("SEEN SPEAKERS - OVRL MOS", " RAVE:", seen_ovrl_mos, " DIFF-VC:", seen_ovrl_mos_avc)
print("SEEN SPEAKERS - SIG MOS:", " RAVE:", seen_sig_mos, " DIFF-VC:", seen_sig_mos_avc)
print("SEEN SPEAKERS - BAK MOS:", " RAVE:", seen_bak_mos, " DIFF-VC:", seen_bak_mos_avc)

print("\nUNSEEN SPEAKERS - WER in %", " RAVE:", unseen_wer, " DIFF-VC:", unseen_wer_avc)
print("UNSEEN SPEAKERS - CER in %", " RAVE:", unseen_cer, " DIFF-VC:", unseen_cer_avc)
print("UNSEEN SPEAKERS - SSIM in %", " RAVE:", unseen_ssim, " DIFF-VC:", unseen_ssim_avc)
print("UNSEEN SPEAKERS - OVRL MOS", " RAVE:", unseen_ovrl_mos, " DIFF-VC:", unseen_ovrl_mos_avc)
print("UNSEEN SPEAKERS - SIG MOS:", " RAVE:", unseen_sig_mos, " DIFF-VC:", unseen_sig_mos_avc)
print("UNSEEN SPEAKERS - BAK MOS:", " RAVE:", unseen_bak_mos, " DIFF-VC:", unseen_bak_mos_avc)
print("------------------------------------")

print("\nFULL - WER in %:", " RAVE:", (unseen_wer + seen_wer) / 2, " DIFF-VC:",  (unseen_wer_avc + seen_wer_avc) / 2)
print("FULL - CER in %:", " RAVE:", (unseen_cer + seen_cer) / 2, " DIFF-VC:", (unseen_cer_avc + seen_cer_avc) / 2)
print("FULL - SSIM in %:", " RAVE:", (unseen_ssim + seen_ssim) / 2, " DIFF-VC:", (unseen_ssim_avc + seen_ssim_avc) / 2)
print("FULL - OVRL MOS:", (unseen_ovrl_mos + seen_ovrl_mos) / 2,  " DIFF-VC:", (unseen_ovrl_mos_avc + seen_ovrl_mos_avc) / 2)
print("FULL - SIG MOS:", (unseen_sig_mos + seen_sig_mos) / 2, " DIFF-VC:", (unseen_sig_mos_avc + seen_sig_mos_avc) / 2)
print("FULL - BAK MOS:", (unseen_bak_mos + seen_bak_mos) / 2, " DIFF-VC:", (unseen_bak_mos_avc + seen_bak_mos_avc) / 2)

