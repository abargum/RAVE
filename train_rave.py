import torch
from torch.utils.data import DataLoader, random_split

from rave.model import RAVE
from rave.core import random_phase_mangle, EMAModelCheckPoint
from rave.core import search_for_run

#from udls import SimpleDataset, simple_audio_preprocess
import sys
import wandb

sys.path.insert(1, '../udls_extended/')

from udls_extended import SimpleDataset_VCTK as SimpleDataset
from udls_extended.transforms import Compose, RandomApply, Dequantize, RandomCrop, Perturb
from udls_extended import simple_audio_preprocess
from effortless_config import Config, setting
import pytorch_lightning as pl
from os import environ, path
import os
import numpy as np

import GPUtil as gpu

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    class args(Config):
        groups = ["small", "large"]

        DATA_SIZE = 16
        CAPACITY = setting(default=64, small=32, large=64)
        LATENT_SIZE = 64
        BIAS = True
        NO_LATENCY = False
        RATIOS = setting(
            default=[4, 4, 4, 2],
            small=[4, 4, 4, 2],
            large=[4, 4, 2, 2, 2],
        )
        RATIOS_DOWN = [1, 2, 4, 4]

        MIN_KL = 1e-1
        MAX_KL = 1e-1
        CROPPED_LATENT_SIZE = 0
        FEATURE_MATCH = True

        LOUD_STRIDE = 1

        USE_NOISE = True
        NOISE_RATIOS = [4, 4, 4]
        NOISE_BANDS = 5

        D_CAPACITY = 16
        D_MULTIPLIER = 4
        D_N_LAYERS = 4

        WARMUP = setting(default=1000000, small=1000000, large=3000000)
        MODE = "hinge"
        CKPT = None

        PREPROCESSED = None
        WAV = None
        SR = 48000
        N_SIGNAL = 65536
        MAX_STEPS = setting(default=6000000, small=3000000, large=6000000)
        VAL_EVERY = 10000
        BLOCK_SIZE = 128

        BATCH = 8

        SPEAKER_ENCODER = 'RESNET'
        #CONTRASTIVE_LOSS = True
        #CONTENT_LOSS = True

        NAME = None
        
        PLOT = False

    args.parse_args()

    assert args.NAME is not None

    # -------------------------------
    # Initialize W and B
    # -------------------------------
    wandb.init(project="RAVE", name=f"{args.NAME}")
    
    # -------------------------------

    model = RAVE(
        data_size=args.DATA_SIZE,
        capacity=args.CAPACITY,
        latent_size=args.LATENT_SIZE,
        ratios=args.RATIOS,
        ratios_down=args.RATIOS_DOWN,
        bias=args.BIAS,
        loud_stride=args.LOUD_STRIDE,
        use_noise=args.USE_NOISE,
        noise_ratios=args.NOISE_RATIOS,
        noise_bands=args.NOISE_BANDS,
        d_capacity=args.D_CAPACITY,
        d_multiplier=args.D_MULTIPLIER,
        d_n_layers=args.D_N_LAYERS,
        warmup=args.WARMUP,
        mode=args.MODE,
        block_size=args.BLOCK_SIZE,
        speaker_encoder=args.SPEAKER_ENCODER,
        #contrastive_loss = args.CONTRASTIVE_LOSS,
        #content_loss = args.CONTENT_LOSS,
        no_latency=args.NO_LATENCY,
        sr=args.SR,
        min_kl=args.MIN_KL,
        max_kl=args.MAX_KL,
        cropped_latent_size=args.CROPPED_LATENT_SIZE,
        feature_match=args.FEATURE_MATCH,
    )

    if args.SPEAKER_ENCODER == "RESNET":
        speaker_size = 512
    else:
        speaker_size = 192

    x = {'data_clean': torch.zeros(args.BATCH, 2**16),
        'data_perturbed_1': torch.zeros(args.BATCH, 2**16),
        'data_perturbed_2': torch.zeros(args.BATCH, 2**16),
        'speaker_emb': torch.zeros(args.BATCH, speaker_size),
        'speaker_id_avg': torch.zeros(args.BATCH, speaker_size)}
    
    model.validation_step(x, 0)

    preprocess = lambda name: simple_audio_preprocess(
        args.SR,
        2 * args.N_SIGNAL,
    )(name).astype(np.float16)

    dataset = SimpleDataset(
        args.SR,
        args.SPEAKER_ENCODER,
        torch.device('cuda'),
        args.PREPROCESSED,
        args.WAV,
        preprocess_function=preprocess,
        split_set="full",
        transforms=Perturb([
            lambda x, x_p_1, x_p_2: (x.astype(np.float32), x_p_1.astype(np.float32), x_p_2.astype(np.float32)),
            RandomCrop(args.N_SIGNAL),
            RandomApply(
                lambda x: random_phase_mangle(x, 20, 2000, .99, args.SR),
                p=.8,
            ),
            Dequantize(16),
            lambda x, x_p_1, x_p_2: (x.astype(np.float32), x_p_1.astype(np.float32), x_p_2.astype(np.float32)),
        ],
        args.SR),
    )
    
    # -------------- PLOT SPEAKER EMBEDDINGS ----------------
    if args.PLOT:
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

        plt.savefig('scatter_plot_embeddings.png', bbox_inches='tight')
            
        # ---------------------------------------------------

    val = max((2 * len(dataset)) // 500, 1)
    train = len(dataset) - val
    train, val = random_split(
        dataset,
        [train, val],
        generator=torch.Generator().manual_seed(42),
    )

    num_workers = 0 if os.name == "nt" else 10
    train = DataLoader(train,
                       args.BATCH,
                       True,
                       drop_last=True,
                       num_workers=num_workers)
    val = DataLoader(val, args.BATCH, False, num_workers=num_workers)

    example = next(iter(train))
    clean_example = example['data_clean']
    perturbed_example = example['data_perturbed_1']
    speaker_emb = example['speaker_emb']
    speaker_id = example['speaker_id']
    print("Input Audio Shape:", clean_example.shape,
          "\nPerturbed Audio Shape:", perturbed_example.shape,
          "\nSpeaker Embedding Shape:", speaker_emb.shape,
          "\nSpeaker ID Example:", speaker_id)
    
    # CHECKPOINT CALLBACKS
    validation_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="validation",
        filename="best",
    )
    last_checkpoint = pl.callbacks.ModelCheckpoint(filename="last")

    CUDA = gpu.getAvailable(maxMemory=.05)
    VISIBLE_DEVICES = environ.get("CUDA_VISIBLE_DEVICES", "")

    if VISIBLE_DEVICES:
        use_gpu = int(int(VISIBLE_DEVICES) >= 0)
    elif len(CUDA):
        environ["CUDA_VISIBLE_DEVICES"] = str(CUDA[0])
        use_gpu = 1
    elif torch.cuda.is_available():
        print("Cuda is available but no fully free GPU found.")
        print("Training may be slower due to concurrent processes.")
        use_gpu = 1
    else:
        print("No GPU found.")
        use_gpu = 0

    val_check = {}
    if len(train) >= args.VAL_EVERY:
        val_check["val_check_interval"] = args.VAL_EVERY
    else:
        nepoch = args.VAL_EVERY // len(train)
        val_check["check_val_every_n_epoch"] = nepoch

    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(path.join("runs", args.NAME),
                                            name="rave"),
        gpus=use_gpu,
        callbacks=[validation_checkpoint, last_checkpoint],
        max_epochs=100000,
        max_steps=args.MAX_STEPS,
        **val_check,
    )

    run = search_for_run(args.CKPT, mode="last")
    if run is None: run = search_for_run(args.CKPT, mode="best")
    if run is not None:
        step = torch.load(run, map_location='cpu')["global_step"]
        trainer.fit_loop.epoch_loop._batches_that_stepped = step

    trainer.fit(model, train, val, ckpt_path=run)
