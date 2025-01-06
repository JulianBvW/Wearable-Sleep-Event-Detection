'''
Use the VAE model to turn the 1-dim 256Hz Pleth signal into a 8-dim 1Hz signal 
'''

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import os

from wearsed.dataset.WearSEDDataset import WearSEDDataset
from wearsed.models.vae.VAE_conv_5s import VAE as VAE_Conv_5s

FREQ = 256
OUT_DIR = '/vol/sleepstudy/datasets/mesa/features/pleth_vae_latents'
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VAE_Conv_5s().to(device)
model.load_state_dict(torch.load('wearsed/training/vae/output_conv_5s/model_final.pth', weights_only=True, map_location=device))
model.eval()

def make_signal_padding(signal):
    values = signal.values
    pad_start, pad_end = values[:2*FREQ], values[len(signal)-2*FREQ:]
    return torch.tensor(np.concatenate([pad_start, values, pad_end]), dtype=torch.float).to(device)

def get_vae_latents(signal):
    signal = make_signal_padding(signal)

    signal_latents = []
    start, end = 2*FREQ, len(signal) - 2*FREQ
    for sec_idx in range(start, end, FREQ):
        second = signal[sec_idx-2*FREQ:sec_idx+3*FREQ].view(1, -1)
        second_latent, _ = model.encode(second)                 # [1, 8]
        signal_latents.append(second_latent[0].cpu().detach())  # [8]
    signal_latents = torch.stack(signal_latents, dim=1)         # [8, len]

    return pd.DataFrame({
        'latent_dim_0': signal_latents[0].numpy(),
        'latent_dim_1': signal_latents[1].numpy(),
        'latent_dim_2': signal_latents[2].numpy(),
        'latent_dim_3': signal_latents[3].numpy(),
        'latent_dim_4': signal_latents[4].numpy(),
        'latent_dim_5': signal_latents[5].numpy(),
        'latent_dim_6': signal_latents[6].numpy(),
        'latent_dim_7': signal_latents[7].numpy()
    })

dataset = WearSEDDataset(signals_to_read=['Pleth'], return_recording=True)
for r_id in tqdm(range(len(dataset))):
    recording = dataset[r_id]
    mesa_id = recording.id
    latents = get_vae_latents(recording.psg['Pleth'])
    latents.to_csv(f'{OUT_DIR}/{mesa_id:04}.csv', index=False)