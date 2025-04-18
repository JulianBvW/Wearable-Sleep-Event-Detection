from torch.utils.data import Dataset
import pandas as pd
import torch
import os

from wearsed.dataset.Recording import Recording
from wearsed.dataset.utils import RESP_EVENT_TYPES

class WearSEDDataset(Dataset):
    def __init__(self, mesaid_path='wearsed/dataset/data_ids/', scoring_from='somnolyzer', signals_to_read=['HR', 'SpO2', 'Flow', 'Pleth'], return_recording=False, use_predicted_hypnogram=False, pleth_statistical=False, pleth_vae_latents=False):
        if not os.path.isfile(mesaid_path + 'mesa_root.txt') or not os.path.isfile(mesaid_path + f'mesa_ids_{scoring_from}.csv'):
            raise Exception(f'MESA IDs from {scoring_from} not loaded. Run `wearsed/dataset/data_ids/load_mesa.py <MESA ROOT PATH> {scoring_from}`.')
    
        with open(mesaid_path + 'mesa_root.txt', 'r') as f:
            self.mesa_root = f.readline()
        
        self.mesa_ids = pd.read_csv(mesaid_path + f'mesa_ids_{scoring_from}.csv')['id']
        self.subject_infos = pd.read_csv(self.mesa_root + 'datasets/mesa-sleep-harmonized-dataset-0.7.0.csv')
        self.subject_infos.set_index('mesaid', inplace=True)

        self.scoring_from     = scoring_from
        self.signals_to_read  = signals_to_read
        self.return_recording = return_recording

        self.use_predicted_hypnogram = use_predicted_hypnogram
        self.pleth_statistical = pleth_statistical
        self.pleth_vae_latents = pleth_vae_latents

    def __len__(self):
        return len(self.mesa_ids)

    def __getitem__(self, idx):
        mesa_id = self.mesa_ids[idx]
        return self.from_id(mesa_id)
    
    def from_id(self, mesa_id):
        subject_info = self.subject_infos.loc[mesa_id]
        subject_info = self.subject_infos.loc[mesa_id]
        recording = Recording(mesa_id, subject_info, signals_to_read=self.signals_to_read, scoring_from=self.scoring_from, events_as_list=self.return_recording, use_predicted_hypnogram=self.use_predicted_hypnogram)

        if self.return_recording:
            return recording
        
        ### Preprocesing for Baseline model
        # 3 inputs: Hypnogram (1Hz), SpO2 (1Hz), Pleth (1Hz, 13-dim)
        # 1 output: Event vs No Event (1Hz)

        # Inputs
        inputs = [recording.hypnogram, recording.psg['SpO2']]
        if self.pleth_statistical:
            pleth_statistical = pd.read_csv(f'/vol/sleepstudy/datasets/mesa/features/pleth/{mesa_id:04}.csv')
            for key in pleth_statistical.keys():
                inputs.append(pleth_statistical[key])
        if self.pleth_vae_latents:
            pleth_vae_latents = pd.read_csv(f'/vol/sleepstudy/datasets/mesa/features/pleth_vae_latents/{mesa_id:04}.csv')
            for key in pleth_vae_latents.keys():
                inputs.append(pleth_vae_latents[key])
        shortest_len = min([len(signal) for signal in inputs])
        inputs = [torch.tensor(signal, dtype=torch.float)[:shortest_len] for signal in inputs]
        inputs = torch.stack(inputs)

        # Output
        event_or_not = torch.Tensor(recording.event_df[RESP_EVENT_TYPES].any(axis=1).astype(int))[:shortest_len]

        return inputs, event_or_not  # [15, len], [len]

def process_higher_freqs(signal, freq):
    list_mean = []
    list_std  = []
    list_min  = []
    list_max  = []

    for sec_idx in range(len(signal)//freq):
        second = signal[sec_idx*freq:sec_idx*freq+freq]
        list_mean.append(second.mean())
        list_std.append(second.std())
        list_min.append(second.min())
        list_max.append(second.max())
    
    return torch.Tensor(list_mean), torch.Tensor(list_std), torch.Tensor(list_min), torch.Tensor(list_max)
    
def get_shakiness(signal, freq):
    shake = []

    for sec in range(len(signal)//freq):
        cur_shake = 0
        last_direc = False
        last_point = signal[sec*freq]
        for i in range(1, freq):
            cur_point = signal[sec*freq+i]
            cur_direc = cur_point > last_point  # True = Signal goes Up, False = Signal goes Down
            if last_direc != cur_direc:
                cur_shake += 1
                last_direc = cur_direc
            last_point = cur_point
        shake.append(cur_shake)
    
    return torch.Tensor(shake)
