from torch.utils.data import Dataset
import pandas as pd
import torch
import os

from wearsed.dataset.Recording import Recording
from wearsed.dataset.utils import RESP_EVENT_TYPES

class WearSEDDataset(Dataset):
    def __init__(self, mesaid_path='wearsed/dataset/data_ids/', scoring_from='somnolyzer', signals_to_read=['HR', 'SpO2', 'Flow', 'Pleth'], return_recording=False, use_predicted_hypnogram=False, denoised_ppg='none'):
        if not os.path.isfile(mesaid_path + 'mesa_root.txt') or not os.path.isfile(mesaid_path + f'mesa_ids_{scoring_from}.csv'):
            raise Exception(f'MESA IDs from {scoring_from} not loaded. Run `wearsed/dataset/data_ids/load_mesa.py <MESA ROOT PATH> {scoring_from}`.')
    
        with open(mesaid_path + 'mesa_root.txt', 'r') as f:
            self.mesa_root = f.readline()
        
        self.mesa_ids = pd.read_csv(mesaid_path + f'mesa_ids_{scoring_from}.csv')['id']
        self.subject_infos = pd.read_csv(self.mesa_root + 'datasets/mesa-sleep-harmonized-dataset-0.7.0.csv')
        self.subject_infos.set_index('mesaid', inplace=True)

        self.scoring_from            = scoring_from
        self.signals_to_read         = signals_to_read
        self.return_recording        = return_recording
        self.use_predicted_hypnogram = use_predicted_hypnogram
        self.denoised_ppg            = denoised_ppg

    def __len__(self):
        return len(self.mesa_ids)

    def __getitem__(self, idx):
        mesa_id = self.mesa_ids[idx]
        return self.from_id(mesa_id)
    
    def from_id(self, mesa_id):
        subject_info = self.subject_infos.loc[mesa_id]
        recording = Recording(mesa_id, subject_info, signals_to_read=self.signals_to_read, scoring_from=self.scoring_from, events_as_list=self.return_recording, use_predicted_hypnogram=self.use_predicted_hypnogram, denoised_ppg=self.denoised_ppg)

        if self.return_recording:
            return recording
        
        ### Preprocesing for Baseline model
        # 3 inputs: Hypnogram (1Hz), SpO2 (1Hz), Pleth (256Hz)
        # 1 output: Event vs No Event (1Hz)

        # Inputs
        hypnogram = torch.Tensor(recording.hypnogram)
        spo2 = torch.Tensor(recording.psg['SpO2'])
        pleth = torch.Tensor(recording.psg['Pleth'])

        # Output
        evs = recording.event_df
        target = torch.Tensor(evs['OSA'] + 2*evs['MSA'] + 3*evs['CSA'] + 4*evs['HYP'])

        return (hypnogram, spo2, pleth), target
