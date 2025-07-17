from wearsed.dataset.utils import RESP_EVENT_TYPES
from torch.utils.data import Dataset
import pandas as pd
import pyedflib
import torch
import os

class CFSDataset(Dataset):
    def __init__(self, cfs_id_path='wearsed/dataset/data_ids/'):
        if not os.path.isfile(cfs_id_path + 'cfs_root.txt') or not os.path.isfile(cfs_id_path + f'cfs_ids.csv'):
            raise Exception(f'CFS IDs not loaded. Run `wearsed/dataset/data_ids/load_cfs.py <CFS ROOT PATH>`.')
    
        with open(cfs_id_path + 'cfs_root.txt', 'r') as f:
            self.cfs_root = f.readline()
        
        self.cfs_ids = pd.read_csv(cfs_id_path + f'cfs_ids.csv', header=None)[0]

    def __len__(self):
        return len(self.cfs_ids)

    def get_cfs_id(self, idx):
        return int(self.cfs_ids[idx])

    def __getitem__(self, idx):
        cfs_id = self.get_cfs_id(idx)
        psg = self.load_psg(cfs_id)
        hypnogram = pd.read_csv(self.cfs_root + f'scorings/somnolyzer/hypnogram/hypnogram-{cfs_id}.csv', header=None)[0]

        # Inputs
        hypnogram = torch.Tensor(hypnogram)
        spo2 = torch.Tensor(psg['SpO2'])[:len(hypnogram)]
        ppg  = torch.Tensor(psg['PPG'])[:len(hypnogram)*128]

        # Output
        event_or_not = torch.Tensor(pd.read_csv(self.cfs_root + f'scorings/somnolyzer/events/events-{cfs_id}.csv')[RESP_EVENT_TYPES].any(axis=1).astype(int))

        return (hypnogram, spo2, ppg), event_or_not

    def load_psg(self, cfs_id):
        edf_reader = pyedflib.EdfReader(self.cfs_root + f'polysomnography/edfs/cfs-visit5-{cfs_id}.edf')

        signal_labels = edf_reader.getSignalLabels()

        spo2, ppg = None, None
        for i in range(edf_reader.signals_in_file):
            if signal_labels[i] == 'SpO2':
                spo2 = edf_reader.readSignal(i)  # SpO2  @ 1 Hz
            if signal_labels[i] == 'PlethWV':
                ppg  = edf_reader.readSignal(i)  # Pleth @ 128 Hz
                ppg  = ppg - ppg.mean()

        edf_reader.close()

        return {'SpO2': spo2, 'PPG': ppg}