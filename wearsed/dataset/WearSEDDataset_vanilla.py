from torch.utils.data import Dataset
import pandas as pd
import torch
import os

from wearsed.dataset.Recording import Recording

class WearSEDDataset(Dataset):
    def __init__(self, mesaid_path='wearsed/dataset/data_ids/', signals_to_read=['HR', 'SpO2', 'Flow', 'Pleth'], preprocess=False):
        if not os.path.isfile(mesaid_path+'mesa_root.txt') or not os.path.isfile(mesaid_path+'mesa_ids.csv'):
            raise Exception('MESA IDs not loaded. Run `wearsed/dataset/data_ids/load_mesa.py <MESA ROOT PATH>`.')
    
        with open(mesaid_path+'mesa_root.txt', 'r') as f:
            self.mesa_root = f.readline()
        
        self.mesa_ids = pd.read_csv(mesaid_path+'mesa_ids.csv', header=None)[0]
        self.subject_infos = pd.read_csv(self.mesa_root + 'datasets/mesa-sleep-harmonized-dataset-0.7.0.csv')
        self.subject_infos.set_index('mesaid', inplace=True)

        self.signals_to_read = signals_to_read
        self.preprocess = preprocess

    def __len__(self):
        return len(self.mesa_ids)

    def __getitem__(self, idx):
        mesa_id = self.mesa_ids[idx]
        recording = Recording(mesa_id, self.subject_infos.loc[mesa_id], signals_to_read=self.signals_to_read)

        if not self.preprocess:
            return recording
        
        ### Preprocesing for Baseline model
        # 3 inputs: Hypnogram (1Hz), SpO2 (1Hz), Pleth (256Hz)
        # 1 output: Event vs No Event (1Hz)

        # Inputs
        hypnogram = torch.Tensor(recording.hypnogram)
        spo2 = torch.Tensor(recording.psg['SpO2'])
        pleth = recording.psg['Pleth']
        pleth_mean, pleth_std, pleth_min, pleth_max = process_higher_freqs(pleth, 256)  # Convert to 4 inputs at 1Hz
        smallest = min([len(s) for s in [hypnogram, spo2, pleth_mean, pleth_std, pleth_min, pleth_max]])
        signals = torch.stack([hypnogram[:smallest], spo2[:smallest], pleth_mean[:smallest], pleth_std[:smallest], pleth_min[:smallest], pleth_max[:smallest]])

        # Output
        events_to_look_at = ['Obstructive apnea', 'Hypopnea']
        event_or_not = torch.zeros(len(hypnogram))
        for event in recording.get_events(events_to_look_at):
            event_or_not[int(event.start):int(event.end)] = 1

        return signals, event_or_not#, torch.tensor(pleth)

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
