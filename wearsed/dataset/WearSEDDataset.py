from torch.utils.data import Dataset
import pandas as pd
import os

from wearsed.dataset.Recording import Recording

class WearSEDDataset(Dataset):
    def __init__(self):
        if not os.path.isfile('wearsed/dataset/data_ids/mesa_root.txt') or not os.path.isfile('wearsed/dataset/data_ids/mesa_ids.csv'):
            raise Exception('MESA IDs not loaded. Run `wearsed/dataset/data_ids/load_mesa.py <MESA ROOT PATH>`.')
    
        with open('wearsed/dataset/data_ids/mesa_root.txt', 'r') as f:
            self.mesa_root = f.readline()
        
        self.mesa_ids = pd.read_csv('wearsed/dataset/data_ids/mesa_ids.csv', header=None)[0]
        self.subject_infos = pd.read_csv(self.mesa_root + 'datasets/mesa-sleep-harmonized-dataset-0.7.0.csv')
        self.subject_infos.set_index('mesaid', inplace=True)

    def __len__(self):
        return len(self.mesa_ids)

    def __getitem__(self, idx):
        mesa_id = self.mesa_ids[idx]
        recording = Recording(mesa_id, self.subject_infos.loc[mesa_id])

        return recording