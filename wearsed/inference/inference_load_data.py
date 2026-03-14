'''
Functions for loading and batching CFS dataset recordings for the Apnea Detection Model
'''

import pandas as pd
import numpy as np
import pyedflib
import torch

SEQ_LENGTH = 30*60
OVERLAP_WINDOW = 60


def load_hypnogram(input_hypnogram):
    predicted_hypnogram = pd.read_csv(input_hypnogram, index_col='ts')['PPG_4cl']
    predicted_hypnogram[predicted_hypnogram == 'W'] = 0
    predicted_hypnogram[predicted_hypnogram == 'N1/N2'] = 1
    predicted_hypnogram[predicted_hypnogram == 'N3'] = 3
    predicted_hypnogram[predicted_hypnogram == 'R'] = 5
    return pd.Series(np.repeat(predicted_hypnogram.values, 30))


def load_edf(input_edf):
    edf_reader = pyedflib.EdfReader(input_edf)
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


def load_datapoint(input_edf, input_hypnogram):
    hypnogram = torch.Tensor(load_hypnogram(input_hypnogram))

    psg = load_edf(input_edf)
    spo2 = torch.Tensor(psg['SpO2'])[:len(hypnogram)]
    ppg  = torch.Tensor(psg['PPG'])[:len(hypnogram)*128]
    return hypnogram, spo2, ppg


def create_batch(hypnogram, spo2, ppg):
    step = SEQ_LENGTH - 2 * OVERLAP_WINDOW
    batch_signals = []

    for start in range(0, len(hypnogram), step):
        end = start + SEQ_LENGTH
        if len(hypnogram[start:end]) < SEQ_LENGTH:
            break
        seq_hypnogram = hypnogram[start:end].view((1, -1))
        seq_spo2      = spo2[start:end].view((1, -1))
        seq_ppg       = ppg.repeat_interleave(2)[start*256:end*256].view((256, -1))  # Repeat PPG for 256Hz
        combined_signal = torch.cat([seq_hypnogram, seq_spo2, seq_ppg], dim=0)
        batch_signals.append(combined_signal)

    return torch.stack(batch_signals)