'''
Feature extraction script to get mean, std, min, max, mean peak intervals from 5 second windows of the pleth recordings
'''

from scipy.signal import find_peaks
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

from wearsed.dataset.WearSEDDataset import WearSEDDataset

import warnings
warnings.filterwarnings('ignore')

FREQ = 256
OUT_DIR = '/vol/sleepstudy/datasets/mesa/features/pleth'
os.makedirs(OUT_DIR, exist_ok=True)

def mean_peak_interval(signal):
    peaks, _ = find_peaks(signal.values, width=0.1*FREQ, distance=0.3*FREQ)
    if len(peaks) < 2:
        return 0.0
    intervals_sec = np.diff(peaks) / FREQ
    return np.mean(intervals_sec)

def compute_rolling_peak(signal, window_around_in_sec):
    rolling_peak = []
    signal_smooth = signal.rolling(32).mean()
    for sec_idx in range(len(signal)//FREQ):
        start, end = (sec_idx-window_around_in_sec)*256, (sec_idx+1+window_around_in_sec)*256
        start, end = max(0, start), min(len(signal), end)
        signal_window = signal_smooth[start:end]
        rolling_peak.append(mean_peak_interval(signal_window))
    return np.array(rolling_peak)

def get_features(signal, window_in_sec=5):
    window_size = window_in_sec * FREQ
    rolling_signal = signal.rolling(window=window_size, min_periods=1, center=True)
    features = pd.DataFrame({
        'mean': rolling_signal.mean().reset_index(drop=True),
        'std':  rolling_signal.std().reset_index(drop=True),
        'min':  rolling_signal.min().reset_index(drop=True),
        'max':  rolling_signal.max().reset_index(drop=True)
    }).iloc[::FREQ].reset_index(drop=True)
    rolling_peak = compute_rolling_peak(signal, window_in_sec//2)
    features['peak'] = rolling_peak
    return features

dataset = WearSEDDataset(signals_to_read=['Pleth'], return_recording=True)
for r_id in tqdm(range(len(dataset))):
    recording = dataset[r_id]
    mesa_id = recording.id
    features = get_features(recording.psg['Pleth'])
    features.to_csv(f'{OUT_DIR}/{mesa_id:04}.csv', index=False)