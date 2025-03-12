'''
Preprocessing the PPG signal using frequency filters and wavelets
'''

from skimage.restoration import denoise_wavelet
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import pandas as pd
import numpy as np
import h5py
import os

from wearsed.dataset.WearSEDDataset import WearSEDDataset

from argparse import ArgumentParser
parser = ArgumentParser(description='Preprocessing script')
parser.add_argument('--start', help='start', default=0, type=int)
parser.add_argument('--end', help='end', default=2000, type=int)
args = parser.parse_args()

FREQ = 256
OUT_DIR = '/vol/sleepstudy/datasets/mesa/preprocessing/pleth'
OUT_DIR_LOWPASS  = OUT_DIR + '/lowpass/'
OUT_DIR_BANDPASS = OUT_DIR + '/bandpass/'
OUT_DIR_WV_DB4   = OUT_DIR + '/wavelet_db4/'
OUT_DIR_WV_DMEY  = OUT_DIR + '/wavelet_dmey/'
OUT_DIR_WV_HAAR  = OUT_DIR + '/wavelet_haar/'
OUT_DIRS = [OUT_DIR_LOWPASS, OUT_DIR_BANDPASS, OUT_DIR_WV_DB4, OUT_DIR_WV_DMEY, OUT_DIR_WV_HAAR]
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)

def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass_filtfilt(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    y = filtfilt(b, a, data)
    return y

def get_preprocessed_signal(signal):
    lowpass  = pd.Series(butter_lowpass_filtfilt(signal, 5, FREQ))
    bandpass = pd.Series(butter_bandpass_filtfilt(signal, 0.5, 5, FREQ))
    wv_db4   = pd.Series(denoise_wavelet(signal, wavelet='db4', mode='soft', wavelet_levels=4, sigma=0.5))
    wv_dmey  = pd.Series(denoise_wavelet(signal, wavelet='dmey', mode='soft', wavelet_levels=4, sigma=0.5))
    wv_haar  = pd.Series(denoise_wavelet(signal, wavelet='haar', mode='soft', wavelet_levels=4, sigma=0.5))
    return lowpass, bandpass, wv_db4, wv_dmey, wv_haar

dataset = WearSEDDataset(signals_to_read=['Pleth'], return_recording=True)
for r_id in tqdm(range(len(dataset))):
    if r_id < args.start or r_id > args.end:
        continue
    recording = dataset[r_id]
    mesa_id = recording.id
    preprocessed_signals = get_preprocessed_signal(recording.psg['Pleth'])
    for out_dir, prepocessed_signal in zip(OUT_DIRS, preprocessed_signals):
        with h5py.File(f'{out_dir}/{mesa_id:04}.hdf5', 'w') as f:
            dset = f.create_dataset('signal', data=prepocessed_signal.values.astype(np.float32))