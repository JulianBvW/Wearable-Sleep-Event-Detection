'''
Class to capture all relevant recordings, annotations, and subject info from the PSG
'''

import h5py
import pyedflib
import numpy as np
import pandas as pd
from lxml import etree
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

from wearsed.dataset.Event import Event
from wearsed.dataset.utils import from_clock, to_clock, to_length, EVENT_COLORS, EVENT_TYPES

class Recording():
    def __init__(self, subject_id, subject_info=None, signals_to_read=['HR', 'SpO2', 'Flow', 'Pleth'], scoring_from='somnolyzer', events_as_list=False, use_predicted_hypnogram=False, denoised_ppg='none'):
        self.id = subject_id

        path_dataset = '/vol/sleepstudy/datasets/mesa/'  # TODO make modular as argument
        path_psg     = path_dataset + f'polysomnography/edfs/mesa-sleep-{subject_id:04}.edf'
        path_scoring = path_dataset + f'scorings/{scoring_from}/'
        path_subject = path_dataset + 'datasets/mesa-sleep-harmonized-dataset-0.7.0.csv'

        self.load_psg(path_psg, signals_to_read=signals_to_read, denoised_ppg=denoised_ppg, path_dataset=path_dataset)
        self.load_scorings(path_scoring, subject_id, events_as_list)
        self.load_subject_data(path_subject, subject_id, subject_info)

        self.post_process(use_predicted_hypnogram, path_dataset)

    def get_event_count(self, event_type):
        assert self.events is not None, 'Events aren\'t loaded as lists ; Initialize with `Recording(..., events_as_list=True)`'
        return len(self.get_events(event_type))

    def get_events(self, event_type):
        assert self.events is not None, 'Events aren\'t loaded as lists ; Initialize with `Recording(..., events_as_list=True)`'
        event_types = event_type if type(event_type) is list else [event_type]
        return list(filter(lambda event: event.type in event_types, self.events))

    def get_ahi(self):  # Apnea Hypopnea Index
        assert self.events is not None, 'Events aren\'t loaded as lists ; Initialize with `Recording(..., events_as_list=True)`'
        return self.get_event_count(['HYP', 'OSA', 'CSA', 'MSA']) / (self.total_sleep_time_in_sec / 60 / 60)

    def get_ari(self):  # Arousal Index
        assert self.events is not None, 'Events aren\'t loaded as lists ; Initialize with `Recording(..., events_as_list=True)`'
        return self.get_event_count(['ARO']) / (self.total_sleep_time_in_sec / 60 / 60)

    def get_ahi_severity_class(self):
        assert self.events is not None, 'Events aren\'t loaded as lists ; Initialize with `Recording(..., events_as_list=True)`'
        ahi = self.get_ahi()

        if ahi < 5:   # Normal
            return 0
        if ahi < 15:  # Mild
            return 1
        if ahi < 30:  # Moderate
            return 2
        return 3      # Severe

    def look_at(self, time=None, window_size=None, events=EVENT_TYPES):
        assert len(self.events) > 0, 'Events aren\'t loaded as lists ; Initialize with `Recording(..., events_as_list=True)`'

        if time is None:
            start, end = 0, len(self.hypnogram)
        elif type(time) == int:
            start, end = time-window_size, time+window_size
        elif type(time) == str:
            time, window_size = from_clock(time), from_clock(window_size)
            start, end = time-window_size, time+window_size
        
        signal_count = 1 + len(self.psg.keys())
        _, axs = plt.subplots(signal_count, 1, figsize=(20, 2 * signal_count), sharex=True)
        colors = ['blue', 'orange', 'purple', 'green', 'red']

        # Plotting the Hypnogram
        axs[0].plot(self.hypnogram[start:end], color=colors[0])
        axs[0].set_ylabel('Sleep Stage')
        axs[0].legend(['Hypnogram'], loc='upper right')

        # Plotting the PSG signals
        for i, signal_name in enumerate(self.psg.keys()):
            signal = self.psg[signal_name]
            signal[signal == 0] = None
            freq = self.psg_freqs[signal_name]

            timeline = np.arange(0, len(signal)) / freq
            time_start, time_end = start * freq, end * freq

            axs[i+1].plot(timeline[time_start:time_end], signal[time_start:time_end], color=colors[(i+1) % len(colors)])
            final_legend = axs[i+1].legend([signal_name], loc='upper right')

        # Highlight events
        event_types = {}
        for event in self.events:
            if event.type in events and event.start >= start and event.end <= end:
                event_types[event.type] = 0
                for i in range(signal_count):
                    axs[i].axvspan(event.start, event.end, facecolor=EVENT_COLORS[event.type], alpha=0.33)

        patches = []
        for event_type in sorted(list(event_types.keys())):
            patches.append(mpatches.Patch(color=EVENT_COLORS[event_type], alpha=0.33, label=event_type))
        axs[signal_count-1].legend(handles=patches, loc='lower right', ncols=len(patches))
        axs[signal_count-1].add_artist(final_legend)

        axs[signal_count-1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: to_clock(int(x))))
        axs[signal_count-1].set_xlabel('Time')
        plt.xticks(range(start, end, int((end-start)/20)))
        plt.tight_layout()
        plt.show()

    def load_psg(self, path_psg, signals_to_read, denoised_ppg='none', path_dataset=''):
        edf_reader = pyedflib.EdfReader(path_psg)

        signal_labels = edf_reader.getSignalLabels()

        self.psg = {}
        self.psg_freqs = {}
        for i in range(edf_reader.signals_in_file):
            if signal_labels[i] in signals_to_read:
                self.psg[signal_labels[i]] = pd.Series(edf_reader.readSignal(i))
                self.psg_freqs[signal_labels[i]] = int(edf_reader.getSampleFrequency(i))

        edf_reader.close()
        
        if denoised_ppg != 'none':
            with h5py.File(f'{path_dataset}/preprocessing/pleth/{denoised_ppg}/{self.id:04}.hdf5', 'r') as f:
                self.psg['Pleth'] = pd.Series(np.array(f['signal']), dtype=float)

    def load_scorings(self, path_scoring, subject_id, events_as_list):
        self.hypnogram = pd.read_csv(path_scoring + f'hypnogram/hypnogram-{subject_id:04}.csv', header=None)[0]
        self.event_df  = pd.read_csv(path_scoring + f'events/events-{subject_id:04}.csv')

        self.events = None
        if events_as_list:
            self.events = []
            event_list = pd.read_csv(path_scoring + f'event_list/event-list-{subject_id:04}.csv')
            for _, event in event_list.iterrows():
                self.events.append(Event((event['Type'], event['Start'], event['End']), direct=True))

    def load_subject_data(self, path_subject, subject_id, subject_info):
        if subject_info is None:
            all_subjects = pd.read_csv(path_subject)
            all_subjects.set_index('mesaid', inplace=True)
            subject_info = all_subjects.loc[subject_id]
        
        self.subject_data = {
            'age': subject_info.loc['nsrr_age'],
            'bmi': subject_info.loc['nsrr_bmi'],
            'sex': subject_info.loc['nsrr_sex'],
            'race': subject_info.loc['nsrr_race'],
            'cur_smoker': subject_info.loc['nsrr_current_smoker'],
            'ever_smoked': subject_info.loc['nsrr_ever_smoker']
        }
    
    def post_process(self, use_predicted_hypnogram=False, path_dataset=None):
        ''' Do postprocessing after loading
        - Calculate total sleep time (TST)
        - Cut off the awake phase at the end
        - (Optional) Switch to predicted hypnogram
        '''

        awake_phases = self.hypnogram[self.hypnogram != 0]
        self.total_sleep_time_in_sec = len(awake_phases)

        end_point = min(awake_phases.index[-1]+10, len(self.hypnogram))
        self.hypnogram = self.hypnogram[0:end_point]
        self.event_df = self.event_df[0:end_point]
        for signal in self.psg.keys():
            dyn_end_point = end_point * self.psg_freqs[signal]
            self.psg[signal] = to_length(self.psg[signal], dyn_end_point)
        
        if use_predicted_hypnogram:
            predicted_hypnogram = pd.read_csv(f'{path_dataset}/predicted_hypnogram/mesa-{self.id:04}-1.csv', index_col='ts')['PPG_4cl']
            predicted_hypnogram[predicted_hypnogram == 'W'] = 0
            predicted_hypnogram[predicted_hypnogram == 'N1/N2'] = 1
            predicted_hypnogram[predicted_hypnogram == 'N3'] = 3
            predicted_hypnogram[predicted_hypnogram == 'R'] = 5
            self.hypnogram = to_length(pd.Series(np.repeat(predicted_hypnogram.values, 30)), end_point)
