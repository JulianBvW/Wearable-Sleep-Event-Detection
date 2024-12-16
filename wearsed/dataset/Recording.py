'''
Class to capture all relevant recordings, annotations, and subject info from the PSG
'''

import pyedflib
import numpy as np
import pandas as pd
from lxml import etree
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

from wearsed.dataset.Event import Event
from wearsed.dataset.utils import from_clock, to_clock, to_obj, EVENT_COLORS, EVENT_TYPES

class Recording():
    def __init__(self, subject_id, subject_info=None, signals_to_read=['HR', 'SpO2', 'Flow', 'Pleth']):
        self.id = subject_id

        path_dataset = '/vol/sleepstudy/datasets/mesa/'
        path_psg     = path_dataset + f'polysomnography/edfs/mesa-sleep-{subject_id:04}.edf'
        path_annot   = path_dataset + f'polysomnography/annotations-events-nsrr/mesa-sleep-{subject_id:04}-nsrr.xml'
        path_subject = path_dataset + 'datasets/mesa-sleep-harmonized-dataset-0.7.0.csv'

        self.load_psg(path_psg, signals_to_read=signals_to_read)
        self.load_annotations(path_annot)
        self.load_subject_data(path_subject, subject_id, subject_info)

        self.post_process()
    
    def get_subject_info(self):
        return self.subject_data

    def get_event_count(self, event_type):
        return len(self.get_events(event_type))

    def get_events(self, event_type):
        event_types = event_type if type(event_type) is list else [event_type]
        return list(filter(lambda event: event.type in event_types, self.events))

    def get_ahi(self):  # Apnea Hypopnea Index
        return self.get_event_count(['Hypopnea', 'Obstructive apnea', 'Central apnea']) / (self.total_sleep_time_in_sec / 60 / 60)

    def get_ari(self):  # Arousal Index
        return self.get_event_count(['Arousal']) / (self.total_sleep_time_in_sec / 60 / 60)

    def look_at(self, time=None, window_size=None, events=EVENT_TYPES):
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

    def load_psg(self, path_psg, signals_to_read):
        edf_reader = pyedflib.EdfReader(path_psg)

        signal_labels = edf_reader.getSignalLabels()

        self.psg = {}
        self.psg_freqs = {}
        for i in range(edf_reader.signals_in_file):
            if signal_labels[i] in signals_to_read:
                self.psg[signal_labels[i]] = pd.Series(edf_reader.readSignal(i))
                self.psg_freqs[signal_labels[i]] = int(edf_reader.getSampleFrequency(i))
        
        edf_reader.close()

    def load_annotations(self, path_annot):
        all_events = etree.parse(path_annot).getroot().xpath("//ScoredEvent")

        self.events = []
        self.hypnogram = pd.Series(0, index=range(16*60*60))  # Empty Hypnogram of 16 hours

        for event in all_events:
            self.handle_event(to_obj(event))

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
        self.bla = subject_info # TODO delete this
    
    def handle_event(self, event):

        # Recording Start
        if event['EventType'] == None:
            self.recording_start = event['ClockTime']
        
        # Respiratory Events or Arousals
        elif event['EventType'] in ['Respiratory|Respiratory', 'Arousals|Arousals']:
            self.events.append(Event(event))
            
        # Stages
        elif event['EventType'] == 'Stages|Stages':
            stage = int(event['EventConcept'].split('|')[1])
            if stage > 5:
                return
            start = int(float(event['Start']))
            end = start + int(float(event['Duration']))
            self.hypnogram[start:end] = stage
    
    def post_process(self):
        ''' Do postprocessing after loading
        - Calculate total sleep time (TST)
        - Cut off the awake phase at the end
        - (TODO) Downsample ">1Hz" signals
        '''

        awake_phases = self.hypnogram[self.hypnogram != 0]
        self.total_sleep_time_in_sec = len(awake_phases)

        end_point = awake_phases.index[-1]+10
        self.hypnogram = self.hypnogram[0:end_point]
        for signal in self.psg.keys():
            dyn_end_point = end_point * self.psg_freqs[signal]
            self.psg[signal] = self.psg[signal][0:dyn_end_point]
