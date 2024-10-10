'''
Class to capture all relevant recordings, annotations, and subject info from the PSG
'''

import pyedflib
import pandas as pd
from lxml import etree

class Recording():
    def __init__(self, subject_id, subject_info=None):
        self.id = subject_id

        path_dataset = '/vol/sleepstudy/datasets/mesa/'
        path_psg     = path_dataset + f'polysomnography/edfs/mesa-sleep-{subject_id:04}.edf'
        path_rpoint  = path_dataset + f'polysomnography/annotations-rpoints/mesa-sleep-{subject_id:04}-rpoint.csv'
        path_annot   = path_dataset + f'polysomnography/annotations-events-nsrr/mesa-sleep-{subject_id:04}-nsrr.xml'
        path_subject = path_dataset + 'datasets/mesa-sleep-harmonized-dataset-0.7.0.csv'

        self.load_psg(path_psg)
        self.load_rpoints(path_rpoint)
        self.load_annotations(path_annot)
        self.load_subject_data(path_subject, subject_id, subject_info)
    
    def get_subject_info(self):
        return self.subject_data

    def get_total_time(self):
        pass

    def get_event_count(self, event):
        pass

    def get_events(self, event):
        pass

    def get_ahi(self):
        pass

    def look_at(self, sec):
        pass

    def load_psg(self, path_psg, signals_to_read=['HR', 'SpO2', 'Pleth']):
        edf_reader = pyedflib.EdfReader(path_psg)

        signal_labels = edf_reader.getSignalLabels()

        self.psg = {}
        for i in range(edf_reader.signals_in_file):
            if signal_labels[i] in signals_to_read:
                self.psg[signal_labels[i]] = pd.Series(edf_reader.readSignal(i))
        
        edf_reader.close()

    def load_rpoints(self, path_rpoint):
        pass  # TODO Can I use them?

    def load_annotations(self, path_annot):
        all_events = etree.parse(path_annot).getroot().xpath("//ScoredEvent")

        self.events = []
        self.sleep_stages = []

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
            self.sleep_stages += [stage]*int(event['Duration'].split('.')[0])

class Event():
    def __init__(self, event):
        self.type = event['EventConcept'].split('|')[0]
        self.start = float(event['Start'])
        self.duration = float(event['Duration'])
        self.end = self.start + self.duration

        # TODO needed?
        self.location = event['SignalLocation']
        self.SpO2Nadir = event['SpO2Nadir'] if 'SpO2Nadir' in event.keys() else None
        self.SpO2Baseline = event['SpO2Baseline'] if 'SpO2Baseline' in event.keys() else None
    
    def __str__(self):
        SpO2 = '' if self.SpO2Nadir is None else f' ({self.SpO2Nadir}, base {self.SpO2Baseline})'
        return f'[{to_clock(self.start)}, {to_clock(self.end)}] {self.type} at {self.location}{SpO2}'

def to_obj(event):
    obj = {}
    for child in event:
        obj[child.tag] = child.text
    return obj

def to_clock(sec):
    sec = int(sec)
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f'{h:02}:{m:02}:{s:02}'
