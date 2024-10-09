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
        pass

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
        pass

    def load_annotations(self, path_annot):
        all_events = etree.parse(path_annot).getroot().xpath("//ScoredEvent")

        event_categories = {
            None: [],  # Recording Start
            'Respiratory|Respiratory': [],
            'Arousals|Arousals': [],
            'Limb Movement|Limb Movement': [],
            'Stages|Stages': []
        }

        for event in all_events:
            for child in event:
                if child.tag == 'EventType':
                    event_categories[child.text].append(Event(event))
        
        recording_start = recording_start[0]['ClockTime']

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


class Event():  # TODO!
    def __init__(self, event):
        obj = {}
        for child in event:
            if not child.tag == 'EventType':
                obj[child.tag] = child.text
        return obj

### EDF Files (/vol/sleepstudy/datasets/mesa/polysomnography/edfs/mesa-sleep-0001.edf)

def load_psg(path):
    edf_reader = pyedflib.EdfReader(path)

    signal_labels = edf_reader.getSignalLabels()

    signals = {}
    for i in range(edf_reader.signals_in_file):
        signals[signal_labels[i]] = pd.Series(edf_reader.readSignal(i))
    
    edf_reader.close()
    return signals

### R-Point CSV Files (/vol/sleepstudy/datasets/mesa/polysomnography/annotations-rpoints/mesa-sleep-0001-rpoint.csv)

def load_rpoint(path):
    df = pd.read_csv(path)
    return df

### NSRR Annotation Files (/vol/sleepstudy/datasets/mesa/polysomnography/annotations-events-nsrr/mesa-sleep-0001-nsrr.xml)

def to_obj(event):
    obj = {}
    for child in event:
        if not child.tag == 'EventType':
            obj[child.tag] = child.text
    return obj

def load_annotations(path):
    tree = etree.parse(path)
    root = tree.getroot()
    all_events = root.xpath("//ScoredEvent")

    recording_start = []
    events_respiratory = []
    events_arousals = []
    events_limb_movements = []
    events_stages = []

    event_categories = {
        None: recording_start,
        'Respiratory|Respiratory': events_respiratory,
        'Arousals|Arousals': events_arousals,
        'Limb Movement|Limb Movement': events_limb_movements,
        'Stages|Stages': events_stages
    }

    for event in all_events:
        for child in event:
            if child.tag == 'EventType':
                event_categories[child.text].append(to_obj(event))
    
    recording_start = recording_start[0]['ClockTime']
    return recording_start, \
        pd.DataFrame(events_respiratory), \
        pd.DataFrame(events_arousals), \
        pd.DataFrame(events_limb_movements), \
        pd.DataFrame(events_stages)
