'''
Utility functions for the WearSED Dataset
'''

import pandas as pd

def to_obj(event):
    obj = {}
    for child in event:
        obj[child.tag] = child.text
    return obj

def to_clock(sec, detail=True):
    sec = int(sec)
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f'{h:02}:{m:02}:{s:02}\n({sec})' if detail else f'{h:02}:{m:02}:{s:02}'

def from_clock(clock):
    h, m, s = map(int, clock.split(':'))
    return h*60*60 + m*60 + s

def to_length(series, end_point):
    if len(series) >= end_point:
        return series[0:end_point]
    missing = pd.Series([0] * (end_point-len(series)))
    return pd.concat([series, missing], ignore_index=True)


EVENT_TYPES = ['OSA', 'CSA', 'MSA', 'HYP', 'ARO']
RESP_EVENT_TYPES = ['OSA', 'CSA', 'MSA', 'HYP']
EVENT_COLORS = {
    'OSA': 'purple',
    'CSA': 'gold',
    'MSA': 'grey',
    'HYP': 'cyan',
    'ARO': 'red'
}

EVENT_TYPES_NSRR =      ['Obstructive apnea', 'Central apnea', 'Mixed apnea', 'Hypopnea', 'Arousal']
RESP_EVENT_TYPES_NSRR = ['Obstructive apnea', 'Central apnea', 'Mixed apnea', 'Hypopnea']
from_nsrr = {
    'Obstructive apnea': 'OSA',
    'Central apnea':     'CSA',
    'Mixed apnea':       'MSA',
    'Hypopnea':          'HYP',
    'Arousal':           'ARO'
}

EVENT_TYPES_SOMNOLYZER =      ['ObstructiveApnea', 'CentralApnea', 'MixedApnea', 'Hypopnea', 'Arousal']
RESP_EVENT_TYPES_SOMNOLYZER = ['ObstructiveApnea', 'CentralApnea', 'MixedApnea', 'Hypopnea']
from_somnolyzer = {
    'ObstructiveApnea': 'OSA',
    'CentralApnea':     'CSA',
    'MixedApnea':       'MSA',
    'Hypopnea':         'HYP',
    'Arousal':          'ARO'
}










EVENT_COLORS_old = {
    'SpO2 desaturation': 'gold',
    'Hypopnea':          'cyan',
    'Unsure':            'grey',
    'Obstructive apnea': 'purple',
    'Central apnea':     'magenta',
    'SpO2 artifact':     'red',
    'Arousal':           'lime'
}

EVENT_TYPES_old = list(EVENT_COLORS_old.keys())