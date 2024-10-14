'''
Utility functions for the WearSED Dataset
'''

def to_obj(event):
    obj = {}
    for child in event:
        obj[child.tag] = child.text
    return obj

def to_clock(sec):
    sec = int(sec)
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f'{h:02}:{m:02}:{s:02}\n({sec})'

def from_clock(clock):
    h, m, s = map(int, clock.split(':'))
    return h*60*60 + m*60 + s

EVENT_COLORS = {
    'SpO2 desaturation': 'gold',
    'Hypopnea':          'magenta',
    'Unsure':            'grey',
    'Obstructive apnea': 'purple',
    'SpO2 artifact':     'cyan',
    'Arousal':           'lime'
}

EVENT_TYPES = list(EVENT_COLORS.keys())
