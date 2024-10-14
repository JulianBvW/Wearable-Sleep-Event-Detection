from wearsed.dataset.utils import to_clock

class Event():
    def __init__(self, event):
        self.type = event['EventConcept'].split('|')[0]  # SpO2 desaturation, Hypopnea, Unsure, Obstructive apnea, SpO2 artifact, Arousal
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