from wearsed.dataset.utils import to_clock

class Event():
    def __init__(self, event, direct=False):

        if direct:
            self.type, self.start, self.end = event
            self.duration = self.end - self.start
        else:
            self.type = event['EventConcept'].split('|')[0]
            self.start = int(float(event['Start']))
            self.duration = int(float(event['Duration']))
            self.end = self.start + self.duration
    
    def __str__(self):
        return f'[{to_clock(self.start, detail=False)}, {to_clock(self.end, detail=False)}] {self.type}'