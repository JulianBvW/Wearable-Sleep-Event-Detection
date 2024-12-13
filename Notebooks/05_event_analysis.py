from wearsed.dataset.WearSEDDataset import WearSEDDataset
from tqdm import tqdm
import pandas as pd

dataset = WearSEDDataset()

AHIs = {i/2: 0 for i in range(100)}
event_durations = {i: 0 for i in range(100)}

try:
    for recording in tqdm(dataset):

        # Calculate AHIs
        ahi = round(recording.get_ahi()*2)/2
        if ahi not in AHIs.keys():
            print(f'Very high AHI detected: {ahi} in {recording.id}')
            AHIs[ahi] = 0
        AHIs[ahi] += 1

        # Calculate Event Durations
        events = recording.get_events(['Hypopnea', 'Obstructive apnea'])
        for event in events:
            duration = round(event.duration)
            if duration not in event_durations.keys():
                print(f'Very high duration detected: {duration} in {recording.id}')
                event_durations[duration] = 0
            event_durations[duration] += 1
except:
    pass

pd.Series(AHIs).to_csv('05_AHIs.csv', index=False)
pd.Series(event_durations).to_csv('05_event_durations.csv', index=False)
