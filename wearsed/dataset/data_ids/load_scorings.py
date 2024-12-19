import pandas as pd
from lxml import etree

from wearsed.dataset.Event import Event # TODO needed?
from wearsed.dataset.utils import EVENT_TYPES_NSRR, RESP_EVENT_TYPES_NSRR, EVENT_TYPES_SOMNOLYZER, RESP_EVENT_TYPES_SOMNOLYZER, from_nsrr, from_somnolyzer, from_clock, to_clock, to_obj, EVENT_COLORS, EVENT_TYPES ## TODO names, NOR OSA CSA MSA HYP ARO


def load_scorings_nsrr(mesa_root, id):
    annot_file = mesa_root + f'polysomnography/annotations-events-nsrr/mesa-sleep-{id:04}-nsrr.xml'
    all_scorings = etree.parse(annot_file).getroot().xpath("//ScoredEvent")

    # Handle Scoring
    events = []
    hypnogram = pd.Series(0, index=range(16*60*60))  # Empty Hypnogram of 16 hours
    for scoring in all_scorings:
        event_or_stage, scoring = handle_scoring_nsrr(to_obj(scoring))
        if event_or_stage == 'event':
            events.append(scoring)
        elif event_or_stage == 'stage':
            stage, start, end = scoring
            hypnogram[start:end] = stage
    
    # Filter interesting events and respiratory events from wake phases
    events = list(filter(lambda ev: ev.type in EVENT_TYPES_NSRR, events))  # Filter out uninteresting events
    events = list(filter(lambda ev: ev.type not in RESP_EVENT_TYPES_NSRR or hypnogram[ev.start] > 0, events))  # Filter out respiratory events during wake

    # Cut off hypnogram at the end
    recording_length = hypnogram[hypnogram != 0].index[-1]+10
    hypnogram = hypnogram[:recording_length]

    # Create event DataFrame
    event_df = pd.DataFrame(0, index=range(recording_length), columns=EVENT_TYPES_NSRR)
    for event in events:
        event_df.loc[event.start:event.end, event.type] = 1
    event_df.columns = EVENT_TYPES  # Use [OSA, HYP, ...] instead of long form names
    
    event_list_type, event_list_start, event_list_end = [], [], []
    for event in events:
        event_list_type.append(from_nsrr[event.type])
        event_list_start.append(event.start)
        event_list_end.append(event.end)
    event_list = pd.DataFrame({
        'Type': event_list_type,
        'Start': event_list_start,
        'End': event_list_end
    })

    hypnogram.to_csv(mesa_root + f'scorings/nsrr/hypnogram/hypnogram-{id:04}.csv', header=False, index=False)
    event_df.to_csv(mesa_root + f'scorings/nsrr/events/events-{id:04}.csv', index=False)
    event_list.to_csv(mesa_root + f'scorings/nsrr/event_list/event-list-{id:04}.csv', index=False)

def load_scorings_somnolyzer(mesa_root, id):
    root = etree.parse(mesa_root + f'somnolyzer_scorings/mesa-sleep-{id:04}.rml').getroot()
    scoring_root = root.find('{http://www.respironics.com/PatientStudy.xsd}ScoringData')

    hypnogram = read_hypnogram(scoring_root.find('{http://www.respironics.com/PatientStudy.xsd}StagingData').find('{http://www.respironics.com/PatientStudy.xsd}MachineStaging').find('{http://www.respironics.com/PatientStudy.xsd}NeuroAdultAASMStaging'))
    events = read_events(scoring_root.find('{http://www.respironics.com/PatientStudy.xsd}Events'))

    # Filter events during wake
    events = list(filter(lambda ev: ev['Start'] < len(hypnogram), events))  # Filter out events outside hypnogram
    events = list(filter(lambda ev: ev['Type'] not in RESP_EVENT_TYPES_SOMNOLYZER or hypnogram[ev['Start']] > 0, events))

    # Create event DataFrame
    event_df = pd.DataFrame(0, index=range(len(hypnogram)), columns=EVENT_TYPES_SOMNOLYZER)
    for event in events:
        event_df.loc[event['Start']:event['End'], event['Type']] = 1
    event_df.columns = EVENT_TYPES  # Use [OSA, HYP, ...] instead of long form names
    
    event_list_type, event_list_start, event_list_end = [], [], []
    for event in events:
        event_list_type.append(from_somnolyzer[event['Type']])
        event_list_start.append(event['Start'])
        event_list_end.append(event['End'])
    event_list = pd.DataFrame({
        'Type': event_list_type,
        'Start': event_list_start,
        'End': event_list_end
    })

    hypnogram.to_csv(mesa_root + f'scorings/somnolyzer/hypnogram/hypnogram-{id:04}.csv', header=False, index=False)
    event_df.to_csv(mesa_root + f'scorings/somnolyzer/events/events-{id:04}.csv', index=False)
    event_list.to_csv(mesa_root + f'scorings/somnolyzer/event_list/event-list-{id:04}.csv', index=False)





###########################
# nsrr

def handle_scoring_nsrr(scoring):
    
    # Respiratory Events or Arousals
    if scoring['EventType'] in ['Respiratory|Respiratory', 'Arousals|Arousals']:
        return 'event', Event(scoring)
        
    # Stages
    elif scoring['EventType'] == 'Stages|Stages':
        stage = int(scoring['EventConcept'].split('|')[1])
        if stage > 5:
            return '', None
        start = int(float(scoring['Start']))
        end = start + int(float(scoring['Duration']))
        return 'stage', (stage, start, end)
    
    return '', None

###########################
# somnolyzer

translate = {
    'Wake': 0,
    'NonREM1': 1,
    'NonREM2': 2,
    'NonREM3': 3,
    'REM': 5
}

def read_hypnogram(hypnogram_scoring_root):
    stages = []
    for child in hypnogram_scoring_root:
        stages.append(child.attrib)
        
    hypnogram = pd.Series(0, index=range(int(stages[-1]['Start'])+10))

    for a, b in zip(stages, stages[1:]):
        stage = translate[a['Type']]
        start = int(a['Start'])
        end   = int(b['Start'])
        hypnogram[start:end] = stage
    
    return hypnogram

def read_events(events_root):
    events = []
    for ev in events_root:
        ev = ev.attrib
        if ev['Family'] in ['Respiratory', 'Neuro']:
            event_type = ev['Type']
            if event_type in EVENT_TYPES_SOMNOLYZER:
                event_start = int(float(ev['Start']))
                event_end = event_start + int(float(ev['Duration']))
                events.append({'Type': event_type, 'Start': event_start, 'End': event_end})
    return events
