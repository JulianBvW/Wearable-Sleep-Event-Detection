'''
Calculating AHI and TST from the predictions of the Apnea Detection Model for CFS dataset recordings
'''

from scipy.ndimage import binary_opening, binary_closing
import pandas as pd
import numpy as np
import torch

THR = 0.25


def correct(y, size=3):
    struct = np.ones(size, dtype=bool)
    closed = binary_closing(y == 1, structure=struct)  # 1. Merge events separated by short gaps
    opened = binary_opening(closed, structure=struct)  # 2. Remove events that are too short
    corrected = opened.astype(int)
    return corrected


def to_event_list(event_or_not):
    event_or_not = pd.Series(event_or_not)
    events = torch.tensor(event_or_not[event_or_not == 1].index.values)
    events_shift = torch.concat([torch.tensor([0]), events[:-1]])
    diff = events - events_shift

    starts = diff != 1
    ends = torch.concat([(diff != 1)[1:], torch.tensor([True])])
    
    event_list = [{'start': int(start), 'end': int(end)} for start, end in zip(events[starts], events[ends])] if len(starts) > 0 else []
    return event_list


def post_process(preds, hypnogram):
    hypnogram = pd.Series(hypnogram)
    tst = (hypnogram > 0).sum() / 60 / 60

    preds = pd.Series(preds)
    preds = (preds > THR)*1

    event_list = to_event_list(correct(preds))
    ahi = len(event_list) / tst
    return ahi, tst, event_list