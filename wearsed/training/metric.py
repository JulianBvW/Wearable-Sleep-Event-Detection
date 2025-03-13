'''
Contains code to get a meaningful metric from "0 or 1" model output tensor
'''

from scipy.ndimage import binary_opening, binary_closing
import pandas as pd
import numpy as np
import torch
import time

def correct(y, size=10):
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

def do_overlap(ev1, ev2):
    if ev1['end'] >= ev2['start'] and ev1['start'] <= ev2['end']:  # Events overlap
        return 1
    if ev2['start'] > ev1['end']:  # ev2 (true event) happens after ev1 (pred event)
        return -1
    return 0                       # ev1 (pred event) happens after ev2 (true event)

def calc_metrics(y_pred_list, y_true_list):
    true_idx, pred_idx = 0, 0
    TPs = 0

    while pred_idx < len(y_pred_list) and true_idx < len(y_true_list):
        pred_event = y_pred_list[pred_idx]
        true_event = y_true_list[true_idx]

        match do_overlap(pred_event, true_event):
            case 1:
                TPs += 1
                true_idx += 1
                pred_idx += 1
            case -1:
                pred_idx += 1
            case 0:
                true_idx += 1

    FPs = len(y_pred_list) - TPs
    FNs = len(y_true_list) - TPs

    return TPs, FPs, FNs

def metric(y_pred, y_true, correctify=True):
    y_pred_list = to_event_list(correct(y_pred)) if correctify else to_event_list(y_pred)
    y_true_list = to_event_list(y_true)  # [0,0,0,1,1,1,1,1,0,0,0,1,1,1,0] -> [{start: 10, end: 29}, {start: 100, ..}]
    metrics = calc_metrics(y_pred_list, y_true_list)
    return metrics
