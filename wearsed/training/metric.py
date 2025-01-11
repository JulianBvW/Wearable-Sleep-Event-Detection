'''
Contains code to get a meaningful metric from "0 or 1" model output tensor
'''

from scipy.ndimage import binary_opening, binary_closing
import numpy as np

def correct(y, size=10):
    struct = np.ones(size, dtype=bool)
    closed = binary_closing(y == 1, structure=struct)  # 1. Merge events separated by short gaps
    opened = binary_opening(closed, structure=struct)  # 2. Remove events that are too short
    corrected = opened.astype(int)
    return corrected

def to_event_list(event_or_not):
    last = 0
    events = []
    current_event = {}
    for i, current in enumerate(event_or_not):
        current = current.item()
        if current == last:
            continue

        if current > last:  # Beginning of a new event
            current_event['start'] = i
            last = current
            continue

        if current < last:  # End of the new event
            current_event['end'] = i
            last = current
            events.append(current_event)
            current_event = {}
            continue
    return events

def do_overlap(ev1, ev2):
    return ev1['end'] >= ev2['start'] and ev1['start'] <= ev2['end']

def calc_metrics(y_pred_list, y_true_list):
    TPs = []
    for i, event_pred in enumerate(y_pred_list):
        for j, event_true in enumerate(y_true_list):
            if do_overlap(event_pred, event_true):
                TPs.append(i)
                del y_true_list[j]
                break
    for i in TPs[::-1]:
        del y_pred_list[i]
    TPs = len(TPs)
    FPs = len(y_pred_list)
    FNs = len(y_true_list)
    return TPs, FPs, FNs

def metric(y_pred, y_true):
    y_pred_list = to_event_list(correct(y_pred))
    y_true_list = to_event_list(y_true)
    return calc_metrics(y_pred_list, y_true_list)