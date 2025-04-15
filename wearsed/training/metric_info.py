'''
Contains code to get a meaningful metric from "0 or 1" model output tensor
'''

from scipy.ndimage import binary_opening, binary_closing
import pandas as pd
import numpy as np
import torch

def correct(y, size=10):
    struct = np.ones(size, dtype=bool)
    closed = binary_closing(y == 1, structure=struct)  # 1. Merge events separated by short gaps
    opened = binary_opening(closed, structure=struct)  # 2. Remove events that are too short
    corrected = opened.astype(int)
    return corrected

def to_event_list(event_or_not, hypnogram_data, class_data):
    event_or_not = pd.Series(event_or_not)
    events = torch.tensor(event_or_not[event_or_not == 1].index.values)
    events_shift = torch.concat([torch.tensor([0]), events[:-1]])
    diff = events - events_shift

    starts = diff != 1
    ends = torch.concat([(diff != 1)[1:], torch.tensor([True])])
    
    event_list = [{'start': int(start), 'end': int(end), 'duration': int(end)-int(start), 'sleep_stage': int(hypnogram_data[int(start)]), 'event_class': int(class_data[int(start)])} for start, end in zip(events[starts], events[ends])] if len(starts) > 0 else []
    return event_list

def do_overlap(ev1, ev2):
    if ev1['end'] >= ev2['start'] and ev1['start'] <= ev2['end']:  # Events overlap
        return 1
    if ev2['start'] > ev1['end']:  # ev2 (true event) happens after ev1 (pred event)
        return -1
    return 0                       # ev1 (pred event) happens after ev2 (true event)

def calc_metrics(y_pred_list, y_true_list):
    true_idx, pred_idx = 0, 0
    info_durations_true = []
    info_durations_pred = []
    info_sleep_stage_found = {i: 0 for i in range(6)}
    info_sleep_stage_not_found = {i: 0 for i in range(6)}
    info_event_class_found = {i: 0 for i in range(16)}
    info_event_class_not_found = {i: 0 for i in range(16)}
    TPs = 0

    while pred_idx < len(y_pred_list) and true_idx < len(y_true_list):
        pred_event = y_pred_list[pred_idx]
        true_event = y_true_list[true_idx]

        match do_overlap(pred_event, true_event):
            case 1:
                TPs += 1
                true_idx += 1
                pred_idx += 1
                info_durations_true.append(true_event['duration'])
                info_durations_pred.append(pred_event['duration'])
                info_sleep_stage_found[true_event['sleep_stage']] += 1
                info_event_class_found[true_event['event_class']] += 1
            case -1:
                pred_idx += 1
            case 0:
                true_idx += 1
                info_sleep_stage_not_found[true_event['sleep_stage']] += 1
                info_event_class_not_found[true_event['event_class']] += 1

    FPs = len(y_pred_list) - TPs
    FNs = len(y_true_list) - TPs

    return TPs, FPs, FNs, info_durations_true, info_durations_pred, info_sleep_stage_found, info_sleep_stage_not_found, info_event_class_found, info_event_class_not_found

def metric(y_pred, y_true, hypnogram_data, class_data, correctify=True, correctify_size=10):
    y_pred_list = to_event_list(correct(y_pred, size=correctify_size), hypnogram_data, class_data) if correctify else to_event_list(y_pred, hypnogram_data, class_data)
    y_true_list = to_event_list(y_true, hypnogram_data, class_data)
    metrics = calc_metrics(y_pred_list, y_true_list)
    return metrics

def combine_fold_results(run, folds, epoch, drop_sep=True):
    outputs = []
    for fold in folds:
        output = pd.read_csv(f'../wearsed/training/attention_unet/output/{run}/f-{fold}/test_preds_epoch_{epoch}.csv')
        outputs.append(output)
    output = pd.concat(outputs, ignore_index=True)
    if drop_sep:
        output = output.drop(output[output['targets'] == -999].index)
        output = output.reset_index(drop=True)
    y_pred, y_true = output['predictions'], output['targets']
    hypnogram_data, class_data = output['hypnogram_data'], output['class_data']
    return y_pred, y_true, hypnogram_data, class_data
