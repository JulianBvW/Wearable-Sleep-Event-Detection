import torch
import numpy as np

def get_random_start(labels, max_time, seq_length):
    start = torch.randint(0, max_time, (1,)).item()
    end = start + seq_length
    return start, labels[start:end].sum() > 0

def get_random_starts(labels, batch_size, seq_length, max_time):
    tries = 0
    random_starts = []
    while len(random_starts) < batch_size:
        random_start, has_positive_class = get_random_start(labels, max_time, seq_length)
        if has_positive_class or tries >= batch_size:
            random_starts.append(random_start)
        tries += 1

    np.random.shuffle(random_starts)
    return random_starts

def get_batch(signals, labels, batch_size, seq_length):
    (hypnogram, spo2, pleth) = signals
    max_time = len(labels) - seq_length

    random_starts = get_random_starts(labels, batch_size, seq_length, max_time)

    batch_signals = []
    batch_labels = []
    for start in random_starts:
        end = start + seq_length
        seq_hypnogram = hypnogram[start:end].view((1, -1))
        seq_spo2 = spo2[start:end].view((1, -1))
        seq_pleth = pleth[start*256:end*256].view((256, -1))
        try:
            combined_signal = torch.cat([seq_hypnogram, seq_spo2, seq_pleth], dim=0)
        except:
            print(f'### FAIL at {start}')
            raise Exception(f'### FAIL at {start}')
        batch_signals.append(combined_signal)
        batch_labels.append(labels[start:end])

    return torch.stack(batch_signals), torch.stack(batch_labels)

def get_multi_batch(dataset, datapoint_ids, i, multi_batch_size, batch_size, seq_length):
    multi_batch_signals = []
    multi_batch_labels  = []
    for j in range(multi_batch_size):
        datapoint_id = datapoint_ids[multi_batch_size*i+j]
        (hypnogram, spo2, pleth), event_or_not = dataset.from_id(datapoint_id)
        try:
            batch_signal, batch_label = get_batch((hypnogram, spo2, pleth), event_or_not, batch_size, seq_length)
        except:
            print(f'### get_multi_batch at {i=}, {j=}, {datapoint_id=}')
            raise Exception(f'### get_multi_batch at {i=}, {j=}, {datapoint_id=}')
        multi_batch_signals.append(batch_signal)
        multi_batch_labels.append(batch_label)
    return torch.cat(multi_batch_signals), torch.cat(multi_batch_labels)

def get_test_batch(datapoint, seq_length, overlap_window):
    (hypnogram, spo2, pleth), event_or_not = datapoint
    step = seq_length - 2 * overlap_window
    batch_signals = []
    batch_labels  = []
    for start in range(0, len(event_or_not), step):
        end = start + seq_length
        seq_hypnogram = hypnogram[start:end].view((1, -1))
        seq_spo2 = spo2[start:end].view((1, -1))
        seq_pleth = pleth[start*256:end*256].view((256, -1))
        try:
            combined_signal = torch.cat([seq_hypnogram, seq_spo2, seq_pleth], dim=0)
        except:
            print(f'### FAIL (test) at {start}')
            raise Exception(f'### FAIL (test) at {start}')
        batch_signals.append(combined_signal)
        batch_labels.append(event_or_not[start:end])
    del batch_signals[-1]
    del batch_labels[-1]
    return torch.stack(batch_signals), torch.stack(batch_labels)

