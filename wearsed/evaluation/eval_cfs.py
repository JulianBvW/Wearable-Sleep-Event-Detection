'''
Evaluation code for the Attention U-Net model on the CFS dataset
'''

from wearsed.dataset.CFSDataset import CFSDataset
from wearsed.models.attention_unet.AttentionUNet import AttentionUNet

from tqdm import tqdm
import pandas as pd
import numpy as np
import os

import torch.nn as nn
import torch

def get_test_batch(datapoint, seq_length, overlap_window):
    (hypnogram, spo2, ppg), event_or_not = datapoint
    step = seq_length - 2 * overlap_window
    batch_signals = []
    batch_labels  = []
    for start in range(0, len(event_or_not), step):
        end = start + seq_length
        if len(event_or_not[start:end]) < seq_length:
            break
        seq_hypnogram = hypnogram[start:end].view((1, -1))
        seq_spo2      = spo2[start:end].view((1, -1))
        seq_ppg       = ppg.repeat_interleave(2)[start*256:end*256].view((256, -1))  # Repeat PPG for 256Hz
        combined_signal = torch.cat([seq_hypnogram, seq_spo2, seq_ppg], dim=0)
        batch_signals.append(combined_signal)
        batch_labels.append(event_or_not[start:end])
    return torch.stack(batch_signals), torch.stack(batch_labels)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Output folder
OUTPUT_DIR = f'wearsed/evaluation/output/with_scorings/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Dataset and Model
dataset = CFSDataset()
model = AttentionUNet().to(device)
model.load_state_dict(torch.load('wearsed/training/attention_unet/output/final_default/f-0/model_final.pth', weights_only=True))
model.eval()

# Evaluation
predictions, targets = [], []
with torch.no_grad():
    for i in tqdm(range(len(dataset))):
        
        cfs_id = dataset.get_cfs_id(i)
        x, y = get_test_batch(dataset[i], seq_length=30*60, overlap_window=60)
        x, y = x.to(device), y.to(device)

        # Forward pass
        y_hat = model(x)
        prediction = torch.sigmoid(y_hat)
        
        # Remember predictions
        predictions.append(torch.tensor([cfs_id]))
        predictions.append(prediction.cpu()[:, 60:30*60-60].flatten())
        targets.append(torch.tensor([-999]))
        targets.append(y.cpu()[:, 60:30*60-60].flatten())

# Save predictions
predictions, targets = torch.cat(predictions), torch.cat(targets)
pd.DataFrame({'targets': targets, 'predictions': predictions}).to_csv(OUTPUT_DIR + f'/test_preds.csv', index=False)
