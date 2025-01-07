'''
Train code for training the Baseline model
'''

from wearsed.dataset.WearSEDDataset import WearSEDDataset
from wearsed.models.baseline_conv.BaselineConv_no_spo2 import BaselineConv

from argparse import ArgumentParser
from random import shuffle
from tqdm import tqdm
import pandas as pd
import os

import torch.nn.functional as F
import torch.nn as nn
import torch

parser = ArgumentParser(description='Train code for training the Baseline model')
parser.add_argument('--epochs', help='number of epochs', default=50, type=int)
parser.add_argument('--batch-size', help='how many random sequences per recording', default=32, type=int)
parser.add_argument('--multi-batch-size', help='how many different recordings per batch', default=4, type=int)
parser.add_argument('--seq-length', help='length of the individual segments parsed to the model', default=30*60, type=int)
parser.add_argument('--out-dir', help='name of the output directory', default=None, type=str, required=True)
args = parser.parse_args()

print('--=={ Running script with arguments }==--')
for k, v in vars(args).items():
    print(f'    {k:17}: {v}')
print('--=====================================--')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_random_start(labels, max_time, seq_length):
    start = torch.randint(0, max_time, (1,)).item()
    end = start + seq_length
    return start, labels[start:end].sum() > 0

def get_batch(signals, labels, batch_size, seq_length):
    (hypnogram, spo2, pleth) = signals
    max_time = len(labels) - seq_length

    tries = 0
    random_starts = []
    while len(random_starts) < batch_size:
        random_start, has_positive_class = get_random_start(labels, max_time, seq_length)
        if has_positive_class or tries >= batch_size // 2:
            random_starts.append(random_start)
        tries += 1
    shuffle(random_starts)

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

def get_multi_batch(dataset, i, multi_batch_size, batch_size, seq_length):
    multi_batch_signals = []
    multi_batch_labels  = []
    for j in range(multi_batch_size):
        (hypnogram, spo2, pleth), event_or_not = dataset[multi_batch_size*i+j]
        try:
            batch_signal, batch_label = get_batch((hypnogram, spo2, pleth), event_or_not, batch_size, seq_length)
        except:
            print(f'### get_multi_batch at {i=}, {j=}')
            raise Exception(f'### get_multi_batch at {i=}, {j=}')
        multi_batch_signals.append(batch_signal)
        multi_batch_labels.append(batch_label)
    return torch.cat(multi_batch_signals), torch.cat(multi_batch_labels)

# Output folder
OUTPUT_DIR = f'wearsed/training/baseline_conv/output/{args.out_dir}/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset
full_dataset = WearSEDDataset(signals_to_read=['SpO2', 'Pleth'])
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])

# Model, Optimizer, Criterion
model = BaselineConv(in_channels=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

# Training Loop
train_losses = []
test_losses = []

train_fails = 0
test_fails = 0

for epoch in range(args.epochs):

    train_fails = 0
    test_fails = 0

    # Taining
    model.train()
    train_loss = 0
    for i in tqdm(range(len(train_dataset) // args.multi_batch_size)):
        optimizer.zero_grad()

        try:
            x, y = get_multi_batch(train_dataset, i, multi_batch_size=args.multi_batch_size, batch_size=args.batch_size, seq_length=args.seq_length)
        except:
            print(f'### Failed at TRAINING {i}')
            train_fails += 1
            continue
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        y_hat = model(x)
        
        # Loss computation
        loss = criterion(y_hat, y)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    # Testing
    model.eval()
    test_loss = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset) // args.multi_batch_size)):
            
            try:
                x, y = get_multi_batch(train_dataset, i, multi_batch_size=args.multi_batch_size, batch_size=args.batch_size, seq_length=args.seq_length)
            except:
                print(f'### Failed at TEST {i}')
                test_fails += 1
                continue
            x, y = x.to(device), y.to(device)

            # Forward pass
            y_hat = model(x)
            prediction = torch.sigmoid(y_hat)
            
            # Loss computation
            loss = criterion(y_hat, y)
            test_loss += loss.item()
            predictions.append(prediction.cpu().flatten())
            targets.append(y.cpu().flatten())
    
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    pd.DataFrame({'targets': targets, 'predictions': predictions}).to_csv(OUTPUT_DIR + f'/test_preds_epoch_{epoch}.csv', index=False)
    #tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
    #accuracy = (tn+tp)/(tn+fp+fn+tp)
    
    #print(f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_dataset):.4f}, Test Loss: {test_loss / len(test_dataset):.4f}, Test Accuracy: {accuracy*100:.3}% ({tn=}, {fp=}, {fn=}, {tp=})')
    train_ds_len = len(train_dataset) - (len(train_dataset) % args.multi_batch_size) - train_fails*args.multi_batch_size
    test_ds_len  = len(test_dataset)  - (len(test_dataset) % args.multi_batch_size) - test_fails*args.multi_batch_size
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss / train_ds_len:.4f}, Test Loss: {test_loss / test_ds_len:.4f}')
    train_losses.append(train_loss / train_ds_len)
    test_losses.append(test_loss / test_ds_len)
    # cm_tn.append(tn)
    # cm_fp.append(fp)
    # cm_fn.append(fn)
    # cm_tp.append(tp)
    # accuracies.append(accuracy)

    if epoch % 10 == 0:
        torch.save(model.state_dict(), OUTPUT_DIR + f'/model_epoch_{epoch}.pth')

# Save model and losses
torch.save(model.state_dict(), OUTPUT_DIR + '/model_final.pth')
results = pd.DataFrame({
    'train_losses': train_losses,
    'test_losses': test_losses,
    # 'cm_tn': cm_tn,
    # 'cm_fp': cm_fp,
    # 'cm_fn': cm_fn,
    # 'cm_tp': cm_tp,
    # 'accuracies': accuracies
})
results.to_csv(OUTPUT_DIR + '/losses.csv', index=False)

print(results)

