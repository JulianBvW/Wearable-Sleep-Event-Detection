'''
Train code for training the Baseline model
'''

from wearsed.dataset.WearSEDDataset_vae import WearSEDDataset, get_shakiness
from wearsed.models.vae.VAE import VAE
from wearsed.models.baseline_conv.BaselineConv import BaselineConv

from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd
import os

from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import torch.nn as nn
import torch

parser = ArgumentParser(description='Train code for training the Baseline model')
parser.add_argument('--epochs', help='number of epochs', default=50, type=int)
parser.add_argument('--batch-size', help='batch size', default=64, type=int)
parser.add_argument('--seq-length', help='length of the individual segments parsed to the model', default=10*60, type=int)
parser.add_argument('--out-dir', help='name of the output directory', default='output', type=str)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def vae_downsample(signal, freq):
    downsampled = []
    for sec in range(len(signal)//freq):
        seq = signal[sec*freq:sec*freq+freq]
        seq = seq.to(device)
        enc = vae_model.encode(seq.unsqueeze(0))[0].squeeze(0).to('cpu')
        downsampled.append(enc)
    return torch.stack(downsampled, dim=1)

def get_random_sequence(signals, labels, pleth, max_time, seq_length):
    start = torch.randint(0, max_time, (1,))
    end = start + seq_length
    return signals[:, start:end], labels[start:end], pleth[start*256:end*256]

def get_batch(signals, labels, pleth, batch_size, seq_length):
    max_time = signals.shape[1] - seq_length

    tries = 0
    batch_signals = []
    batch_labels = []
    for i in range(batch_size):
        signals_seq, labels_seq, pleth_seq = get_random_sequence(signals, labels, pleth, max_time, seq_length)
        if i < 0.8*batch_size or tries < 20:  # If there are not enough sequences with events and we haven't tried long enough
            if labels_seq.sum() == 0:         # ..check if this is a sequence without events and if so, roll again
                signals_seq, labels_seq, pleth_seq = get_random_sequence(signals, labels, pleth, max_time, seq_length)  # TODO could still be negative sample
                ds = vae_downsample(pleth_seq, 256)
                batch_signals.append(torch.cat([signals_seq, ds]))
                batch_labels.append(labels_seq)
                tries += 1
                continue
        ds = vae_downsample(pleth_seq, 256)
        batch_signals.append(torch.cat([signals_seq, ds]))
        batch_labels.append(labels_seq)
    return torch.stack(batch_signals), torch.stack(batch_labels)

# Output folder
OUTPUT_DIR = f'wearsed/training/baseline_conv/{args.out_dir}/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset
full_dataset = WearSEDDataset(signals_to_read=['SpO2', 'Pleth'], preprocess=True)
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])

# Model, Optimizer, Criterion
model = BaselineConv(in_channels=18).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))

vae_model = VAE(input_dim=256, latent_dim=16).to(device)
vae_model.load_state_dict(torch.load('wearsed/training/vae/output/bs_256_epochs_10/latent_size_16/model.pth', weights_only=True, map_location=device))
vae_model.eval()

# Training Loop
train_losses = []
test_losses = []
#cm_tn, cm_fp, cm_fn, cm_tp = [], [], [], []
#accuracies = []

for epoch in range(args.epochs):

    # Taining
    model.train()
    train_loss = 0
    for signals, labels, pleth in tqdm(train_dataset):
        optimizer.zero_grad()

        x, y = get_batch(signals, labels, pleth, batch_size=args.batch_size, seq_length=args.seq_length)
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
        for signals, labels, pleth in tqdm(test_dataset):

            x, y = get_batch(signals, labels, pleth, batch_size=args.batch_size, seq_length=args.seq_length)
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
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_dataset):.4f}, Test Loss: {test_loss / len(test_dataset):.4f}')
    train_losses.append(train_loss / len(train_dataset))
    test_losses.append(test_loss / len(test_dataset))
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

