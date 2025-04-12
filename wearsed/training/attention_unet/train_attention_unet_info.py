'''
Train code for training the Attention U-Net model
'''

from wearsed.training.utils import show_script_info
from wearsed.training.metric import get_best_f1_score
from wearsed.training.kfold.load_kfold import get_fold
from wearsed.dataset.WearSEDDataset_info import WearSEDDataset
from wearsed.training.batching import get_multi_batch, get_test_batch
from wearsed.models.attention_unet.AttentionUNet import AttentionUNet

from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

import torch.nn as nn
import torch

parser = ArgumentParser(description='Train code for training the Attention U-Net model')
parser.add_argument('--epochs', help='number of epochs', default=30, type=int)
parser.add_argument('--fold-nr', help='which fold to use with k-fold', default=0, type=int, required=True)
parser.add_argument('--batch-size', help='how many random sequences per recording', default=32, type=int)
parser.add_argument('--out-dir', help='name of the output directory', default=None, type=str, required=True)
parser.add_argument('--multi-batch-size', help='how many different recordings per batch', default=4, type=int)
parser.add_argument('--seq-length', help='length of the individual segments parsed to the model', default=30*60, type=int)
parser.add_argument('--use-attention', help='what attention parts should be used [gates,bottleneck]', default='', type=str, required=False)
parser.add_argument('--use-predicted-hypnogram', help='use the predicted hypnogram instead of ground truth', action='store_true')
parser.add_argument('--denoised-ppg', help='which denoising strategy to use if wanted [lowpass,bandpass,wavelet_[db4,dmey,haar]]', default='none', type=str, required=False)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

OVERLAP_WINDOW = 60  # 1 minute
FOLD_NAME = 'fold-4-somnolyzer'
SEED = 42

# Output folder
OUTPUT_DIR = f'wearsed/training/attention_unet/output/{args.out_dir}/f-{args.fold_nr}/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset
full_dataset = WearSEDDataset(signals_to_read=['SpO2', 'Pleth'], use_predicted_hypnogram=args.use_predicted_hypnogram, denoised_ppg=args.denoised_ppg)
train_ids, test_ids = get_fold(FOLD_NAME, args.fold_nr, seed=SEED)

# Model, Optimizer, Criterion
model = AttentionUNet(in_channels=6, use_attention=args.use_attention.split(',')).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

show_script_info(args, model, 2+256)

# Training Loop
train_losses = []
test_losses = []
best_f1s = []

train_fails = 0
test_fails = 0

for epoch in range(args.epochs):

    # Taining
    model.train()
    train_loss, train_batches = 0, 0
    for i in tqdm(range(len(train_ids) // args.multi_batch_size)):
        optimizer.zero_grad()

        x, y = get_multi_batch(full_dataset, train_ids, i, multi_batch_size=args.multi_batch_size, batch_size=args.batch_size, seq_length=args.seq_length)
        x, y = x.to(device), y.to(device)
        y[y>0] = 1
        
        # Forward pass
        y_hat = model(x)
        
        # Loss computation
        loss = criterion(y_hat, y)
        loss.backward()
        train_loss += loss.item() * y.shape[0]
        train_batches += y.shape[0]
        optimizer.step()
    
    # Testing
    model.eval()
    test_loss, test_batches = 0, 0
    predictions, targets, hypnogram_data, class_data = [], [], [], []
    with torch.no_grad():
        for i in tqdm(range(len(test_ids))):
            
            x, y = get_test_batch(full_dataset.from_id(test_ids[i]), seq_length=args.seq_length, overlap_window=OVERLAP_WINDOW)
            x, y = x.to(device), y.to(device)
            class_data.append(torch.clone(y).cpu()[:, OVERLAP_WINDOW:args.seq_length-OVERLAP_WINDOW].flatten())
            class_data.append(torch.tensor([-999]))
            y[y>0] = 1

            # Forward pass
            y_hat = model(x)
            prediction = torch.sigmoid(y_hat)
            
            # Loss computation
            loss = criterion(y_hat, y)
            test_loss += loss.item() * y.shape[0]
            test_batches += y.shape[0]
            targets.append(y.cpu()[:, OVERLAP_WINDOW:args.seq_length-OVERLAP_WINDOW].flatten())
            targets.append(torch.tensor([-999]))
            predictions.append(prediction.cpu()[:, OVERLAP_WINDOW:args.seq_length-OVERLAP_WINDOW].flatten())
            predictions.append(torch.tensor([-499]))
            hypnogram_data.append(x.cpu()[:, 0, OVERLAP_WINDOW:args.seq_length-OVERLAP_WINDOW].flatten())
            hypnogram_data.append(torch.tensor([-199]))
    
    predictions, targets, hypnogram_data, class_data = torch.cat(predictions), torch.cat(targets), torch.cat(hypnogram_data), torch.cat(class_data)
    best_f1, _, _ = get_best_f1_score(predictions, targets)
    
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss / train_batches:.4f}, Test Loss: {test_loss / test_batches:.4f}, Best F1: {best_f1}')
    train_losses.append(train_loss / train_batches)
    test_losses.append(test_loss / test_batches)
    best_f1s.append(best_f1)

    # Save intermediate results
    if epoch % 4 == 0:
        pd.DataFrame({'targets': targets, 'predictions': predictions, 'hypnogram_data': hypnogram_data, 'class_data': class_data}).to_csv(OUTPUT_DIR + f'/test_preds_epoch_{epoch}.csv', index=False)
    if epoch % 10 == 0:
        torch.save(model.state_dict(), OUTPUT_DIR + f'/model_epoch_{epoch}.pth')

# Save model and losses
torch.save(model.state_dict(), OUTPUT_DIR + '/model_final.pth')
results = pd.DataFrame({
    'train_losses': train_losses,
    'test_losses': test_losses,
    'best_f1': best_f1s
})
results.to_csv(OUTPUT_DIR + '/losses.csv', index=False)

print(results)
