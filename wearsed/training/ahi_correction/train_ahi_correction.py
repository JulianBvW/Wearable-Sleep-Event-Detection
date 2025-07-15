'''
Train code for training the AHI correction model
'''

from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

import torch.nn as nn
import torch

parser = ArgumentParser(description='Train code for training the AHI correction model')
parser.add_argument('--epochs', help='number of epochs', default=20, type=int)
parser.add_argument('--fold-nr', help='which fold to use with k-fold', default=0, type=int, required=True)
parser.add_argument('--batch-size', help='how many random sequences per recording', default=64, type=int)
parser.add_argument('--out-dir', help='name of the output directory', default=None, type=str, required=True)
args = parser.parse_args()


class AHICorrectionModel(nn.Module):
    def __init__(self):
        super(AHICorrectionModel, self).__init__()
        
        self.hidden = nn.Linear(6, 12)
        self.relu = nn.ReLU()
        self.out = nn.Linear(12, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        return self.out(x)

def get_fold(full_dataset, fold_nr):
    ds_len = len(full_dataset)
    fold_len = ds_len // 4
    folds = [
        full_dataset[:fold_len],
        full_dataset[fold_len:fold_len*2],
        full_dataset[fold_len*2:fold_len*3],
        full_dataset[fold_len*3:]
    ]
    train_dataset = pd.concat(folds[:fold_nr] + folds[fold_nr+1:])
    test_dataset = folds[fold_nr]
    return train_dataset, test_dataset

def get_X_Y(ds):
    X = ds[['age', 'bmi', 'sex', 'cur_smoker', 'ever_smoker', 'pred_ahi']].values
    Y = ds['true_ahi'].values
    X, Y = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    X, Y = torch.nan_to_num(X), torch.nan_to_num(Y)
    return X, Y

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Output folder
OUTPUT_DIR = f'wearsed/training/ahi_correction/output/{args.out_dir}/f-{args.fold_nr}/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset
full_dataset = pd.read_csv('Notebooks/63_ahi_dataset.csv', index_col=False)
train_dataset, test_dataset = get_fold(full_dataset, args.fold_nr)
X_train, Y_train = get_X_Y(train_dataset)
X_test, Y_test = get_X_Y(test_dataset)

# Model, Optimizer, Criterion
model = AHICorrectionModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training Loop
train_losses = []
test_losses = []

for epoch in range(args.epochs):

    # Taining
    model.train()
    train_loss = 0
    for batch_id in tqdm(range(len(X_train) // args.batch_size)):
        optimizer.zero_grad()

        x = X_train[batch_id * args.batch_size:(batch_id + 1) * args.batch_size].to(device)
        y = Y_train[batch_id * args.batch_size:(batch_id + 1) * args.batch_size].to(device).view(-1, 1)
        
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
    ahi_predictions, ahi_targets = [], []
    with torch.no_grad():
        for batch_id in tqdm(range(len(X_test) // args.batch_size)):

            x = X_test[batch_id * args.batch_size:(batch_id + 1) * args.batch_size].to(device)
            y = Y_test[batch_id * args.batch_size:(batch_id + 1) * args.batch_size].to(device).view(-1, 1)

            # Forward pass
            y_hat = model(x)
            ahi_predictions.append(y_hat.cpu().view(-1).numpy())
            ahi_targets.append(y.cpu().view(-1).numpy())
            
            # Loss computation
            loss = criterion(y_hat, y)
            test_loss += loss.item()
    
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss / (len(X_train) // args.batch_size):.4f}, Test Loss: {test_loss / (len(X_test) // args.batch_size):.4f}')
    train_losses.append(train_loss / (len(X_train) // args.batch_size))
    test_losses.append(test_loss / (len(X_test) // args.batch_size))
    pd.DataFrame({
        'ahi_predictions': np.concat(ahi_predictions, axis=0),
        'ahi_targets': np.concat(ahi_targets, axis=0)
    }).to_csv(OUTPUT_DIR + f'/epoch_{epoch}_predictions.csv', index=False)

# Save model and losses
torch.save(model.state_dict(), OUTPUT_DIR + '/model_final.pth')
results = pd.DataFrame({
    'train_losses': train_losses,
    'test_losses': test_losses
})
results.to_csv(OUTPUT_DIR + '/losses.csv', index=False)

print(results)
