'''
Train code for training the down sampling VAE
'''

from wearsed.dataset.WearSEDDataset import WearSEDDataset
from wearsed.models.vae.VAE_5s import VAE, vae_loss

from argparse import ArgumentParser
from tqdm import tqdm
import pandas as pd
import os

import torch.nn.functional as F
import torch

parser = ArgumentParser(description='Train code for training the down sampling VAE with a window of 5x')
parser.add_argument('--epochs', help='number of epochs', default=50, type=int)
parser.add_argument('--batch-size', help='batch size', default=256, type=int)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SIGNAL = 'Pleth'
FREQ = 256  # If any other signal is chosen with a different frequency, the code has to be adapted elsewhere too

def get_batch(recording, batch_size):
    signal = recording.psg[SIGNAL]
    max_time = len(signal) - 5*FREQ
    batch = []
    for timepoint in torch.randint(2*FREQ, max_time, (batch_size,)):
        batch.append(torch.Tensor(signal[timepoint.item():timepoint.item()+5*FREQ].values))
    return torch.stack(batch)

# Output folder
OUTPUT_DIR = f'wearsed/training/vae/output_5s/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset
full_dataset = WearSEDDataset(signals_to_read=[SIGNAL], return_recording=True)
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])

# Model and Optimizer
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training Loop
train_losses = []
test_losses = []

for epoch in range(args.epochs):

    # Taining
    model.train()
    train_loss = 0
    for recording in tqdm(train_dataset):
        optimizer.zero_grad()

        batch = get_batch(recording, batch_size=args.batch_size).to(device)
        
        # Forward pass
        recon, mu, log_var = model(batch)
        
        # Loss computation
        loss = vae_loss(recon, batch[:, 2*FREQ:3*FREQ], mu, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    # Testing
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for recording in tqdm(test_dataset):

            batch = get_batch(recording, batch_size=args.batch_size).to(device)

            # Forward pass
            recon, mu, log_var = model(batch)
            
            # Loss computation
            recon_loss = F.mse_loss(recon, batch[:, 2*FREQ:3*FREQ], reduction='sum')
            test_loss += recon_loss.item()
    
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_dataset):.4f}, Test MSE Loss: {test_loss / (len(test_dataset) * FREQ):.4f}')
    train_losses.append(train_loss / len(train_dataset))
    test_losses.append(test_loss / (len(test_dataset) * FREQ))

    if epoch % 10 == 0:
        torch.save(model.state_dict(), OUTPUT_DIR + f'/model_epoch_{epoch}.pth')

# Save model and losses
torch.save(model.state_dict(), OUTPUT_DIR + '/model_final.pth')
losses = pd.DataFrame({'train_losses': train_losses, 'test_losses': test_losses})
losses.to_csv(OUTPUT_DIR + '/losses.csv', index=False)

print(losses)

