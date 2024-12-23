'''
Variational Autoencoder (VAE) for down sampling high frequency signals like `Pleth`
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim=8):
        super(VAE, self).__init__()
        
        # Encoder
        self.enc_conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.enc_conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.enc_fc    = nn.Linear(32*64, 256)
        self.mu_layer = nn.Linear(256, latent_dim)
        self.log_var_layer = nn.Linear(256, latent_dim)
        
        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, 256)
        self.dec_fc2 = nn.Linear(256, 2048)
        self.dec_conv1 = nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.tanh = nn.Tanh()

    def encode(self, x):                  # [bs,     256]
        x = x.unsqueeze(1)                # [bs,  1, 256]
        x = F.relu(self.enc_conv1(x))     # [bs, 16, 128]
        x = F.relu(self.enc_conv2(x))     # [bs, 32,  64]
        x = x.view(x.shape[0], -1)        # [bs,    2048]
        x = F.relu(self.enc_fc(x))        # [bs,     256]
        mu, log_var = self.mu_layer(x), self.log_var_layer(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)    # Standard deviation
        eps = torch.randn_like(std)       # Random noise
        return mu + eps * std             # Reparameterized sample
    
    def decode(self, z):
        x = F.relu(self.dec_fc1(z))       # [bs,     256]
        x = F.relu(self.dec_fc2(x))       # [bs,    2048]
        x = x.view(x.shape[0], 32, 64)    # [bs, 32,  64]
        x = F.relu(self.dec_conv1(x))     # [bs, 16, 128]
        x = self.tanh(self.dec_conv2(x))  # [bs,  1, 256]
        return x.squeeze(1)               # [bs,     256]

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z) * 6  # Scaling the tanh (range [-1, 1]) to [-6, 6]
        return recon, mu, log_var

def vae_loss(recon, x, mu, log_var):
    recon_loss = F.mse_loss(recon, x, reduction='sum')                   # Reconstruction Loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # KL-Divergence Loss
    return recon_loss + kl_loss
