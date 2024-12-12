'''
Variational Autoencoder (VAE) for down sampling high frequency signals like `Pleth`
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=256, latent_dim=8):
        super(VAE, self).__init__()

        self.hidden_dim_1 = input_dim // 2
        self.hidden_dim_2 = input_dim // 4
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim_1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_1, self.hidden_dim_2),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(self.hidden_dim_2, latent_dim)
        self.log_var_layer = nn.Linear(self.hidden_dim_2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dim_2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_2, self.hidden_dim_1),
            nn.ReLU(),
            nn.Linear(self.hidden_dim_1, input_dim),
            nn.Tanh()
        )
    
    def encode(self, x):
        hidden = self.encoder(x)
        mu, log_var = self.mu_layer(hidden), self.log_var_layer(hidden)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # Standard deviation
        eps = torch.randn_like(std)     # Random noise
        return mu + eps * std           # Reparameterized sample
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z) * 6  # Scaling the tanh (range [-1, 1]) to [-6, 6]
        return recon, mu, log_var

def vae_loss(recon, x, mu, log_var):
    recon_loss = F.mse_loss(recon, x, reduction='sum')                 # Reconstruction Loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # KL-Divergence Loss
    return recon_loss + kl_loss
