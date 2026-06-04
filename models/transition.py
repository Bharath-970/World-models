"""
Transition Model: (z_t, action_t) → z_{t+1}

Predicts next latent state given current latent state and action.
"""
import torch
import torch.nn as nn


class TransitionModel(nn.Module):
    def __init__(self, latent_dim=128, action_dim=7, hidden_dim=256):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Action embedding (discrete actions)
        self.action_embed = nn.Embedding(action_dim, 32)
        
        # Transition network
        self.fc1 = nn.Linear(latent_dim + 32, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)
        
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(latent_dim)
    
    def forward(self, z_t, action_t):
        """
        Args:
            z_t: (batch, latent_dim) - current latent state
            action_t: (batch,) - action taken
        
        Returns:
            z_next: (batch, latent_dim) - predicted next latent state
        """
        # Embed action
        a_embed = self.action_embed(action_t)  # (batch, 32)
        
        # Concatenate latent and action
        x = torch.cat([z_t, a_embed], dim=-1)  # (batch, latent_dim + 32)
        
        # Predict next latent state
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        z_next = self.fc3(x)
        z_next = self.layer_norm(z_next)
        
        return z_next
