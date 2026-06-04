"""
Transition Model: (z_t, action_t) → z_{t+1}

Residual form: z_{t+1} = z_t + Δ(z_t, a)

The residual parameterization prevents the trivial "predict z_t" collapse
that happens with LayerNorm + MSE on random-policy data. The model can
still output Δ ≈ 0 if the data supports it, but is forced to add
information on top of z_t rather than absorb z_t through a normalization
layer.
"""
import torch
import torch.nn as nn


class TransitionModel(nn.Module):
    def __init__(self, latent_dim=128, action_dim=7, hidden_dim=256, residual_scale=0.1):
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.residual_scale = residual_scale

        self.action_embed = nn.Embedding(action_dim, 32)

        self.fc1 = nn.Linear(latent_dim + 32, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)

        self.relu = nn.ReLU()

    def forward(self, z_t, action_t):
        a_embed = self.action_embed(action_t)
        x = torch.cat([z_t, a_embed], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        delta = self.fc3(x)
        z_next = z_t + self.residual_scale * delta
        return z_next
