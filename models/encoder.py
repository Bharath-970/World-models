"""
CNN Encoder: Image → Latent Vector

Encodes MiniGrid observations (7x7x3) into compact latent representations.
"""
import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, input_shape=(7, 7, 3), latent_dim=128):
        super().__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        # MiniGrid observations are 7x7x3
        # Conv layers: 7x7 → 6x6 → 5x5 → 4x4
        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0)
        
        # Calculate flattened size: 64 * 4 * 4 = 1024
        self.flat_size = 64 * 4 * 4
        
        # Fully connected layer to latent space
        self.fc = nn.Linear(self.flat_size, latent_dim)
        
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(latent_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch, H, W, C) or (batch, C, H, W)

        Returns:
            z: (batch, latent_dim)
        """
        # Ensure input is (batch, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(0)

        if x.shape[-1] == 3 and x.shape[1] != 3:
            # Input is (batch, H, W, C), convert to (batch, C, H, W)
            x = x.permute(0, 3, 1, 2).contiguous()

        # Normalize to [0, 1] if input is uint8
        if x.dtype == torch.uint8:
            x = x.float() / 8.0
        elif x.max() > 1.5:
            x = x / 8.0
        
        # Conv layers
        x = self.relu(self.conv1(x))  # 7x7 → 6x6
        x = self.relu(self.conv2(x))  # 6x6 → 5x5
        x = self.relu(self.conv3(x))  # 5x5 → 4x4
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Project to latent space
        z = self.fc(x)
        z = self.layer_norm(z)
        
        return z
