"""
CNN Decoder: Latent Vector → Image

Decodes latent representations back to image space for visualization.
"""
import torch
import torch.nn as nn


class CNNDecoder(nn.Module):
    def __init__(self, latent_dim=128, output_shape=(7, 7, 3)):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_shape = output_shape
        
        # Start from 4x4 feature map
        self.fc = nn.Linear(latent_dim, 64 * 4 * 4)
        
        # Transpose conv layers: 4x4 → 5x5 → 6x6 → 7x7
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=1, padding=0)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=1, padding=0)
        self.deconv3 = nn.ConvTranspose2d(16, 3, kernel_size=2, stride=1, padding=0)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, z):
        """
        Args:
            z: (batch, latent_dim)
        
        Returns:
            x_recon: (batch, H, W, C)
        """
        # Project to feature map
        x = self.fc(z)
        x = x.view(x.size(0), 64, 4, 4)
        
        # Transpose conv layers
        x = self.relu(self.deconv1(x))  # 4x4 → 5x5
        x = self.relu(self.deconv2(x))  # 5x5 → 6x6
        x = self.sigmoid(self.deconv3(x))  # 6x6 → 7x7
        
        # Permute to (batch, H, W, C)
        x = x.permute(0, 2, 3, 1)
        
        return x
