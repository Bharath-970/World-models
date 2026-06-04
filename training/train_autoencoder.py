"""
Train autoencoder for visual validation.

Freezes encoder from world model and trains decoder to reconstruct images.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.encoder import CNNEncoder
from models.decoder import CNNDecoder


def train_autoencoder(
    trajectories_path='data/trajectories.pkl',
    encoder_path='checkpoints/world_model_best.pt',
    latent_dim=128,
    batch_size=64,
    num_epochs=30,
    lr=1e-3,
    save_dir='checkpoints',
    freeze_encoder=True
):
    """
    Train autoencoder for visual validation.
    
    Args:
        trajectories_path: Path to collected trajectories
        encoder_path: Path to trained encoder
        latent_dim: Dimension of latent space
        batch_size: Training batch size
        num_epochs: Number of training epochs
        lr: Learning rate
        save_dir: Directory to save checkpoints
        freeze_encoder: Whether to freeze encoder weights
    """
    # Load trajectories
    print(f"Loading trajectories from {trajectories_path}...")
    with open(trajectories_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    # Extract observations
    observations = []
    for episode in trajectories:
        for step in episode:
            observations.append(step['obs'])
    
    print(f"Total observations: {len(observations)}")
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Initialize models
    encoder = CNNEncoder(input_shape=(7, 7, 3), latent_dim=latent_dim).to(device)
    decoder = CNNDecoder(latent_dim=latent_dim, output_shape=(7, 7, 3)).to(device)
    
    # Load trained encoder
    print(f"Loading encoder from {encoder_path}...")
    checkpoint = torch.load(encoder_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    
    # Freeze encoder if requested
    if freeze_encoder:
        print("Freezing encoder weights...")
        for param in encoder.parameters():
            param.requires_grad = False
        encoder.eval()
    
    # Optimizer (only decoder if encoder frozen)
    if freeze_encoder:
        optimizer = optim.Adam(decoder.parameters(), lr=lr)
    else:
        params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = optim.Adam(params, lr=lr)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    print(f"\nTraining autoencoder for {num_epochs} epochs...")
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle observations
        np.random.shuffle(observations)
        
        pbar = tqdm(range(0, len(observations), batch_size), 
                   desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i in pbar:
            batch = observations[i:i+batch_size]
            obs_batch = torch.FloatTensor(batch).to(device)
            
            # Encode
            if freeze_encoder:
                with torch.no_grad():
                    z = encoder(obs_batch)
            else:
                z = encoder(obs_batch)
            
            # Decode
            obs_recon = decoder(z)
            
            # Compute loss
            loss = criterion(obs_recon, obs_batch)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
        
        avg_loss = epoch_loss / num_batches
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Save checkpoint if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path / 'autoencoder_best.pt')
    
    # Save final model
    save_path = Path(save_dir)
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'loss': best_loss,
    }, save_path / 'autoencoder_final.pt')
    
    print(f"\n✓ Training complete!")
    print(f"✓ Best loss: {best_loss:.6f}")
    print(f"✓ Models saved to {save_dir}/")
    
    return encoder, decoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train autoencoder')
    parser.add_argument('--trajectories', type=str, default='data/trajectories.pkl',
                        help='Path to trajectories')
    parser.add_argument('--encoder', type=str, default='checkpoints/world_model_best.pt',
                        help='Path to trained encoder')
    parser.add_argument('--latent-dim', type=int, default=128,
                        help='Latent dimension')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--no-freeze', action='store_true',
                        help='Do not freeze encoder')
    
    args = parser.parse_args()
    
    train_autoencoder(
        trajectories_path=args.trajectories,
        encoder_path=args.encoder,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir,
        freeze_encoder=not args.no_freeze
    )
