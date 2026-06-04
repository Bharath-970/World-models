"""
Train world model to predict next latent state.

Trains encoder + transition model with MSE loss.
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
from models.transition import TransitionModel


def select_device(device_name=None):
    """
    Select the best available device for training.

    Args:
        device_name: Optional explicit device override.
    """
    if device_name:
        return torch.device(device_name)

    if torch.cuda.is_available():
        return torch.device('cuda')

    if torch.backends.mps.is_available():
        return torch.device('mps')

    return torch.device('cpu')


def load_training_state(checkpoint_path, encoder, transition, optimizer, device):
    """
    Restore model and optimizer state from a checkpoint.

    Returns:
        Tuple of (start_epoch, best_loss)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    transition.load_state_dict(checkpoint['transition_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint.get('epoch', 0)
    best_loss = checkpoint.get('loss', float('inf'))

    print(f"Resumed from {checkpoint_path} at epoch {start_epoch} (best loss: {best_loss:.6f})")
    return start_epoch, best_loss


def train_world_model(
    trajectories_path='data/trajectories.pkl',
    latent_dim=128,
    batch_size=64,
    num_epochs=50,
    lr=1e-3,
    save_dir='checkpoints',
    use_wandb=False,
    resume_from=None,
    device_name=None,
):
    """
    Train world model (encoder + transition model).
    
    Args:
        trajectories_path: Path to collected trajectories
        latent_dim: Dimension of latent space
        batch_size: Training batch size
        num_epochs: Number of training epochs
        lr: Learning rate
        save_dir: Directory to save checkpoints
        use_wandb: Whether to log to Weights & Biases
    """
    # Setup wandb
    if use_wandb:
        import wandb
        wandb.init(project='world-models', name='mvp-training', config={
            'latent_dim': latent_dim,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'lr': lr
        })
    
    # Load trajectories
    print(f"Loading trajectories from {trajectories_path}...")
    with open(trajectories_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    # Flatten trajectories into transitions
    transitions = []
    for episode in trajectories:
        for i in range(len(episode) - 1):
            transitions.append({
                'obs': episode[i]['obs'],
                'action': episode[i]['action'],
                'next_obs': episode[i]['next_obs']
            })
    
    print(f"Total transitions: {len(transitions)}")
    
    # Setup device
    device = select_device(device_name)
    print(f"Using device: {device}")
    
    # Initialize models
    encoder = CNNEncoder(input_shape=(7, 7, 3), latent_dim=latent_dim).to(device)
    transition = TransitionModel(latent_dim=latent_dim, action_dim=7).to(device)
    
    # Optimizer
    params = list(encoder.parameters()) + list(transition.parameters())
    optimizer = optim.Adam(params, lr=lr)
    
    # Loss function
    criterion = nn.MSELoss()

    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    best_loss = float('inf')
    start_epoch = 0

    if resume_from:
        start_epoch, best_loss = load_training_state(
            resume_from, encoder, transition, optimizer, device
        )

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle transitions
        np.random.shuffle(transitions)
        
        pbar = tqdm(range(0, len(transitions), batch_size), 
                   desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i in pbar:
            batch = transitions[i:i+batch_size]
            
            # Prepare batch
            obs_batch = torch.FloatTensor([t['obs'] for t in batch]).to(device)
            action_batch = torch.LongTensor([t['action'] for t in batch]).to(device)
            next_obs_batch = torch.FloatTensor([t['next_obs'] for t in batch]).to(device)
            
            # Encode current and next observations
            z_t = encoder(obs_batch)
            z_next_true = encoder(next_obs_batch)
            
            # Predict next latent state
            z_next_pred = transition(z_t, action_batch)
            
            # Compute loss
            loss = criterion(z_next_pred, z_next_true)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
        
        avg_loss = epoch_loss / num_batches
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
        
        # Log to wandb
        if use_wandb:
            import wandb
            wandb.log({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'lr': lr
            })
        
        # Save checkpoint if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'transition_state_dict': transition.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path / 'world_model_best.pt')
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_path = Path(save_dir)
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'transition_state_dict': transition.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path / f'checkpoint_epoch_{epoch+1}.pt')
    
    # Save final model
    save_path = Path(save_dir)
    torch.save({
        'epoch': num_epochs,
        'encoder_state_dict': encoder.state_dict(),
        'transition_state_dict': transition.state_dict(),
        'loss': best_loss,
    }, save_path / 'world_model_final.pt')
    
    print(f"\n✓ Training complete!")
    print(f"✓ Best loss: {best_loss:.6f}")
    print(f"✓ Models saved to {save_dir}/")
    
    if use_wandb:
        import wandb
        wandb.finish()
    
    return encoder, transition


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train world model')
    parser.add_argument('--trajectories', type=str, default='data/trajectories.pkl',
                        help='Path to trajectories')
    parser.add_argument('--latent-dim', type=int, default=128,
                        help='Latent dimension')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--wandb', action='store_true',
                        help='Log to Weights & Biases')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Optional checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                        help='Optional device override (cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    train_world_model(
        trajectories_path=args.trajectories,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir,
        use_wandb=args.wandb,
        resume_from=args.resume_from,
        device_name=args.device,
    )
