"""
Visualize image reconstructions from autoencoder.

Shows original vs reconstructed images side-by-side.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.encoder import CNNEncoder
from models.decoder import CNNDecoder


def visualize_reconstructions(
    autoencoder_path='checkpoints/autoencoder_best.pt',
    trajectories_path='data/trajectories.pkl',
    num_examples=10,
    save_dir='evaluation/results'
):
    """
    Visualize image reconstructions.
    
    Args:
        autoencoder_path: Path to trained autoencoder
        trajectories_path: Path to trajectories
        num_examples: Number of examples to show
        save_dir: Directory to save visualizations
    """
    print("Loading model...")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = CNNEncoder(input_shape=(7, 7, 3), latent_dim=128).to(device)
    decoder = CNNDecoder(latent_dim=128, output_shape=(7, 7, 3)).to(device)
    
    checkpoint = torch.load(autoencoder_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    encoder.eval()
    decoder.eval()
    
    # Load trajectories
    print("Loading trajectories...")
    with open(trajectories_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    # Pick random examples
    print(f"Selecting {num_examples} random examples...")
    examples = []
    for _ in range(num_examples):
        episode = trajectories[np.random.randint(0, len(trajectories))]
        step = episode[np.random.randint(0, len(episode))]
        examples.append(step['obs'])
    
    # Create figure
    fig, axes = plt.subplots(num_examples, 2, figsize=(8, 3*num_examples))
    
    print("Generating reconstructions...")
    for i, obs in enumerate(examples):
        # Original
        axes[i, 0].imshow(obs)
        axes[i, 0].set_title('Original', fontsize=14, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Reconstruction
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        with torch.no_grad():
            z = encoder(obs_tensor)
            obs_recon = decoder(z)
        
        obs_recon_np = obs_recon.cpu().numpy().squeeze()
        axes[i, 1].imshow(obs_recon_np)
        axes[i, 1].set_title('Reconstruction', fontsize=14, fontweight='bold')
        axes[i, 1].axis('off')
        
        # Compute reconstruction error
        mse = np.mean((obs - obs_recon_np) ** 2)
        axes[i, 0].text(0, -1, f'MSE: {mse:.4f}', fontsize=10, ha='left')
    
    plt.suptitle('Autoencoder Reconstructions', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / 'reconstruction_visualization.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path / 'reconstruction_visualization.png'}")
    
    # Additional analysis: reconstruction quality statistics
    print("\nComputing reconstruction statistics...")
    all_mse = []
    
    for episode in trajectories[:50]:  # Sample 50 episodes
        for step in episode[:10]:  # First 10 steps
            obs = step['obs']
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                z = encoder(obs_tensor)
                obs_recon = decoder(z)
            
            obs_recon_np = obs_recon.cpu().numpy().squeeze()
            mse = np.mean((obs - obs_recon_np) ** 2)
            all_mse.append(mse)
    
    print(f"\nReconstruction Statistics (n={len(all_mse)}):")
    print(f"  Mean MSE: {np.mean(all_mse):.6f}")
    print(f"  Std MSE: {np.std(all_mse):.6f}")
    print(f"  Min MSE: {np.min(all_mse):.6f}")
    print(f"  Max MSE: {np.max(all_mse):.6f}")
    
    # Plot MSE distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(all_mse, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(all_mse), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_mse):.4f}')
    ax.set_xlabel('Mean Squared Error', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Reconstruction Error Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'reconstruction_error_distribution.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path / 'reconstruction_error_distribution.png'}")
    
    plt.close('all')
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize reconstructions')
    parser.add_argument('--model', type=str, default='checkpoints/autoencoder_best.pt',
                        help='Path to trained autoencoder')
    parser.add_argument('--trajectories', type=str, default='data/trajectories.pkl',
                        help='Path to trajectories')
    parser.add_argument('--examples', type=int, default=10,
                        help='Number of examples to show')
    parser.add_argument('--save-dir', type=str, default='evaluation/results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    visualize_reconstructions(
        autoencoder_path=args.model,
        trajectories_path=args.trajectories,
        num_examples=args.examples,
        save_dir=args.save_dir
    )
