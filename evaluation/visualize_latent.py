"""
Visualize latent space using t-SNE and PCA.

Shows structure in learned representations.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pickle
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.encoder import CNNEncoder
from models.transition import TransitionModel


def visualize_latent_space(
    model_path='checkpoints/world_model_final.pt',
    trajectories_path='data/trajectories.pkl',
    num_samples=500,
    save_dir='evaluation/results'
):
    """
    Visualize latent space with t-SNE and PCA.
    
    Args:
        model_path: Path to trained world model
        trajectories_path: Path to trajectories
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
    """
    print("Loading model...")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = CNNEncoder(input_shape=(7, 7, 3), latent_dim=128).to(device)
    transition = TransitionModel(latent_dim=128, action_dim=7).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    transition.load_state_dict(checkpoint['transition_state_dict'])
    
    encoder.eval()
    transition.eval()
    
    # Load trajectories
    print("Loading trajectories...")
    with open(trajectories_path, 'rb') as f:
        trajectories = pickle.load(f)
    
    # Collect latent vectors
    print(f"Encoding {num_samples} observations...")
    latents = []
    actions = []
    episode_ids = []
    
    for i, episode in enumerate(trajectories[:50]):
        for step in episode[:10]:
            obs = torch.FloatTensor(step['obs']).unsqueeze(0).to(device)
            action = step['action']
            
            with torch.no_grad():
                z = encoder(obs)
            
            latents.append(z.cpu().numpy().flatten())
            actions.append(action)
            episode_ids.append(i)
            
            if len(latents) >= num_samples:
                break
        
        if len(latents) >= num_samples:
            break
    
    latents = np.array(latents)
    actions = np.array(actions)
    episode_ids = np.array(episode_ids)
    
    print(f"Collected {len(latents)} latent vectors")
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # t-SNE visualization
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latents_2d = tsne.fit_transform(latents)
    
    # PCA visualization
    print("Computing PCA...")
    pca = PCA(n_components=2, random_state=42)
    latents_pca = pca.fit_transform(latents)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: t-SNE colored by action
    scatter = axes[0, 0].scatter(latents_2d[:, 0], latents_2d[:, 1], 
                                  c=actions, cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(scatter, ax=axes[0, 0], label='Action')
    axes[0, 0].set_xlabel('t-SNE Dimension 1')
    axes[0, 0].set_ylabel('t-SNE Dimension 2')
    axes[0, 0].set_title('Latent Space (t-SNE, Colored by Action)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: t-SNE colored by episode
    scatter = axes[0, 1].scatter(latents_2d[:, 0], latents_2d[:, 1],
                                  c=episode_ids, cmap='plasma', alpha=0.6, s=20)
    plt.colorbar(scatter, ax=axes[0, 1], label='Episode')
    axes[0, 1].set_xlabel('t-SNE Dimension 1')
    axes[0, 1].set_ylabel('t-SNE Dimension 2')
    axes[0, 1].set_title('Latent Space (t-SNE, Colored by Episode)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: PCA colored by action
    scatter = axes[1, 0].scatter(latents_pca[:, 0], latents_pca[:, 1],
                                  c=actions, cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(scatter, ax=axes[1, 0], label='Action')
    axes[1, 0].set_xlabel('PCA Component 1')
    axes[1, 0].set_ylabel('PCA Component 2')
    axes[1, 0].set_title('Latent Space (PCA, Colored by Action)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: PCA colored by episode
    scatter = axes[1, 1].scatter(latents_pca[:, 0], latents_pca[:, 1],
                                  c=episode_ids, cmap='plasma', alpha=0.6, s=20)
    plt.colorbar(scatter, ax=axes[1, 1], label='Episode')
    axes[1, 1].set_xlabel('PCA Component 1')
    axes[1, 1].set_ylabel('PCA Component 2')
    axes[1, 1].set_title('Latent Space (PCA, Colored by Episode)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'latent_space_visualization.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path / 'latent_space_visualization.png'}")
    
    # Additional analysis: latent statistics
    print("\nLatent Statistics:")
    print(f"  Mean: {np.mean(latents):.4f}")
    print(f"  Std: {np.std(latents):.4f}")
    print(f"  Min: {np.min(latents):.4f}")
    print(f"  Max: {np.max(latents):.4f}")
    print(f"  Norm (mean): {np.mean(np.linalg.norm(latents, axis=1)):.4f}")
    
    # Plot latent dimension distributions
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(32, latents.shape[1])):
        axes[i].hist(latents[:, i], bins=30, alpha=0.7, color='steelblue')
        axes[i].set_title(f'Dim {i}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Count')
    
    # Hide unused subplots
    for i in range(32, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Latent Dimension Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path / 'latent_distributions.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path / 'latent_distributions.png'}")
    
    plt.close('all')
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize latent space')
    parser.add_argument('--model', type=str, default='checkpoints/world_model_final.pt',
                        help='Path to trained model')
    parser.add_argument('--trajectories', type=str, default='data/trajectories.pkl',
                        help='Path to trajectories')
    parser.add_argument('--samples', type=int, default=500,
                        help='Number of samples to visualize')
    parser.add_argument('--save-dir', type=str, default='evaluation/results',
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    visualize_latent_space(
        model_path=args.model,
        trajectories_path=args.trajectories,
        num_samples=args.samples,
        save_dir=args.save_dir
    )
