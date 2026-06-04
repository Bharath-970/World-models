"""
Interactive test of the world model.

Shows real observations vs predicted observations side-by-side.
"""
import gymnasium as gym
import minigrid
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from models.encoder import CNNEncoder
from models.transition import TransitionModel
from models.decoder import CNNDecoder


def interactive_test(
    world_model_path='checkpoints/world_model_best.pt',
    autoencoder_path='checkpoints/autoencoder_best.pt',
    env_name='MiniGrid-Empty-5x5-v0',
    num_steps=10
):
    """
    Interactive test showing real vs predicted observations.
    
    Args:
        world_model_path: Path to trained world model
        autoencoder_path: Path to trained autoencoder
        env_name: Environment name
        num_steps: Number of steps to test
    """
    print("="*60)
    print("World Model Interactive Test")
    print("="*60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    print("\nLoading models...")
    encoder = CNNEncoder(input_shape=(7, 7, 3), latent_dim=128).to(device)
    transition = TransitionModel(latent_dim=128, action_dim=7).to(device)
    decoder = CNNDecoder(latent_dim=128, output_shape=(7, 7, 3)).to(device)
    
    # Load world model
    checkpoint = torch.load(world_model_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    transition.load_state_dict(checkpoint['transition_state_dict'])
    
    # Load decoder
    ae_checkpoint = torch.load(autoencoder_path, map_location=device)
    decoder.load_state_dict(ae_checkpoint['decoder_state_dict'])
    
    encoder.eval()
    transition.eval()
    decoder.eval()
    
    print("✓ Models loaded")
    
    # Create environment
    print(f"\nCreating environment: {env_name}")
    env = gym.make(env_name, render_mode='rgb_array')
    obs, info = env.reset()
    
    print("✓ Environment ready")
    
    # Collect real trajectory
    print(f"\nCollecting {num_steps} steps with random policy...")
    real_observations = []
    actions_taken = []
    
    current_obs = obs['image']
    real_observations.append(current_obs)
    
    for step in range(num_steps):
        action = env.action_space.sample()
        actions_taken.append(action)
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        current_obs = next_obs['image']
        real_observations.append(current_obs)
        
        if terminated or truncated:
            print(f"  Episode ended at step {step+1}")
            break
    
    env.close()
    
    print(f"✓ Collected {len(real_observations)} observations")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predicted_observations = [real_observations[0]]  # Start from same initial state
    
    # Encode initial observation
    with torch.no_grad():
        z_current = encoder(torch.FloatTensor(real_observations[0]).unsqueeze(0).to(device))
    
    # Predict future states
    for step in range(len(actions_taken)):
        action = actions_taken[step]
        action_tensor = torch.LongTensor([action]).to(device)
        
        with torch.no_grad():
            z_next = transition(z_current, action_tensor)
            obs_recon = decoder(z_next)
        
        predicted_observations.append(obs_recon.cpu().numpy().squeeze())
        z_current = z_next
    
    print(f"✓ Generated {len(predicted_observations)} predictions")
    
    # Create visualization
    print("\nCreating visualization...")
    num_to_show = min(6, len(real_observations))
    
    fig, axes = plt.subplots(num_to_show, 3, figsize=(15, 4*num_to_show))
    
    for i in range(num_to_show):
        # Real observation
        axes[i, 0].imshow(real_observations[i])
        axes[i, 0].set_title(f'Real (Step {i})', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        # Predicted observation
        if i < len(predicted_observations):
            axes[i, 1].imshow(predicted_observations[i])
            axes[i, 1].set_title(f'Predicted (Step {i})', fontsize=12, fontweight='bold')
            axes[i, 1].axis('off')
            
            # Compute error
            mse = np.mean((real_observations[i] - predicted_observations[i]) ** 2)
            axes[i, 1].text(0, -1, f'MSE: {mse:.4f}', fontsize=10, ha='left')
        
        # Action taken
        if i < len(actions_taken):
            action_names = ['left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']
            action_text = f"Action: {action_names[actions_taken[i]]}"
            axes[i, 2].text(0.5, 0.5, action_text, fontsize=14, ha='center', va='center',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[i, 2].axis('off')
    
    plt.suptitle('World Model Predictions vs Reality', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = Path('evaluation/results')
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / 'interactive_test.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path / 'interactive_test.png'}")
    
    # Compute overall statistics
    print("\n" + "="*60)
    print("Prediction Statistics")
    print("="*60)
    
    errors = []
    for i in range(min(len(real_observations), len(predicted_observations))):
        mse = np.mean((real_observations[i] - predicted_observations[i]) ** 2)
        errors.append(mse)
    
    print(f"Steps predicted: {len(errors)}")
    print(f"Mean MSE: {np.mean(errors):.4f}")
    print(f"Std MSE: {np.std(errors):.4f}")
    print(f"Min MSE: {np.min(errors):.4f}")
    print(f"Max MSE: {np.max(errors):.4f}")
    
    # Plot error over time
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(errors)), errors, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('Prediction Error Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'prediction_error_over_time.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path / 'prediction_error_over_time.png'}")
    
    plt.close('all')
    
    print("\n" + "="*60)
    print("✓ Interactive test complete!")
    print("="*60)
    print("\nResults:")
    print("  - interactive_test.png: Side-by-side comparison")
    print("  - prediction_error_over_time.png: Error accumulation")
    print("\nOpen the images to see how well the model predicts!")


if __name__ == "__main__":
    interactive_test()
