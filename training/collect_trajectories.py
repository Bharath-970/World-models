"""
Collect trajectories from MiniGrid using random policy.

Stores (observation, action, next_observation, reward, done) tuples.
"""
import gymnasium as gym
import minigrid
import numpy as np
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm


def collect_trajectories(env_name, num_episodes=1000, max_steps=100, save_path='data/trajectories.pkl'):
    """
    Collect trajectories using random policy.
    
    Args:
        env_name: Name of MiniGrid environment
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        save_path: Path to save trajectories
    
    Returns:
        trajectories: List of episodes, each containing list of transitions
    """
    print(f"Collecting {num_episodes} episodes from {env_name}...")
    
    # Create environment
    env = gym.make(env_name, render_mode='rgb_array')
    
    trajectories = []
    total_steps = 0
    
    for episode in tqdm(range(num_episodes), desc="Collecting episodes"):
        obs, info = env.reset()
        episode_data = []
        
        for step in range(max_steps):
            # Random action
            action = env.action_space.sample()
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Store transition
            episode_data.append({
                'obs': obs['image'],  # 7x7x3 image
                'action': action,
                'next_obs': next_obs['image'],
                'reward': reward,
                'done': terminated or truncated
            })
            
            obs = next_obs
            total_steps += 1
            
            if terminated or truncated:
                break
        
        trajectories.append(episode_data)
    
    env.close()
    
    # Save trajectories
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(trajectories, f)
    
    print(f"\n✓ Collected {len(trajectories)} episodes ({total_steps} total steps)")
    print(f"✓ Saved to {save_path}")
    
    # Print statistics
    episode_lengths = [len(ep) for ep in trajectories]
    episode_rewards = [sum(t['reward'] for t in ep) for ep in trajectories]
    
    print(f"\nStatistics:")
    print(f"  Episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Episode reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Min/Max length: {np.min(episode_lengths)}/{np.max(episode_lengths)}")
    print(f"  Min/Max reward: {np.min(episode_rewards):.2f}/{np.max(episode_rewards):.2f}")
    
    return trajectories


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect trajectories from MiniGrid')
    parser.add_argument('--env', type=str, default='MiniGrid-Empty-5x5-v0',
                        help='Environment name')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to collect')
    parser.add_argument('--max-steps', type=int, default=100,
                        help='Maximum steps per episode')
    parser.add_argument('--save-path', type=str, default='data/trajectories.pkl',
                        help='Path to save trajectories')
    
    args = parser.parse_args()
    
    collect_trajectories(
        env_name=args.env,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        save_path=args.save_path
    )
