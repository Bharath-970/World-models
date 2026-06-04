import gymnasium as gym
import minigrid
import numpy as np
import pickle
import time
import os
import random as rand
from tqdm import tqdm
from collections import defaultdict

# PPO only if available
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    HAVE_SB3 = True
except ImportError:
    HAVE_SB3 = False
    print("WARNING: stable-baselines3 not installed. PPO-policy collection disabled.")


class ImageObsWrapper(gym.ObservationWrapper):
    """Extract image array from MiniGrid's Dict observation."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space['image']

    def observation(self, observation):
        return observation['image']


def make_env(env_id, seed=0):
    def _init():
        env = gym.make(env_id, max_steps=100, render_mode=None)
        env = ImageObsWrapper(env)
        env.reset(seed=seed)
        return env
    return _init


def train_ppo_policy(env_id="MiniGrid-Empty-5x5-v0", total_timesteps=50000, seed=42):
    """Train a quick PPO policy and return it."""
    vec_env = DummyVecEnv([make_env(env_id, seed=seed)])
    vec_env = VecMonitor(vec_env)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=seed,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    return model


def collect_with_policy(env, policy, n_episodes, ppo_ratio=0.7, max_steps=100):
    """
    Collect trajectories using a mix of PPO policy and random actions.

    Args:
        ppo_ratio: fraction of steps where PPO action is used (vs random)
    """
    episodes = []
    for _ in tqdm(range(n_episodes), desc=f"Collecting ({ppo_ratio:.0%} PPO)"):
        obs, _ = env.reset()
        episode = []
        for _ in range(max_steps):
            # decide PPO vs random
            if rand.random() < ppo_ratio:
                action, _ = policy.predict(obs['image'], deterministic=False)
            else:
                action = env.action_space.sample()

            next_obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

            episode.append({
                'obs': obs['image'].copy(),
                'action': np.int64(action),
                'next_obs': next_obs['image'].copy(),
                'reward': reward,
                'done': done,
            })

            obs = next_obs
            if done:
                break

        episodes.append(episode)

    return episodes


def collect_random(env, n_episodes, max_steps=100):
    """Collect random-policy trajectories (same format as PPO collector)."""
    episodes = []
    for _ in tqdm(range(n_episodes), desc="Collecting (random)"):
        obs, _ = env.reset()
        episode = []
        for _ in range(max_steps):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

            episode.append({
                'obs': obs['image'].copy(),
                'action': np.int64(action),
                'next_obs': next_obs['image'].copy(),
                'reward': reward,
                'done': done,
            })

            obs = next_obs
            if done:
                break

        episodes.append(episode)

    return episodes


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='MiniGrid-Empty-5x5-v0')
    parser.add_argument('--n-episodes', type=int, default=10000,
                        help='Total episodes to collect')
    parser.add_argument('--ppo-ratio', type=float, default=0.7,
                        help='Fraction of PPO actions (rest random)')
    parser.add_argument('--ppo-timesteps', type=int, default=50000,
                        help='Timesteps to train PPO')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='data/trajectories_ppo.pkl')
    parser.add_argument('--no-ppo', action='store_true',
                        help='Skip PPO training, collect random only')
    args = parser.parse_args()

    print(f"Environment: {args.env}")
    print(f"Total episodes: {args.n_episodes}")
    print(f"PPO ratio: {args.ppo_ratio}")
    print(f"Output: {args.output}")

    if args.no_ppo or not HAVE_SB3:
        print("Collecting random-policy trajectories only")
        env = gym.make(args.env, max_steps=100, render_mode=None)
        episodes = collect_random(env, args.n_episodes)
        env.close()
    else:
        # Stage 1: Train PPO policy
        print(f"Training PPO for {args.ppo_timesteps} timesteps...")
        model = train_ppo_policy(args.env, args.ppo_timesteps, args.seed)

        # Stage 2: Collect with mixed policy
        env = gym.make(args.env, max_steps=100, render_mode=None)
        env.reset(seed=args.seed + 1)

        n_ppo = int(args.n_episodes * args.ppo_ratio)
        n_random = args.n_episodes - n_ppo

        ppo_episodes = collect_with_policy(
            env, model, n_ppo, ppo_ratio=1.0, max_steps=100
        )
        random_episodes = collect_random(env, n_random, max_steps=100)
        env.close()

        episodes = ppo_episodes + random_episodes
        rand.shuffle(episodes)

    # Stats
    rewards = [sum(t['reward'] for t in ep) for ep in episodes]
    lengths = [len(ep) for ep in episodes]
    success_rate = np.mean([r > 0 for r in rewards])
    total_transitions = sum(lengths)

    print(f"\nCollected {len(episodes)} episodes")
    print(f"Total transitions: {total_transitions}")
    print(f"Mean episode length: {np.mean(lengths):.1f}")
    print(f"Mean reward: {np.mean(rewards):.3f}")
    print(f"Success rate (reward > 0): {success_rate:.1%}")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(episodes, f)

    print(f"Saved to {args.output}")
    print("Done!")


if __name__ == '__main__':
    main()
