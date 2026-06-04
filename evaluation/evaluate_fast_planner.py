"""
Evaluate planner vs random vs PPO using the fast vectorized CEM planner.
"""
import gymnasium as gym
import minigrid
import torch
import numpy as np
import argparse
from pathlib import Path
import json
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from planning.fast_cem_planner import load_fast_planner


def evaluate_random(env_name, num_episodes, max_steps=None):
    env = gym.make(env_name)
    rewards, successes, steps_to_goal = [], [], []
    for _ in tqdm(range(num_episodes), desc='random'):
        obs, info = env.reset()
        ep_r = 0.0
        success = 0
        steps = 0
        for t in range(max_steps or 10000):
            a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            ep_r += r
            steps += 1
            if term and r > 0:
                success = 1
                steps_to_goal.append(steps)
                break
            if trunc or term:
                break
        rewards.append(ep_r)
        successes.append(success)
    env.close()
    return rewards, successes, steps_to_goal


def evaluate_planner(env_name, planner, num_episodes, max_steps=None):
    env = gym.make(env_name)
    rewards, successes, steps_to_goal = [], [], []
    for _ in tqdm(range(num_episodes), desc='planner'):
        obs, info = env.reset()
        ep_r = 0.0
        success = 0
        steps = 0
        for t in range(max_steps or 10000):
            a = planner.plan(obs['image'], num_steps=1)[0]
            obs, r, term, trunc, info = env.step(a)
            ep_r += r
            steps += 1
            if term and r > 0:
                success = 1
                steps_to_goal.append(steps)
                break
            if trunc or term:
                break
        rewards.append(ep_r)
        successes.append(success)
    env.close()
    return rewards, successes, steps_to_goal


def evaluate_ppo(env_name, total_timesteps, num_episodes, max_steps=None):
    from stable_baselines3 import PPO
    env = gym.make(env_name)

    class FlatObsWrapper(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            img_shape = env.observation_space['image'].shape
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=img_shape, dtype=np.uint8)

        def observation(self, obs):
            return obs['image'].astype(np.uint8)

    env = FlatObsWrapper(env)
    model = PPO('MlpPolicy', env, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    rewards, successes, steps_to_goal = [], [], []
    for _ in tqdm(range(num_episodes), desc='ppo'):
        obs, info = env.reset()
        ep_r = 0.0
        success = 0
        steps = 0
        for t in range(max_steps or 10000):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            obs = obs.astype(np.uint8) if isinstance(obs, np.ndarray) and obs.dtype != np.uint8 else obs
            ep_r += r
            steps += 1
            if term and r > 0:
                success = 1
                steps_to_goal.append(steps)
                break
            if trunc or term:
                break
        rewards.append(ep_r)
        successes.append(success)
    env.close()
    return rewards, successes, steps_to_goal


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='MiniGrid-Empty-5x5-v0')
    p.add_argument('--checkpoints', type=str, default='checkpoints')
    p.add_argument('--latent-dim', type=int, default=128)
    p.add_argument('--horizon', type=int, default=5)
    p.add_argument('--num-samples', type=int, default=128)
    p.add_argument('--num-elites', type=int, default=10)
    p.add_argument('--num-iterations', type=int, default=4)
    p.add_argument('--num-episodes', type=int, default=30)
    p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--ppo-steps', type=int, default=20000)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--out', type=str, default='evaluation/planner_results.json')
    args = p.parse_args()

    print(f"Evaluating on {args.env} ({args.num_episodes} episodes, max {args.max_steps} steps/ep)")
    print(f"Loading fast planner from {args.checkpoints} ...")
    planner, device = load_fast_planner(
        model_dir=args.checkpoints,
        latent_dim=args.latent_dim,
        horizon=args.horizon,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        num_iterations=args.num_iterations,
        device_name=args.device,
    )
    print(f"Device: {device}")

    print("\n[1/3] Random policy...")
    rand_r, rand_s, rand_steps = evaluate_random(args.env, args.num_episodes, args.max_steps)
    print(f"  reward={np.mean(rand_r):.3f}  success={np.mean(rand_s):.2%}")

    print("\n[2/3] CEM planner (vectorized)...")
    plan_r, plan_s, plan_steps = evaluate_planner(args.env, planner, args.num_episodes, args.max_steps)
    print(f"  reward={np.mean(plan_r):.3f}  success={np.mean(plan_s):.2%}")

    print("\n[3/3] PPO baseline...")
    ppo_r, ppo_s, ppo_steps = evaluate_ppo(args.env, args.ppo_steps, args.num_episodes, args.max_steps)
    print(f"  reward={np.mean(ppo_r):.3f}  success={np.mean(ppo_s):.2%}")

    results = {
        'env': args.env,
        'num_episodes': args.num_episodes,
        'max_steps': args.max_steps,
        'random': {
            'reward_mean': float(np.mean(rand_r)),
            'reward_std': float(np.std(rand_r)),
            'success_rate': float(np.mean(rand_s)),
            'mean_steps_to_goal': float(np.mean(rand_steps)) if rand_steps else None,
        },
        'planner': {
            'reward_mean': float(np.mean(plan_r)),
            'reward_std': float(np.std(plan_r)),
            'success_rate': float(np.mean(plan_s)),
            'mean_steps_to_goal': float(np.mean(plan_steps)) if plan_steps else None,
        },
        'ppo': {
            'reward_mean': float(np.mean(ppo_r)),
            'reward_std': float(np.std(ppo_r)),
            'success_rate': float(np.mean(ppo_s)),
            'mean_steps_to_goal': float(np.mean(ppo_steps)) if ppo_steps else None,
            'training_steps': args.ppo_steps,
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("\n=== Summary ===")
    print(f"  Random:  reward={results['random']['reward_mean']:.3f} ± {results['random']['reward_std']:.3f}, success={results['random']['success_rate']:.2%}")
    print(f"  Planner: reward={results['planner']['reward_mean']:.3f} ± {results['planner']['reward_std']:.3f}, success={results['planner']['success_rate']:.2%}")
    print(f"  PPO:     reward={results['ppo']['reward_mean']:.3f} ± {results['ppo']['reward_std']:.3f}, success={results['ppo']['success_rate']:.2%}")


if __name__ == "__main__":
    main()
