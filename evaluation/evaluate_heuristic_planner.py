"""
Eval the RSSM planner with HEURISTIC (pixel-distance) reward.

This is the controlled ablation: if the planner still fails with a
perfect reward signal (distance to goal in pixel space), the world
model is the bottleneck. If it works, the reward predictor is.
"""
import torch
import numpy as np
import argparse
import sys
import json
from pathlib import Path
import gymnasium as gym
import minigrid

sys.path.append(str(Path(__file__).parent.parent))

from planning.rssm_heuristic_planner import load_heuristic_planner


def evaluate_random(env_name, num_episodes, max_steps, seed=0):
    env = gym.make(env_name)
    successes, returns = [], []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_r = 0.0
        for t in range(max_steps):
            a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            total_r += r
            if term or trunc:
                break
        successes.append(bool(term and r > 0))
        returns.append(total_r)
    env.close()
    return {'success_rate': float(np.mean(successes)), 'mean_return': float(np.mean(returns))}


def evaluate_planner(env_name, planner, num_episodes, max_steps, device):
    env = gym.make(env_name)
    successes, returns, lengths = [], [], []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        total_r = 0.0
        for t in range(max_steps):
            try:
                actions = planner.plan(obs['image'], num_steps=1)
                a = actions[0]
            except Exception as e:
                a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            total_r += r
            if term or trunc:
                break
        successes.append(bool(term and r > 0))
        returns.append(total_r)
        lengths.append(t + 1)
    env.close()
    return {
        'success_rate': float(np.mean(successes)),
        'mean_return': float(np.mean(returns)),
        'mean_length': float(np.mean(lengths)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='MiniGrid-Empty-5x5-v0')
    p.add_argument('--num-episodes', type=int, default=20)
    p.add_argument('--max-steps', type=int, default=64)
    p.add_argument('--horizon', type=int, default=4)
    p.add_argument('--num-samples', type=int, default=128)
    p.add_argument('--num-elites', type=int, default=16)
    p.add_argument('--num-iters', type=int, default=4)
    p.add_argument('--sigma', type=float, default=0.1)
    p.add_argument('--rssm', type=str, default='checkpoints/rssm_rollout.pt')
    p.add_argument('--trajectories', type=str, default='data/trajectories.pkl')
    p.add_argument('--out', type=str, default='evaluation/rssm_heuristic_results.json')
    p.add_argument('--device', type=str, default=None)
    args = p.parse_args()

    print(f"=== Heuristic RSSM Planner Eval on {args.env} ===")
    print(f"  h={args.horizon}, samples={args.num_samples}, elites={args.num_elites}, iters={args.num_iters}, sigma={args.sigma}")

    print(f"\n[1/2] Random baseline...")
    rand_res = evaluate_random(args.env, args.num_episodes, args.max_steps)
    print(f"  Random: success={rand_res['success_rate']:.2%}, return={rand_res['mean_return']:.3f}")

    print(f"\n[2/2] Heuristic RSSM planner (pixel-space distance)...")
    planner, device = load_heuristic_planner(
        rssm_path=args.rssm,
        trajectories_path=args.trajectories,
        horizon=args.horizon,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        num_iterations=args.num_iters,
        sigma=args.sigma,
        device_name=args.device,
    )
    plan_res = evaluate_planner(args.env, planner, args.num_episodes, args.max_steps, device)
    print(f"  Planner: success={plan_res['success_rate']:.2%}, return={plan_res['mean_return']:.3f}, length={plan_res['mean_length']:.1f}")

    results = {
        'env': args.env,
        'config': vars(args),
        'random': rand_res,
        'planner': plan_res,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
