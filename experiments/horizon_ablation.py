"""
Horizon ablation for CEM planner.

For a fixed world model, vary CEM planning horizon and report:
- success rate
- mean reward
- mean steps to goal
"""
import argparse
import json
from pathlib import Path
import numpy as np
import sys
from tqdm import tqdm

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import minigrid

from planning.fast_cem_planner import load_fast_planner


def eval_planner(env_name, planner, num_episodes, max_steps, device):
    env = gym.make(env_name)
    rewards, successes, steps_to_goal = [], [], []
    for _ in tqdm(range(num_episodes), desc='eps'):
        obs, info = env.reset()
        ep_r = 0.0
        success = 0
        steps = 0
        for t in range(max_steps):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='MiniGrid-Empty-5x5-v0')
    p.add_argument('--checkpoints', type=str, default='checkpoints')
    p.add_argument('--horizons', type=int, nargs='+', default=[1, 2, 5, 10])
    p.add_argument('--num-samples', type=int, default=128)
    p.add_argument('--num-elites', type=int, default=10)
    p.add_argument('--num-iterations', type=int, default=4)
    p.add_argument('--num-episodes', type=int, default=30)
    p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--out', type=str, default='experiments/horizon_ablation.json')
    args = p.parse_args()

    out = {}
    for h in args.horizons:
        print(f"\n=== Horizon = {h} ===")
        planner, device = load_fast_planner(
            model_dir=args.checkpoints,
            latent_dim=128,
            horizon=h,
            num_samples=args.num_samples,
            num_elites=args.num_elites,
            num_iterations=args.num_iterations,
            device_name=args.device,
        )
        rs, ss, st = eval_planner(args.env, planner, args.num_episodes, args.max_steps, device)
        out[str(h)] = {
            'reward_mean': float(np.mean(rs)),
            'reward_std': float(np.std(rs)),
            'success_rate': float(np.mean(ss)),
            'mean_steps_to_goal': float(np.mean(st)) if st else None,
        }
        print(f"  reward={np.mean(rs):.3f} ± {np.std(rs):.3f}  success={np.mean(ss):.2%}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'env': args.env,
            'checkpoints': args.checkpoints,
            'num_episodes': args.num_episodes,
            'results': out,
        }, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
