"""
Evaluate the RSSM CEM planner on MiniGrid envs.

For each episode:
    1. Reset env
    2. Encode obs → (h, z) using RSSM posterior
    3. CEM: plan action sequence of length horizon
    4. Execute first action
    5. Re-encode, replan (model-predictive control)
    6. Repeat until success, max_steps, or truncation

Compare to random baseline.
"""
import torch
import numpy as np
import argparse
import sys
from pathlib import Path
import json
import gymnasium as gym
import minigrid

sys.path.append(str(Path(__file__).parent.parent))

from planning.rssm_cem_planner import load_rssm_planner


def evaluate(env_name, planner, num_episodes, max_steps, device, seed=0):
    env = gym.make(env_name)
    successes = []
    returns = []
    ep_lengths = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_r = 0.0
        for t in range(max_steps):
            try:
                actions = planner.plan(obs['image'], num_steps=1)
                a = actions[0]
            except Exception as e:
                print(f"  planner error: {e}, falling back to random")
                a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            total_r += r
            if term or trunc:
                break
        success = bool(term and r > 0)
        successes.append(success)
        returns.append(total_r)
        ep_lengths.append(t + 1)
    env.close()
    return {
        'success_rate': float(np.mean(successes)),
        'mean_return': float(np.mean(returns)),
        'mean_length': float(np.mean(ep_lengths)),
        'n_episodes': num_episodes,
        'env': env_name,
    }


def evaluate_random(env_name, num_episodes, max_steps, seed=0):
    env = gym.make(env_name)
    successes = []
    returns = []
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
    return {
        'success_rate': float(np.mean(successes)),
        'mean_return': float(np.mean(returns)),
        'n_episodes': num_episodes,
        'env': env_name,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='MiniGrid-Empty-5x5-v0')
    p.add_argument('--num-episodes', type=int, default=20)
    p.add_argument('--max-steps', type=int, default=64)
    p.add_argument('--horizon', type=int, default=5)
    p.add_argument('--num-samples', type=int, default=64)
    p.add_argument('--num-elites', type=int, default=8)
    p.add_argument('--num-iters', type=int, default=3)
    p.add_argument('--rssm', type=str, default='checkpoints/rssm_rollout.pt')
    p.add_argument('--reward', type=str, default='checkpoints/reward_predictor_rssm_bin.pt')
    p.add_argument('--trajectories', type=str, default='data/trajectories.pkl')
    p.add_argument('--out', type=str, default='evaluation/rssm_planner_results.json')
    p.add_argument('--device', type=str, default=None)
    args = p.parse_args()

    print(f"=== RSSM CEM Planner Eval on {args.env} ===")
    print(f"  num_episodes={args.num_episodes}, max_steps={args.max_steps}")
    print(f"  horizon={args.horizon}, samples={args.num_samples}, elites={args.num_elites}, iters={args.num_iters}")

    print(f"\nLoading planner...")
    planner, device = load_rssm_planner(
        rssm_path=args.rssm,
        reward_predictor_path=args.reward,
        trajectories_path=args.trajectories,
        horizon=args.horizon,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        num_iterations=args.num_iters,
        device_name=args.device,
    )
    print(f"  device: {device}")

    print(f"\n[1/2] Random baseline...")
    rand_res = evaluate_random(args.env, args.num_episodes, args.max_steps)
    print(f"  Random: success={rand_res['success_rate']:.2%}, return={rand_res['mean_return']:.3f}")

    print(f"\n[2/2] RSSM planner...")
    plan_res = evaluate(args.env, planner, args.num_episodes, args.max_steps, device)
    print(f"  Planner: success={plan_res['success_rate']:.2%}, return={plan_res['mean_return']:.3f}, length={plan_res['mean_length']:.1f}")

    results = {
        'env': args.env,
        'config': {
            'horizon': args.horizon,
            'num_samples': args.num_samples,
            'num_elites': args.num_elites,
            'num_iters': args.num_iters,
            'max_steps': args.max_steps,
        },
        'random': rand_res,
        'planner': plan_res,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
