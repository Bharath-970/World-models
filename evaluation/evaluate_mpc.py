"""
Model-Predictive Control CEM planner using the real env for rollouts.

The world model is replaced by the actual env: for each candidate action
sequence, we save the env state via pickle, run the sequence, and sum the
actual rewards. This sidesteps the world-model collapse entirely.

With N samples × horizon H, that's N*H env steps per planner call.
"""
import argparse
import json
import sys
import copy
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np
import gymnasium as gym
import minigrid


def evaluate_random(env_name, num_episodes, max_steps):
    env = gym.make(env_name)
    rewards, successes, steps_to_goal = [], [], []
    for _ in tqdm(range(num_episodes), desc='random'):
        obs, info = env.reset()
        ep_r = 0.0
        success = 0
        for t in range(max_steps):
            a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            ep_r += r
            if term and r > 0:
                success = 1
                steps_to_goal.append(t + 1)
                break
            if trunc:
                break
        rewards.append(ep_r)
        successes.append(success)
    env.close()
    return rewards, successes, steps_to_goal


def save_state(env):
    return copy.deepcopy(env.unwrapped.__getstate__())


def restore_state(env, state):
    env.unwrapped.__setstate__(state)


def mpc_plan(env, horizon=5, num_samples=64, num_elites=8, num_iterations=3, discount=0.99):
    discounts = (discount ** np.arange(horizon)).astype(np.float32)
    action_dim = env.action_space.n

    action_mean = np.full((horizon, action_dim), 1.0 / action_dim, dtype=np.float32)
    action_std = np.ones((horizon, action_dim), dtype=np.float32) * 0.5

    for _ in range(num_iterations):
        samples = np.random.normal(
            action_mean, action_std,
            size=(num_samples, horizon, action_dim),
        ).astype(np.float32)
        samples = np.maximum(samples, 0.0)
        samples_sum = samples.sum(axis=-1, keepdims=True) + 1e-8
        samples = samples / samples_sum
        sampled_actions = np.argmax(samples, axis=-1).astype(np.int64)

        rewards = np.zeros(num_samples, dtype=np.float32)
        for i in range(num_samples):
            saved = save_state(env)
            ep_r = 0.0
            for t in range(horizon):
                obs, r, term, trunc, info = env.step(int(sampled_actions[i, t]))
                ep_r += r * float(discounts[t])
                if term or trunc:
                    break
            rewards[i] = ep_r
            restore_state(env, saved)

        elite_idx = np.argsort(rewards)[-num_elites:]
        elite_actions = samples[elite_idx]
        action_mean = elite_actions.mean(axis=0)
        action_std = elite_actions.std(axis=0) + 1e-6

    return int(np.argmax(action_mean[0]))


def evaluate_mpc(env_name, num_episodes, max_steps, horizon, num_samples, num_elites, num_iterations):
    env = gym.make(env_name)
    rewards, successes, steps_to_goal = [], [], []
    for _ in tqdm(range(num_episodes), desc='mpc'):
        obs, info = env.reset()
        ep_r = 0.0
        success = 0
        for t in range(max_steps):
            a = mpc_plan(env, horizon=horizon, num_samples=num_samples,
                         num_elites=num_elites, num_iterations=num_iterations)
            obs, r, term, trunc, info = env.step(a)
            ep_r += r
            if term and r > 0:
                success = 1
                steps_to_goal.append(t + 1)
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
    p.add_argument('--horizon', type=int, default=8)
    p.add_argument('--num-samples', type=int, default=64)
    p.add_argument('--num-elites', type=int, default=8)
    p.add_argument('--num-iterations', type=int, default=3)
    p.add_argument('--num-episodes', type=int, default=30)
    p.add_argument('--max-steps', type=int, default=256)
    p.add_argument('--out', type=str, default='evaluation/mpc_results.json')
    args = p.parse_args()

    print(f"MPC planner on {args.env}")
    print(f"  episodes={args.num_episodes}, max_steps={args.max_steps}")
    print(f"  horizon={args.horizon}, samples={args.num_samples}, elites={args.num_elites}, iters={args.num_iterations}")

    print("\n[1/2] Random baseline...")
    rs_r, ss_r, sg_r = evaluate_random(args.env, args.num_episodes, args.max_steps)
    print(f"  reward={np.mean(rs_r):.3f} ± {np.std(rs_r):.3f}  success={np.mean(ss_r):.2%}")

    print("\n[2/2] MPC planner (env rollouts)...")
    rs_m, ss_m, sg_m = evaluate_mpc(
        args.env, args.num_episodes, args.max_steps,
        args.horizon, args.num_samples, args.num_elites, args.num_iterations,
    )
    print(f"  reward={np.mean(rs_m):.3f} ± {np.std(rs_m):.3f}  success={np.mean(ss_m):.2%}")

    out = {
        'env': args.env,
        'num_episodes': args.num_episodes,
        'max_steps': args.max_steps,
        'horizon': args.horizon,
        'random': {
            'reward_mean': float(np.mean(rs_r)),
            'reward_std': float(np.std(rs_r)),
            'success_rate': float(np.mean(ss_r)),
            'mean_steps_to_goal': float(np.mean(sg_r)) if sg_r else None,
        },
        'mpc_planner': {
            'reward_mean': float(np.mean(rs_m)),
            'reward_std': float(np.std(rs_m)),
            'success_rate': float(np.mean(ss_m)),
            'mean_steps_to_goal': float(np.mean(sg_m)) if sg_m else None,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.out}")
    print(f"\n=== Summary ===")
    print(f"  Random: reward={out['random']['reward_mean']:.3f}, success={out['random']['success_rate']:.2%}")
    print(f"  MPC:    reward={out['mpc_planner']['reward_mean']:.3f}, success={out['mpc_planner']['success_rate']:.2%}")


if __name__ == "__main__":
    main()

