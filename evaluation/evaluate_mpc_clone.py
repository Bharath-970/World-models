"""
Model-Predictive Control CEM planner using a lightweight env clone.

Uses simple_envs.EmptyGrid (a MiniGrid-Empty-5x5 re-implementation) so
we can clone state cheaply and run many rollouts in parallel.
"""
import argparse
import json
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm


def evaluate_random(env_factory, num_episodes, max_steps):
    rewards, successes, steps_to_goal = [], [], []
    for _ in tqdm(range(num_episodes), desc='random'):
        env = env_factory()
        ep_r = 0.0
        success = 0
        for t in range(max_steps):
            a = np.random.randint(7)
            r, term, trunc = env.step(a)
            ep_r += r
            if term and r > 0:
                success = 1
                steps_to_goal.append(t + 1)
                break
            if trunc:
                break
        rewards.append(ep_r)
        successes.append(success)
    return rewards, successes, steps_to_goal


def mpc_plan(env_state, horizon=8, num_samples=64, num_elites=8, num_iterations=3, discount=0.99, action_dim=7):
    discounts = (discount ** np.arange(horizon)).astype(np.float32)
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
            sim = env_state.clone()
            ep_r = 0.0
            for t in range(horizon):
                r, term, trunc = sim.step(int(sampled_actions[i, t]))
                ep_r += r * float(discounts[t])
                if term or trunc:
                    break
            rewards[i] = ep_r

        elite_idx = np.argsort(rewards)[-num_elites:]
        elite_actions = samples[elite_idx]
        action_mean = elite_actions.mean(axis=0)
        action_std = elite_actions.std(axis=0) + 1e-6

    return int(np.argmax(action_mean[0]))


def evaluate_mpc(env_factory, num_episodes, max_steps, horizon, num_samples, num_elites, num_iterations):
    rewards, successes, steps_to_goal = [], [], []
    for _ in tqdm(range(num_episodes), desc='mpc'):
        env = env_factory()
        ep_r = 0.0
        success = 0
        for t in range(max_steps):
            a = mpc_plan(env, horizon=horizon, num_samples=num_samples,
                         num_elites=num_elites, num_iterations=num_iterations)
            r, term, trunc = env.step(a)
            ep_r += r
            if term and r > 0:
                success = 1
                steps_to_goal.append(t + 1)
                break
            if term or trunc:
                break
        rewards.append(ep_r)
        successes.append(success)
    return rewards, successes, steps_to_goal


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='empty5x5', choices=['empty5x5'])
    p.add_argument('--horizon', type=int, default=10)
    p.add_argument('--num-samples', type=int, default=128)
    p.add_argument('--num-elites', type=int, default=16)
    p.add_argument('--num-iterations', type=int, default=4)
    p.add_argument('--num-episodes', type=int, default=30)
    p.add_argument('--max-steps', type=int, default=64)
    p.add_argument('--out', type=str, default='evaluation/mpc_clone_results.json')
    args = p.parse_args()

    from simple_envs import EmptyGrid
    factory = lambda: EmptyGrid()

    print(f"MPC planner (env clone) on {args.env}")
    print(f"  episodes={args.num_episodes}, max_steps={args.max_steps}")
    print(f"  horizon={args.horizon}, samples={args.num_samples}, elites={args.num_elites}, iters={args.num_iterations}")

    print("\n[1/2] Random baseline...")
    rs_r, ss_r, sg_r = evaluate_random(factory, args.num_episodes, args.max_steps)
    print(f"  reward={np.mean(rs_r):.3f} ± {np.std(rs_r):.3f}  success={np.mean(ss_r):.2%}")

    print("\n[2/2] MPC planner (clone rollouts)...")
    rs_m, ss_m, sg_m = evaluate_mpc(
        factory, args.num_episodes, args.max_steps,
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
