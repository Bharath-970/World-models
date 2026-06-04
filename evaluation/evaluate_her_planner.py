"""
Evaluate goal-conditioned CEM planner (HER-trained reward) vs random.
"""
import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import gymnasium as gym
import minigrid

sys.path.append(str(Path(__file__).parent.parent))

from planning.her_cem_planner import load_goal_conditioned_planner


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


def evaluate_planner(env_name, planner, num_episodes, max_steps):
    env = gym.make(env_name)
    rewards, successes, steps_to_goal = [], [], []
    for _ in tqdm(range(num_episodes), desc='planner'):
        obs, info = env.reset()
        ep_r = 0.0
        success = 0
        for t in range(max_steps):
            a = planner.plan(obs['image'], num_steps=1)[0]
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
    p.add_argument('--checkpoints', type=str, default='checkpoints')
    p.add_argument('--trajectories', type=str, default='data/trajectories.pkl')
    p.add_argument('--her-predictor', type=str, default='checkpoints/reward_predictor_her.pt')
    p.add_argument('--latent-dim', type=int, default=128)
    p.add_argument('--horizon', type=int, default=5)
    p.add_argument('--num-episodes', type=int, default=30)
    p.add_argument('--max-steps', type=int, default=256)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--out', type=str, default='evaluation/her_planner_results.json')
    args = p.parse_args()

    print(f"Device: {args.device or 'auto'}")
    print(f"Evaluating HER planner on {args.env}")
    print(f"  episodes={args.num_episodes}, max_steps={args.max_steps}, horizon={args.horizon}")

    print("\n[1/2] Random policy...")
    rs_r, ss_r, sg_r = evaluate_random(args.env, args.num_episodes, args.max_steps)
    print(f"  reward={np.mean(rs_r):.3f} ± {np.std(rs_r):.3f}  success={np.mean(ss_r):.2%}")

    print("\n[2/2] HER planner...")
    planner, device = load_goal_conditioned_planner(
        model_dir=args.checkpoints,
        trajectories_path=args.trajectories,
        latent_dim=args.latent_dim,
        horizon=args.horizon,
        device_name=args.device,
        her_predictor_path=args.her_predictor,
    )
    rs_p, ss_p, sg_p = evaluate_planner(args.env, planner, args.num_episodes, args.max_steps)
    print(f"  reward={np.mean(rs_p):.3f} ± {np.std(rs_p):.3f}  success={np.mean(ss_p):.2%}")

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
        'her_planner': {
            'reward_mean': float(np.mean(rs_p)),
            'reward_std': float(np.std(rs_p)),
            'success_rate': float(np.mean(ss_p)),
            'mean_steps_to_goal': float(np.mean(sg_p)) if sg_p else None,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.out}")
    print(f"\n=== Summary ===")
    print(f"  Random:  reward={out['random']['reward_mean']:.3f} ± {out['random']['reward_std']:.3f}, success={out['random']['success_rate']:.2%}")
    print(f"  HER:     reward={out['her_planner']['reward_mean']:.3f} ± {out['her_planner']['reward_std']:.3f}, success={out['her_planner']['success_rate']:.2%}")


if __name__ == "__main__":
    main()
