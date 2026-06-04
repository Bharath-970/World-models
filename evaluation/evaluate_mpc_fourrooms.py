"""
Model-Predictive Control CEM using the REAL MiniGrid-FourRooms env.

At each step, the actual env's state is copied into a fast in-process
simulator. The simulator explores 128 candidate action sequences, scores
them with the actual reward, and the best first action is committed
to the real env.

Walls and goal are extracted from the env at each reset since
MiniGrid-FourRooms randomizes door and goal positions.
"""
import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gymnasium as gym
import minigrid


def extract_layout(real_env):
    """Extract walls and goal from the actual FourRooms env."""
    W = real_env.unwrapped.width
    H = real_env.unwrapped.height
    walls = set()
    goal = None
    for y in range(H):
        for x in range(W):
            c = real_env.unwrapped.grid.get(x, y)
            if c is not None and hasattr(c, 'type'):
                if 'wall' in str(c.type):
                    walls.add((x, y))
                elif 'goal' in str(c.type):
                    goal = (x, y)
    return walls, goal


class FourRoomsSim:
    """Mini FourRooms simulator that takes walls and goal at construction."""

    def __init__(self, walls, goal, max_steps=512):
        self.walls = walls
        self.goal = goal
        self.max_steps = max_steps
        self.reset()

    def reset(self, pos=None, direction=0):
        self.agent_pos = list(pos) if pos else [1, 1]
        self.agent_dir = direction
        self.step_count = 0
        self.done = False

    def step(self, action):
        if self.done:
            return 0.0, True, False

        if action == 0:
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == 1:
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == 2:
            dx, dy = [(1, 0), (0, 1), (-1, 0), (0, -1)][self.agent_dir]
            nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
            W = max(x for x, _ in self.walls) + 1
            H = max(y for _, y in self.walls) + 1
            if 0 <= nx < W and 0 <= ny < H and (nx, ny) not in self.walls:
                self.agent_pos = [nx, ny]

        self.step_count += 1
        reward = 0.0
        terminated = False
        truncated = False
        if tuple(self.agent_pos) == self.goal:
            reward = 1.0
            terminated = True
        if self.step_count >= self.max_steps:
            truncated = True
        if terminated or truncated:
            self.done = True
        return reward, terminated, truncated


def mpc_plan(sim, initial_pos, initial_dir, horizon=8, num_samples=64, num_elites=8, num_iterations=3, discount=0.99, action_dim=7):
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
            sim.reset(pos=initial_pos, direction=initial_dir)
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


def evaluate_mpc(env_name, num_episodes, max_steps, horizon, num_samples, num_elites, num_iterations):
    env = gym.make(env_name)
    rewards, successes, steps_to_goal = [], [], []
    for ep in tqdm(range(num_episodes), desc='mpc'):
        obs, info = env.reset()
        walls, goal = extract_layout(env)
        sim = FourRoomsSim(walls, goal, max_steps=max_steps)

        ep_r = 0.0
        success = 0
        for t in range(max_steps):
            a = mpc_plan(
                sim,
                env.unwrapped.agent_pos, env.unwrapped.agent_dir,
                horizon=horizon, num_samples=num_samples,
                num_elites=num_elites, num_iterations=num_iterations,
            )
            obs, r, term, trunc, info = env.step(a)
            ep_r += r
            if term and r > 0:
                success = 1
                steps_to_goal.append(t + 1)
                break
            if term or trunc:
                break
        rewards.append(ep_r)
        successes.append(success)
    env.close()
    return rewards, successes, steps_to_goal


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='MiniGrid-FourRooms-v0')
    p.add_argument('--horizon', type=int, default=20)
    p.add_argument('--num-samples', type=int, default=128)
    p.add_argument('--num-elites', type=int, default=16)
    p.add_argument('--num-iterations', type=int, default=4)
    p.add_argument('--num-episodes', type=int, default=20)
    p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--out', type=str, default='evaluation/mpc_real_fourrooms.json')
    args = p.parse_args()

    print(f"MPC on REAL env: {args.env}")
    print(f"  episodes={args.num_episodes}, max_steps={args.max_steps}")
    print(f"  horizon={args.horizon}, samples={args.num_samples}, elites={args.num_elites}, iters={args.num_iterations}")

    print("\n[1/2] Random baseline...")
    rs_r, ss_r, sg_r = evaluate_random(args.env, args.num_episodes, args.max_steps)
    print(f"  reward={np.mean(rs_r):.3f} ± {np.std(rs_r):.3f}  success={np.mean(ss_r):.2%}")

    print("\n[2/2] MPC planner (real env + sim rollouts)...")
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
