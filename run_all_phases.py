"""
Master runner: executes all remaining phases sequentially on the remote GPU.

Phases:
  3. Multi-step prediction analysis
  4. Reward predictor + CEM planner + evaluation vs PPO
  5. Latent dim ablation (easy + hard) + horizon ablation (easy + hard)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
PY = sys.executable


def run(cmd, log_path, cwd=None):
    print(f"\n>>> {' '.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd)
    return rc


def phase_3(args):
    print("\n========== Phase 3: Multi-step prediction ==========")
    rc1 = run(
        [PY, str(ROOT / 'evaluation/multi_step_prediction.py'),
         '--model', 'checkpoints/world_model_best.pt',
         '--trajectories', 'data/trajectories.pkl',
         '--horizons', '1', '2', '3', '5', '10', '20',
         '--num-episodes', '100',
         '--save-dir', 'evaluation/results',
         '--device', args.device or 'cuda'],
        ROOT / 'logs/phase3_easy.log',
    )
    rc2 = run(
        [PY, str(ROOT / 'evaluation/multi_step_prediction.py'),
         '--model', 'checkpoints_hard/world_model_best.pt',
         '--trajectories', 'data/trajectories_hard.pkl',
         '--horizons', '1', '2', '3', '5', '10', '20',
         '--num-episodes', '100',
         '--save-dir', 'evaluation/results_hard',
         '--device', args.device or 'cuda'],
        ROOT / 'logs/phase3_hard.log',
    )
    return rc1 == 0 and rc2 == 0


def phase_4(args):
    print("\n========== Phase 4: Reward + CEM + PPO evaluation ==========")
    results = []
    for env_tag, env_name, traj, ckpt in [
        ('easy', 'MiniGrid-Empty-5x5-v0', 'data/trajectories.pkl', 'checkpoints'),
        ('hard', 'MiniGrid-FourRooms-v0', 'data/trajectories_hard.pkl', 'checkpoints_hard'),
    ]:
        print(f"\n--- {env_tag}: {env_name} ---")
        rc = run(
            [PY, str(ROOT / 'training/train_reward_predictor.py'),
             '--trajectories', traj,
             '--encoder', f'{ckpt}/world_model_best.pt',
             '--latent-dim', '128',
             '--save-path', f'{ckpt}/reward_predictor.pt',
             '--device', args.device or 'cuda'],
            ROOT / f'logs/phase4_{env_tag}_reward.log',
        )
        if rc != 0:
            print(f"  reward training failed: {rc}")
            continue
        rc = run(
            [PY, str(ROOT / 'evaluation/evaluate_fast_planner.py'),
             '--env', env_name,
             '--checkpoints', ckpt,
             '--latent-dim', '128',
             '--horizon', '5',
             '--num-episodes', str(args.eval_episodes),
             '--max-steps', '256' if env_tag == 'easy' else '512',
             '--ppo-steps', str(args.ppo_steps),
             '--device', args.device or 'cuda',
             '--out', f'evaluation/{env_tag}_planner_results.json'],
            ROOT / f'logs/phase4_{env_tag}_eval.log',
        )
        results.append((env_tag, rc == 0))
    return all(r for _, r in results)


def phase_5_latent(args):
    print("\n========== Phase 5a: Latent dim ablation ==========")
    for env_tag, env_name, traj in [
        ('Empty-5x5', 'MiniGrid-Empty-5x5-v0', 'data/trajectories.pkl'),
        ('FourRooms', 'MiniGrid-FourRooms-v0', 'data/trajectories_hard.pkl'),
    ]:
        print(f"\n--- latent ablation: {env_tag} ---")
        run(
            [PY, str(ROOT / 'experiments/latent_size_ablation.py'),
             '--env', env_name,
             '--trajectories', traj,
             '--latent-dims', '32', '64', '128', '256',
             '--epochs', str(args.epochs),
             '--device', args.device or 'cuda',
             '--out', f'experiments/latent_ablation_{env_tag}.json',
             '--ckpt-root', f'checkpoints_ablation/{env_tag}'],
            ROOT / f'logs/phase5_latent_{env_tag}.log',
        )


def phase_5_horizon(args):
    print("\n========== Phase 5b: Horizon ablation ==========")
    for env_tag, env_name, ckpt in [
        ('Empty-5x5', 'MiniGrid-Empty-5x5-v0', 'checkpoints'),
        ('FourRooms', 'MiniGrid-FourRooms-v0', 'checkpoints_hard'),
    ]:
        print(f"\n--- horizon ablation: {env_tag} ---")
        run(
            [PY, str(ROOT / 'experiments/horizon_ablation.py'),
             '--env', env_name,
             '--checkpoints', ckpt,
             '--horizons', '1', '2', '5', '10',
             '--num-episodes', str(args.eval_episodes),
             '--max-steps', '256' if env_tag == 'Empty-5x5' else '512',
             '--device', args.device or 'cuda',
             '--out', f'experiments/horizon_ablation_{env_tag}.json'],
            ROOT / f'logs/phase5_horizon_{env_tag}.log',
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--eval-episodes', type=int, default=30)
    p.add_argument('--ppo-steps', type=int, default=20000)
    p.add_argument('--skip', type=str, nargs='*', default=[],
                   choices=['3', '4', '5a', '5b'])
    args = p.parse_args()

    t0 = time.time()
    if '3' not in args.skip:
        phase_3(args)
    if '4' not in args.skip:
        phase_4(args)
    if '5a' not in args.skip:
        phase_5_latent(args)
    if '5b' not in args.skip:
        phase_5_horizon(args)
    print(f"\nAll phases done in {(time.time() - t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
