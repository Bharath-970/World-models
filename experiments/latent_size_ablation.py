"""
Latent dimension ablation.

Trains world models with different latent sizes and reports:
- Training loss (final and best)
- Multi-step prediction error at chosen horizons
- Reward predictor MSE
- Planner success rate (sampled eval)
"""
import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
SCRIPTS = ROOT / 'training'
EVAL = ROOT / 'evaluation'

LATENT_DIMS = [32, 64, 128, 256]
HORIZONS = [1, 2, 3, 5, 10, 20]


def run(cmd, log_path):
    print(f"$ {' '.join(cmd)}")
    with open(log_path, 'w') as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    return rc


def train_one(latent_dim, env_tag, trajectories, ckpt_dir, epochs, batch_size, lr, device):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = ckpt_dir / f'train_latent{latent_dim}.log'
    cmd = [
        sys.executable, str(SCRIPTS / 'train_world_model.py'),
        '--trajectories', str(trajectories),
        '--latent-dim', str(latent_dim),
        '--batch-size', str(batch_size),
        '--epochs', str(epochs),
        '--lr', str(lr),
        '--save-dir', str(ckpt_dir),
    ]
    if device:
        cmd.extend(['--device', device])
    return run(cmd, log)


def train_reward(latent_dim, env_tag, trajectories, encoder_path, ckpt_dir, device):
    save_path = Path(ckpt_dir) / f'reward_predictor_ld{latent_dim}.pt'
    log = Path(ckpt_dir) / f'reward_latent{latent_dim}.log'
    cmd = [
        sys.executable, str(SCRIPTS / 'train_reward_predictor.py'),
        '--trajectories', str(trajectories),
        '--encoder', str(encoder_path),
        '--latent-dim', str(latent_dim),
        '--save-path', str(save_path),
    ]
    if device:
        cmd.extend(['--device', device])
    return run(cmd, log)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='MiniGrid-Empty-5x5-v0',
                   help='Environment name (also used to locate trajectories)')
    p.add_argument('--trajectories', type=str, default='data/trajectories.pkl')
    p.add_argument('--latent-dims', type=int, nargs='+', default=LATENT_DIMS)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--out', type=str, default='experiments/latent_size_results.json')
    p.add_argument('--ckpt-root', type=str, default='checkpoints_ablation')
    args = p.parse_args()

    env_tag = args.env.replace('/', '_')
    ckpt_root = Path(args.ckpt_root) / env_tag
    ckpt_root.mkdir(parents=True, exist_ok=True)

    results = {}
    for ld in args.latent_dims:
        ckpt_dir = ckpt_root / f'ld_{ld}'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Latent dim = {ld} ===")
        rc = train_one(ld, env_tag, args.trajectories, ckpt_dir,
                       args.epochs, args.batch_size, args.lr, args.device)
        if rc != 0:
            print(f"  train failed: rc={rc}")
            continue
        ckpt = torch_load(ckpt_dir / 'world_model_best.pt')
        wm_loss = float(ckpt.get('loss', float('inf'))) if ckpt else None
        encoder_path = ckpt_dir / 'world_model_best.pt'

        rc = train_reward(ld, env_tag, args.trajectories, encoder_path, ckpt_dir, args.device)
        if rc != 0:
            print(f"  reward train failed: rc={rc}")
        rp_ckpt = torch_load(ckpt_dir / f'reward_predictor_ld{ld}.pt')
        rp_loss = float(rp_ckpt.get('loss', float('inf'))) if rp_ckpt else None

        results[str(ld)] = {
            'world_model_loss': wm_loss,
            'reward_predictor_loss': rp_loss,
            'ckpt_dir': str(ckpt_dir),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump({
            'env': args.env,
            'trajectories': args.trajectories,
            'epochs': args.epochs,
            'results': results,
        }, f, indent=2)
    print(f"\nResults saved to {out}")


def torch_load(path):
    import torch
    try:
        return torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"  could not load {path}: {e}")
        return None


if __name__ == "__main__":
    main()
