"""
Multi-step prediction analysis.

Rolls the world model forward in latent space for horizons 1..N, comparing
predicted latents to encoded true latents. Reports open-loop MSE growth.
"""
import torch
import numpy as np
import pickle
import argparse
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from models.encoder import CNNEncoder
from models.transition import TransitionModel


def evaluate_multi_step(
    model_path='checkpoints/world_model_best.pt',
    trajectories_path='data/trajectories.pkl',
    horizons=None,
    num_episodes=100,
    device_name=None,
    save_dir='evaluation/results',
):
    if horizons is None:
        horizons = [1, 2, 3, 5, 10, 20]
    if device_name:
        device = torch.device(device_name)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    encoder = CNNEncoder(input_shape=(7, 7, 3), latent_dim=128).to(device)
    transition = TransitionModel(latent_dim=128, action_dim=7).to(device)
    ckpt = torch.load(model_path, map_location=device)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    transition.load_state_dict(ckpt['transition_state_dict'])
    encoder.eval()
    transition.eval()

    with open(trajectories_path, 'rb') as f:
        trajectories = pickle.load(f)

    max_h = max(horizons)
    per_horizon = {h: [] for h in horizons}

    for ep in tqdm(trajectories[:num_episodes], desc='episodes'):
        if len(ep) < max_h + 1:
            continue
        obs_t = torch.FloatTensor(np.array([s['obs'] for s in ep], dtype=np.float32)).to(device)
        with torch.no_grad():
            z_all = encoder(obs_t)

        for start in range(len(ep) - max_h):
            z = z_all[start:start + 1]
            for h in horizons:
                z_pred = z
                for t in range(h):
                    a = ep[start + t]['action']
                    a_t = torch.LongTensor([a]).to(device)
                    z_pred = transition(z_pred, a_t)
                z_true = z_all[start + h:start + h + 1]
                mse = float(((z_pred - z_true) ** 2).mean().item())
                per_horizon[h].append(mse)

    summary = {}
    print("\nMulti-step prediction errors:")
    print(f"{'Horizon':>8} {'Mean MSE':>12} {'Std':>12} {'Median':>12}")
    for h in horizons:
        errs = np.array(per_horizon[h])
        m, s, med = errs.mean(), errs.std(), np.median(errs)
        print(f"{h:>8} {m:>12.6f} {s:>12.6f} {med:>12.6f}")
        summary[str(h)] = {
            'mean': float(m),
            'std': float(s),
            'median': float(med),
            'n': int(len(errs)),
        }

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / 'multi_step_prediction.json', 'w') as f:
        json.dump({
            'env': trajectories_path,
            'model': model_path,
            'num_episodes': num_episodes,
            'results': summary,
        }, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    hs = horizons
    means = [summary[str(h)]['mean'] for h in hs]
    stds = [summary[str(h)]['std'] for h in hs]
    ax.errorbar(hs, means, yerr=stds, marker='o', capsize=4, linewidth=2)
    ax.set_xlabel('Prediction Horizon (steps)')
    ax.set_ylabel('Mean Squared Error (latent space)')
    ax.set_title('Multi-Step Prediction Error Growth')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'multi_step_prediction.png', dpi=150)
    print(f"\nSaved {save_path / 'multi_step_prediction.png'}")

    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='checkpoints/world_model_best.pt')
    p.add_argument('--trajectories', type=str, default='data/trajectories.pkl')
    p.add_argument('--horizons', type=int, nargs='+', default=[1, 2, 3, 5, 10, 20])
    p.add_argument('--num-episodes', type=int, default=100)
    p.add_argument('--save-dir', type=str, default='evaluation/results')
    p.add_argument('--device', type=str, default=None)
    args = p.parse_args()
    evaluate_multi_step(
        model_path=args.model,
        trajectories_path=args.trajectories,
        horizons=args.horizons,
        num_episodes=args.num_episodes,
        device_name=args.device,
        save_dir=args.save_dir,
    )
