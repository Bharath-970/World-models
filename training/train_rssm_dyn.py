"""
Stage B: Add dynamics.

Loads Stage A's checkpoint, adds the GRU + learned prior, and trains:
    recon + β · KL(q ‖ p) + 1-step dynamics

The 1-step dynamics loss is the same as the KL term — it's enforced by
having the prior and posterior both predict z_{t+1} from the same context.
At training time, the posterior uses the actual obs so the prior is
forced to match.

Architecture: full RSSM
Loss: recon + β·KL + (no overshoot yet)
β=0.1 to keep the prior close to the posterior at this stage.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
import argparse
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent.parent))

from models.rssm import RSSM, compute_rssm_losses


def build_sequences(trajectories, seq_len=20):
    obs_list, act_list = [], []
    for ep in trajectories:
        T = len(ep)
        if T < seq_len + 1:
            continue
        for start in range(0, T - seq_len, seq_len):
            obs_chunk = np.array([ep[start + i]['obs'] for i in range(seq_len)], dtype=np.float32)
            act_chunk = np.array([int(ep[start + i].get('action', 0)) for i in range(seq_len)], dtype=np.int64)
            obs_list.append(obs_chunk)
            act_list.append(act_chunk)
    return np.stack(obs_list, axis=0), np.stack(act_list, axis=0)


def train_rssm_dyn(
    ae_path='checkpoints/rssm_ae.pt',
    trajectories_path='data/trajectories.pkl',
    seq_len=20,
    batch_size=16,
    num_epochs=10,
    lr=1e-4,
    beta=0.1,
    save_path='checkpoints/rssm_dyn.pt',
    device_name=None,
):
    if device_name:
        device = torch.device(device_name)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    print(f"Loading trajectories from {trajectories_path}...")
    with open(trajectories_path, 'rb') as f:
        trajectories = pickle.load(f)
    obs_seq, act_seq = build_sequences(trajectories, seq_len=seq_len)
    obs_seq = obs_seq / 8.0
    print(f"Sequences: {obs_seq.shape}, actions: {act_seq.shape}")

    obs_seq = torch.FloatTensor(obs_seq)
    act_seq = torch.LongTensor(act_seq)

    print(f"Loading Stage A checkpoint from {ae_path}...")
    ae_ckpt = torch.load(ae_path, map_location=device)
    rssm = RSSM(
        latent_dim=ae_ckpt['latent_dim'],
        feature_dim=ae_ckpt['feature_dim'],
        hidden_dim=ae_ckpt['hidden_dim'],
        action_dim=7,
        overshoot_ks=(1, 3, 5),
    ).to(device)
    rssm.load_state_dict(ae_ckpt['rssm_state_dict'])
    print(f"  Stage A: recon={ae_ckpt['recon']:.4f}, kl={ae_ckpt['kl']:.4f}")

    print(f"RSSM params: {sum(p.numel() for p in rssm.parameters()):,}")

    opt = optim.Adam(rssm.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)

    n = obs_seq.shape[0]
    history = []
    best = float('inf')

    print(f"\nStage B: dynamics, β={beta}, {num_epochs} epochs...")
    for epoch in range(num_epochs):
        rssm.train()
        perm = torch.randperm(n)
        sums = {'recon': 0.0, 'kl': 0.0, 'kl_per_dim': 0.0, 'prior_std': 0.0, 'post_std': 0.0}
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            obs_b = obs_seq[idx].to(device)
            act_b = act_seq[idx].to(device)
            T = obs_b.shape[1]

            obs_seq_t = obs_b.transpose(0, 1).contiguous()
            act_seq_t = act_b.transpose(0, 1).contiguous()

            out = rssm.forward_train(obs_seq_t, act_seq_t)
            loss, info = compute_rssm_losses(out, obs_seq_t, beta=beta, overshoot_targets=None)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rssm.parameters(), 100.0)
            opt.step()

            sums['recon'] += info['recon']
            sums['kl'] += info['kl']
            sums['kl_per_dim'] += info['kl_per_dim']
            sums['prior_std'] += info['prior_std_mean']
            sums['post_std'] += info['post_std_mean']
            n_batches += 1

        sched.step()
        avg = {k: v / max(n_batches, 1) for k, v in sums.items()}
        history.append(avg)
        print(
            f"Epoch {epoch+1}/{num_epochs}  recon={avg['recon']:.4f}  kl={avg['kl']:.4f}  "
            f"kl/dim={avg['kl_per_dim']:.4f}  prior_std={avg['prior_std']:.3f}  post_std={avg['post_std']:.3f}"
        )

        if avg['kl_per_dim'] < 0.01 and epoch > 3:
            print("  ⚠ KL collapse — increase beta or check encoder")
        if avg['post_std'] < 0.1 and epoch > 3:
            print("  ⚠ Posterior std collapsing — model ignoring stochastic state")

        score = avg['recon'] + beta * avg['kl']
        if score < best:
            best = score
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'rssm_state_dict': rssm.state_dict(),
                'latent_dim': ae_ckpt['latent_dim'],
                'feature_dim': ae_ckpt['feature_dim'],
                'hidden_dim': ae_ckpt['hidden_dim'],
                'beta': beta,
                'epoch': epoch,
                'recon': avg['recon'],
                'kl': avg['kl'],
                'phase': 'B',
            }, save_path)
            print(f"  ✓ saved (recon={avg['recon']:.4f}, kl={avg['kl']:.4f})")

    print(f"\nDone. Best score: {best:.4f}. Saved to {save_path}.")
    with open(save_path.replace('.pt', '_history.json'), 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--ae-path', type=str, default='checkpoints/rssm_ae.pt')
    p.add_argument('--trajectories', type=str, default='data/trajectories.pkl')
    p.add_argument('--seq-len', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--save-path', type=str, default='checkpoints/rssm_dyn.pt')
    p.add_argument('--device', type=str, default=None)
    args = p.parse_args()

    train_rssm_dyn(
        ae_path=args.ae_path,
        trajectories_path=args.trajectories,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        save_path=args.save_path,
        device_name=args.device,
    )
