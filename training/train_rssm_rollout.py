"""
Stage C: Multi-step rollout consistency (Latent Overshooting).

Builds on Stage B. Adds the multi-step rollout loss from compute_rssm_losses:
    total = recon + β·KL + overshoot_weight · overshoot_loss

The overshoot loss is a consistency loss: at each h_t, predict z_{t+k} for
k=1,3,5, and match the actual posterior z_{t+k} at those future steps.

This directly attacks the failure mode the user identified:
    step 1 good, step 3 garbage.

Loss weights per user spec:
    L = recon + KL + 1.0 * 1-step + 0.5 * 3-step + 0.25 * 5-step
    (achieved via overshoot_weight=1.0, single overshoot_heads list with
    weights baked in)
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


def weighted_overshoot_loss(K_mus, K_log_stds, post_mus_seq, post_log_stds_seq, weights):
    """Sum of weighted KLs at 1/3/5-step ahead."""
    ks = [1, 3, 5][:len(K_mus)]
    total = 0.0
    for k_idx, k in enumerate(ks):
        mu_k = K_mus[k_idx]
        log_std_k = K_log_stds[k_idx]
        mu_target = post_mus_seq[k:]
        log_std_target = post_log_stds_seq[k:]
        kl = 0.5 * (
            2 * (log_std_target - log_std_k)
            + (torch.exp(2 * log_std_k) + (mu_k - mu_target) ** 2)
            / (torch.exp(2 * log_std_target) + 1e-8)
            - 1.0
        ).sum(dim=-1).mean()
        total = total + weights[k_idx] * kl
    return total


def train_rssm_rollout(
    dyn_path='checkpoints/rssm_dyn.pt',
    trajectories_path='data/trajectories.pkl',
    seq_len=20,
    batch_size=16,
    num_epochs=8,
    lr=5e-5,
    beta=0.1,
    overshoot_weight=0.5,
    overshoot_w=(1.0, 0.5, 0.25),
    save_path='checkpoints/rssm_rollout.pt',
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

    print(f"Loading Stage B checkpoint from {dyn_path}...")
    dyn_ckpt = torch.load(dyn_path, map_location=device)
    rssm = RSSM(
        latent_dim=dyn_ckpt['latent_dim'],
        feature_dim=dyn_ckpt['feature_dim'],
        hidden_dim=dyn_ckpt['hidden_dim'],
        action_dim=7,
        overshoot_ks=(1, 3, 5),
    ).to(device)
    rssm.load_state_dict(dyn_ckpt['rssm_state_dict'])
    print(f"  Stage B: recon={dyn_ckpt['recon']:.4f}, kl={dyn_ckpt['kl']:.4f}")
    print(f"RSSM params: {sum(p.numel() for p in rssm.parameters()):,}")

    opt = optim.Adam(rssm.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)

    n = obs_seq.shape[0]
    history = []
    best = float('inf')

    print(f"\nStage C: rollout consistency, β={beta}, w={overshoot_w}, {num_epochs} epochs...")
    for epoch in range(num_epochs):
        rssm.train()
        perm = torch.randperm(n)
        sums = {'recon': 0.0, 'kl': 0.0, 'kl_per_dim': 0.0, 'prior_std': 0.0, 'post_std': 0.0, 'overshoot': 0.0}
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            obs_b = obs_seq[idx].to(device)
            act_b = act_seq[idx].to(device)
            T = obs_b.shape[1]

            obs_seq_t = obs_b.transpose(0, 1).contiguous()
            act_seq_t = act_b.transpose(0, 1).contiguous()

            out = rssm.forward_train(obs_seq_t, act_seq_t)
            _, info = compute_rssm_losses(out, obs_seq_t, beta=beta, overshoot_targets=None)

            with torch.no_grad():
                h0 = torch.zeros(act_seq_t.shape[1], rssm.hidden_dim, device=device)
                z0 = torch.zeros(act_seq_t.shape[1], rssm.latent_dim, device=device)
                _, _, _, _, K_mus, K_log_stds = rssm.rollout_overshoot(h0, z0, act_seq_t)

            ov_loss = weighted_overshoot_loss(
                K_mus, K_log_stds,
                out['post_mus'], out['post_log_stds'],
                overshoot_w,
            )
            recon = F.mse_loss(out['x_hats'], obs_seq_t)
            kl = 0.5 * (
                2 * (out['post_log_stds'] - out['prior_log_stds'])
                + (torch.exp(2 * out['prior_log_stds']) + (out['prior_mus'] - out['post_mus']) ** 2)
                / (torch.exp(2 * out['post_log_stds']) + 1e-8)
                - 1.0
            ).sum(dim=-1).mean()
            loss = recon + beta * kl + overshoot_weight * ov_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rssm.parameters(), 100.0)
            opt.step()

            sums['recon'] += float(recon.item())
            sums['kl'] += float(kl.item())
            sums['kl_per_dim'] += float((kl / max(out['zs_post'].shape[-1], 1)).item())
            sums['prior_std'] += float(out['prior_log_stds'].exp().mean().item())
            sums['post_std'] += float(out['post_log_stds'].exp().mean().item())
            sums['overshoot'] += float(ov_loss.item())
            n_batches += 1

        sched.step()
        avg = {k: v / max(n_batches, 1) for k, v in sums.items()}
        history.append(avg)
        print(
            f"Epoch {epoch+1}/{num_epochs}  recon={avg['recon']:.4f}  kl={avg['kl']:.4f}  "
            f"kl/dim={avg['kl_per_dim']:.4f}  prior_std={avg['prior_std']:.3f}  "
            f"post_std={avg['post_std']:.3f}  overshoot={avg['overshoot']:.4f}"
        )
        if avg['kl_per_dim'] < 0.01 and epoch > 3:
            print("  ⚠ KL collapse")

        score = avg['recon'] + beta * avg['kl'] + overshoot_weight * avg['overshoot']
        if score < best:
            best = score
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'rssm_state_dict': rssm.state_dict(),
                'latent_dim': dyn_ckpt['latent_dim'],
                'feature_dim': dyn_ckpt['feature_dim'],
                'hidden_dim': dyn_ckpt['hidden_dim'],
                'beta': beta,
                'overshoot_weight': overshoot_weight,
                'overshoot_w': list(overshoot_w),
                'epoch': epoch,
                'recon': avg['recon'],
                'kl': avg['kl'],
                'overshoot': avg['overshoot'],
                'phase': 'C',
            }, save_path)
            print(f"  ✓ saved (recon={avg['recon']:.4f}, kl={avg['kl']:.4f}, overshoot={avg['overshoot']:.4f})")

    print(f"\nDone. Best score: {best:.4f}. Saved to {save_path}.")
    with open(save_path.replace('.pt', '_history.json'), 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--dyn-path', type=str, default='checkpoints/rssm_dyn.pt')
    p.add_argument('--trajectories', type=str, default='data/trajectories.pkl')
    p.add_argument('--seq-len', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--overshoot-weight', type=float, default=0.5)
    p.add_argument('--overshoot-w1', type=float, default=1.0)
    p.add_argument('--overshoot-w3', type=float, default=0.5)
    p.add_argument('--overshoot-w5', type=float, default=0.25)
    p.add_argument('--save-path', type=str, default='checkpoints/rssm_rollout.pt')
    p.add_argument('--device', type=str, default=None)
    args = p.parse_args()

    train_rssm_rollout(
        dyn_path=args.dyn_path,
        trajectories_path=args.trajectories,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        overshoot_weight=args.overshoot_weight,
        overshoot_w=(args.overshoot_w1, args.overshoot_w3, args.overshoot_w5),
        save_path=args.save_path,
        device_name=args.device,
    )
