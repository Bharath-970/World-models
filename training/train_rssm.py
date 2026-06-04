"""
Stage 1 RSSM training: reconstruction + KL + Latent Overshooting.

No planner. No value. No reward head. Just teach the model to:
    1. Reconstruct observations from the latent.
    2. Keep the prior close to the posterior (KL anchor).
    3. Predict 1/3/5-step-ahead latents consistently (Overshooting).

Monitor kl_loss, recon_loss separately. Watch for KL collapse:
if KL → 0, the model is ignoring the stochastic latent.

β = 0.01 (project convention — start light, raise only if KL explodes).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import json

sys.path.append(str(Path(__file__).parent.parent))

from models.rssm import RSSM, compute_rssm_losses


def build_sequences(trajectories, seq_len=20):
    """
    Cut trajectories into fixed-length sequences of (obs, action).

    Returns:
        obs_seq:  (N, T, H, W, C) float32
        act_seq:  (N, T)         int64
    """
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


def train_rssm(
    trajectories_path='data/trajectories.pkl',
    latent_dim=64,
    feature_dim=256,
    hidden_dim=256,
    seq_len=20,
    batch_size=16,
    num_epochs=20,
    lr=3e-4,
    beta=0.01,
    overshoot_weight=0.1,
    save_path='checkpoints/rssm.pt',
    device_name=None,
):
    if device_name:
        device = torch.device(device_name)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    print(f"Loading trajectories from {trajectories_path}...")
    with open(trajectories_path, 'rb') as f:
        trajectories = pickle.load(f)

    print("Cutting into fixed-length sequences...")
    obs_seq, act_seq = build_sequences(trajectories, seq_len=seq_len)
    obs_seq = obs_seq / 8.0
    print(f"Sequences: {obs_seq.shape}, range [{obs_seq.min():.3f}, {obs_seq.max():.3f}], actions: {act_seq.shape}")

    obs_seq = torch.FloatTensor(obs_seq)
    act_seq = torch.LongTensor(act_seq)

    rssm = RSSM(
        latent_dim=latent_dim,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        action_dim=7,
        overshoot_ks=(1, 3, 5),
    ).to(device)

    n_params = sum(p.numel() for p in rssm.parameters())
    print(f"RSSM params: {n_params:,}")

    vicreg_path = Path('checkpoints/encoder_vicreg.pt')
    if vicreg_path.exists():
        print(f"Initializing trunk + decoder from {vicreg_path}...")
        vc = torch.load(vicreg_path, map_location=device)
        loaded_t, skipped_t = rssm.trunk.load_from_vicreg(vc['encoder_state_dict'])
        own_dec = rssm.decoder.state_dict()
        d_loaded = 0
        for k, v in vc['decoder_state_dict'].items():
            if k in own_dec and own_dec[k].shape == v.shape:
                own_dec[k].copy_(v)
                d_loaded += 1
        print(f"  trunk: {loaded_t} layers loaded, {skipped_t} skipped")
        print(f"  decoder: {d_loaded} layers loaded")
    else:
        print(f"  (no VICReg checkpoint at {vicreg_path}, training from scratch)")

    opt = optim.Adam(rssm.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)

    n = obs_seq.shape[0]
    history = []
    best = float('inf')

    print(f"\nTraining RSSM for {num_epochs} epochs (β={beta}, overshoot_weight={overshoot_weight})...")
    for epoch in range(num_epochs):
        rssm.train()
        perm = torch.randperm(n)
        sums = {'recon': 0.0, 'kl': 0.0, 'overshoot': 0.0, 'prior_std': 0.0, 'post_std': 0.0, 'kl_per_dim': 0.0}
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            obs_b = obs_seq[idx].to(device)     # (B, T, H, W, C)
            act_b = act_seq[idx].to(device)     # (B, T)
            T = obs_b.shape[1]

            obs_seq_t = obs_b.transpose(0, 1).contiguous()    # (T, B, H, W, C)
            act_seq_t = act_b.transpose(0, 1).contiguous()    # (T, B)

            out = rssm.forward_train(obs_seq_t, act_seq_t)

            with torch.no_grad():
                h0 = out['hs'][0] * 0.0
                z0 = out['zs_post'][0] * 0.0
                _, _, _, _, K_mus, K_log_stds = rssm.rollout_overshoot(h0, z0, act_seq_t)
                overshoot_targets = {
                    'overshoot_mus': K_mus,
                    'overshoot_log_stds': K_log_stds,
                }

            loss, info = compute_rssm_losses(
                out, obs_seq_t,
                beta=beta,
                overshoot_targets=overshoot_targets,
                overshoot_weight=overshoot_weight,
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rssm.parameters(), 100.0)
            opt.step()

            sums['recon'] += info['recon']
            sums['kl'] += info['kl']
            sums['kl_per_dim'] += info['kl_per_dim']
            sums['prior_std'] += info['prior_std_mean']
            sums['post_std'] += info['post_std_mean']
            if 'overshoot' in info:
                sums['overshoot'] += info['overshoot']
            n_batches += 1

        sched.step()

        avg = {k: v / max(n_batches, 1) for k, v in sums.items()}
        history.append(avg)
        line = (
            f"Epoch {epoch+1}/{num_epochs}  "
            f"recon={avg['recon']:.4f}  kl={avg['kl']:.4f}  "
            f"kl/dim={avg['kl_per_dim']:.4f}  "
            f"prior_std={avg['prior_std']:.3f}  post_std={avg['post_std']:.3f}  "
            f"overshoot={avg['overshoot']:.4f}"
        )
        print(line)

        # KL collapse warning
        if avg['kl_per_dim'] < 0.01 and epoch > 3:
            print("  ⚠ KL collapse warning: posterior ≈ prior. Increase beta or check encoder.")
        if avg['post_std'] < 0.1 and epoch > 3:
            print("  ⚠ Posterior std collapsing: model ignoring stochastic state.")

        score = avg['recon'] + beta * avg['kl']
        if score < best:
            best = score
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'rssm_state_dict': rssm.state_dict(),
                'latent_dim': latent_dim,
                'feature_dim': feature_dim,
                'hidden_dim': hidden_dim,
                'beta': beta,
                'overshoot_weight': overshoot_weight,
                'epoch': epoch,
                'recon': avg['recon'],
                'kl': avg['kl'],
            }, save_path)
            print(f"  ✓ saved (recon={avg['recon']:.4f}, kl={avg['kl']:.4f})")

    print(f"\nDone. Best score: {best:.4f}. Saved to {save_path}.")

    with open(save_path.replace('.pt', '_history.json'), 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectories', type=str, default='data/trajectories.pkl')
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--feature-dim', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--seq-len', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--overshoot-weight', type=float, default=0.1)
    parser.add_argument('--save-path', type=str, default='checkpoints/rssm.pt')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    train_rssm(
        trajectories_path=args.trajectories,
        latent_dim=args.latent_dim,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        overshoot_weight=args.overshoot_weight,
        save_path=args.save_path,
        device_name=args.device,
    )
