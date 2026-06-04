"""
Stage A: Autoencoder warmup.

The decoder in the VICReg setup learned to predict the mean observation
(constant yellow), because VICReg forces the encoder's output to be
spread but doesn't force the decoder to use it.

Goal of this stage:
    1. Get recon MSE below the VICReg baseline (5.25).
    2. Keep KL > 0 by anchoring the posterior to a fixed N(0, I) prior.

Architecture:
    trunk (encoder)  →  features
    posterior head   →  (μ_q, log_std_q)  given features
    z ~ q(z | o)                              reparameterize
    decoder(z)       →  x_hat

Loss:
    recon + β · KL( q(z|o) ‖ N(0, I) )

β = 0.001 per stage A spec — light pressure, just enough to keep
the posterior from collapsing to a delta.

No GRU, no learned prior, no dynamics. Pure AE+VAE.
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

from models.rssm import RSSM


def kl_to_standard_normal(mu, log_std):
    """KL( N(mu, sigma) ‖ N(0, I) ) per dim, summed."""
    return 0.5 * (mu ** 2 + torch.exp(2 * log_std) - 1.0 - 2 * log_std).sum(dim=-1)


def train_rssm_ae(
    trajectories_path='data/trajectories.pkl',
    latent_dim=64,
    feature_dim=256,
    batch_size=128,
    num_epochs=10,
    lr=3e-4,
    beta=0.001,
    save_path='checkpoints/rssm_ae.pt',
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

    obs_list = []
    for ep in trajectories:
        for step in ep:
            obs_list.append(step['obs'])
    obs_arr = np.array(obs_list, dtype=np.float32) / 8.0
    print(f"Total observations: {len(obs_arr)}, range [{obs_arr.min():.3f}, {obs_arr.max():.3f}], mean={obs_arr.mean():.4f}")

    obs_t = torch.FloatTensor(obs_arr)

    rssm = RSSM(
        latent_dim=latent_dim,
        feature_dim=feature_dim,
        hidden_dim=256,
        action_dim=7,
        overshoot_ks=(1, 3, 5),
    ).to(device)

    n_params = sum(p.numel() for p in rssm.parameters())
    print(f"RSSM params: {n_params:,}")

    vicreg_path = Path('checkpoints/encoder_vicreg.pt')
    if vicreg_path.exists():
        print(f"Initializing trunk + decoder from {vicreg_path}...")
        vc = torch.load(vicreg_path, map_location=device)
        loaded_t, _ = rssm.trunk.load_from_vicreg(vc['encoder_state_dict'])
        own_dec = rssm.decoder.state_dict()
        d_loaded = 0
        for k, v in vc['decoder_state_dict'].items():
            if k in own_dec and own_dec[k].shape == v.shape:
                own_dec[k].copy_(v)
                d_loaded += 1
        print(f"  trunk: {loaded_t} layers, decoder: {d_loaded} layers")

    opt = optim.Adam(rssm.parameters(), lr=lr)

    n = obs_t.shape[0]
    history = []
    best = float('inf')

    print(f"\nStage A: AE warmup, β={beta}, {num_epochs} epochs...")
    for epoch in range(num_epochs):
        rssm.train()
        perm = torch.randperm(n)
        sum_recon = 0.0
        sum_kl = 0.0
        sum_post_std = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x = obs_t[idx].to(device)
            B = x.shape[0]

            features = rssm.trunk(x)
            h = torch.zeros(B, rssm.hidden_dim, device=device)
            post_input = torch.cat([features, h], dim=-1)
            post_params = rssm.posterior_head(post_input)
            mu_q, log_std_q = rssm._split_params(post_params)
            z = rssm.reparameterize(mu_q, log_std_q)

            x_hat = rssm.decoder(z)
            recon = F.mse_loss(x_hat, x)
            kl = kl_to_standard_normal(mu_q, log_std_q).mean()
            loss = recon + beta * kl

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rssm.parameters(), 100.0)
            opt.step()

            sum_recon += float(recon.item())
            sum_kl += float(kl.item())
            sum_post_std += float(log_std_q.exp().mean().item())
            n_batches += 1

        avg_recon = sum_recon / n_batches
        avg_kl = sum_kl / n_batches
        avg_std = sum_post_std / n_batches
        history.append({'recon': avg_recon, 'kl': avg_kl, 'post_std': avg_std})
        print(
            f"Epoch {epoch+1}/{num_epochs}  recon={avg_recon:.4f}  kl={avg_kl:.4f}  "
            f"post_std={avg_std:.3f}"
        )
        if avg_kl < 0.05 and epoch > 2:
            print("  ⚠ KL collapse — increase beta or check trunk")
        if avg_std < 0.1 and epoch > 2:
            print("  ⚠ Posterior std collapsing — model ignoring latent")

        if avg_recon < best:
            best = avg_recon
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'rssm_state_dict': rssm.state_dict(),
                'latent_dim': latent_dim,
                'feature_dim': feature_dim,
                'hidden_dim': 256,
                'beta': beta,
                'epoch': epoch,
                'recon': avg_recon,
                'kl': avg_kl,
                'phase': 'A',
            }, save_path)
            print(f"  ✓ saved (recon={avg_recon:.4f}, kl={avg_kl:.4f})")

    print(f"\nDone. Best recon: {best:.4f}. Saved to {save_path}.")

    with open(save_path.replace('.pt', '_history.json'), 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--trajectories', type=str, default='data/trajectories.pkl')
    p.add_argument('--latent-dim', type=int, default=64)
    p.add_argument('--feature-dim', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--beta', type=float, default=0.001)
    p.add_argument('--save-path', type=str, default='checkpoints/rssm_ae.pt')
    p.add_argument('--device', type=str, default=None)
    args = p.parse_args()

    train_rssm_ae(
        trajectories_path=args.trajectories,
        latent_dim=args.latent_dim,
        feature_dim=args.feature_dim,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        save_path=args.save_path,
        device_name=args.device,
    )
