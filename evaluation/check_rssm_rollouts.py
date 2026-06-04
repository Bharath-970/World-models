"""
Stage 2: Visual rollout check.

Take z_0 from real obs, roll out via the prior (no observations),
decode each z_t back to image space, and save as a grid.

If the open-loop rollouts look like garbage by step 2-3,
we have the same compounding error problem as before. STOP.

Output:
    evaluation/rssm_rollouts.png  — actual vs predicted grids
    plus per-step MSE in the console
"""
import torch
import numpy as np
import argparse
from pathlib import Path
import sys
import pickle

sys.path.append(str(Path(__file__).parent.parent))

from models.rssm import RSSM


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rssm', type=str, default='checkpoints/rssm.pt')
    p.add_argument('--trajectories', type=str, default='data/trajectories.pkl')
    p.add_argument('--horizon', type=int, default=8)
    p.add_argument('--num-episodes', type=int, default=4)
    p.add_argument('--out', type=str, default='evaluation/rssm_rollouts.png')
    p.add_argument('--device', type=str, default=None)
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    print(f"Loading RSSM from {args.rssm}...")
    ckpt = torch.load(args.rssm, map_location=device)
    rssm = RSSM(
        latent_dim=ckpt['latent_dim'],
        feature_dim=ckpt['feature_dim'],
        hidden_dim=ckpt['hidden_dim'],
        action_dim=7,
    ).to(device)
    rssm.load_state_dict(ckpt['rssm_state_dict'])
    rssm.eval()
    print(f"  recon={ckpt['recon']:.4f}, kl={ckpt['kl']:.4f}")

    print(f"Loading trajectories...")
    with open(args.trajectories, 'rb') as f:
        trajs = pickle.load(f)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    chosen = rng.choice(len(trajs), size=args.num_episodes, replace=False)

    H = args.horizon
    fig, axes = plt.subplots(args.num_episodes, 2 * H + 1, figsize=((2 * H + 1) * 1.3, args.num_episodes * 1.5))
    if args.num_episodes == 1:
        axes = np.expand_dims(axes, 0)

    print(f"\nVisualizing {args.num_episodes} episodes × {H} horizon rollouts...")
    mses_per_step = np.zeros((args.num_episodes, H))

    for i, ep_idx in enumerate(chosen):
        ep = trajs[ep_idx]
        if len(ep) < H + 1:
            continue
        start = rng.integers(0, len(ep) - H - 1)
        seq_obs = np.array([ep[start + t]['obs'] for t in range(H + 1)], dtype=np.float32)
        seq_act = np.array([int(ep[start + t].get('action', 0)) for t in range(H)], dtype=np.int64)

        obs_t = torch.FloatTensor(seq_obs).to(device) / 8.0      # (H+1, 7, 7, 3)
        seq_obs_n = obs_t.cpu().numpy()                          # normalized for MSE
        act_t = torch.LongTensor(seq_act).to(device).unsqueeze(1) # (H, 1)

        with torch.no_grad():
            h = torch.zeros(1, rssm.hidden_dim, device=device)
            z = torch.zeros(1, rssm.latent_dim, device=device)
            for t in range(H):
                features_t = rssm.trunk(obs_t[t:t+1])
                h, z, _, _ = rssm.step(h, z, act_t[t], features=features_t)
            h_open, z_open = h, z

            xs_pred = [obs_t[0:1].cpu().numpy()[0]]

            for t in range(H):
                h_open, z_open, prior_params, _ = rssm.step(h_open, z_open, act_t[t], features=None)
                x_hat = rssm.decoder(z_open)
                xs_pred.append(x_hat.cpu().numpy()[0])
                mses_per_step[i, t] = float(((x_hat.cpu().numpy()[0] - seq_obs_n[t + 1]) ** 2).mean())

        xs_pred = np.stack(xs_pred, axis=0)  # (H+1, 7, 7, 3)

        axes[i, 0].imshow(seq_obs_n[0].clip(0, 1))
        axes[i, 0].set_title(f'ep{ep_idx} t=0', fontsize=7)
        for t in range(1, H + 1):
            axes[i, 2 * t - 1].imshow((seq_obs[t] / 8.0).clip(0, 1))
            axes[i, 2 * t - 1].set_title(f'true t={t}', fontsize=7)
            axes[i, 2 * t].imshow(xs_pred[t].clip(0, 1))
            axes[i, 2 * t].set_title(f'pred t={t}', fontsize=7)

        for t in range(2 * H + 1):
            axes[i, t].axis('off')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"Saved grid to {args.out}")

    print(f"\nPer-step open-loop rollout MSE (averaged over episodes):")
    for t in range(H):
        m = mses_per_step[:, t].mean()
        flag = "⚠ DIVERGED" if m > 0.10 else ("✓ OK" if m < 0.04 else "· close")
        print(f"  step {t+1}: mse={m:.4f}  {flag}")


if __name__ == "__main__":
    main()
