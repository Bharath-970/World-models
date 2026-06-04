"""
Position-tracking linear probe analysis.

Tests how much spatial (x, y) information is encoded in RSSM latents
by training a linear model to predict agent position from latent state.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import argparse
from pathlib import Path
import sys
import json
sys.path.append(str(Path(__file__).parent.parent))

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
from models.rssm import RSSM


def encode_obs_batch(rssm, obs_batch, device):
    """Encode a batch of observations to RSSM posterior latents."""
    x = torch.FloatTensor(obs_batch).to(device)
    features = rssm.trunk(x)
    h = torch.zeros(x.shape[0], rssm.hidden_dim, device=device)
    post_input = torch.cat([features, h], dim=-1)
    post_params = rssm.posterior_head(post_input)
    mu, log_std = rssm._split_params(post_params)
    z = rssm.reparameterize(mu, log_std)
    return z, mu


class LinearProbe(nn.Module):
    """Linear probe: latent -> (x, y)."""
    def __init__(self, latent_dim):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 2)

    def forward(self, z):
        return self.linear(z)


def run_position_probe(rssm_path='checkpoints/rssm_rollout.pt',
                       trajectories_path='data/trajectories.pkl',
                       device_name='cuda'):
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    env = gym.make('MiniGrid-Empty-5x5-v0', render_mode=None)
    env = ImgObsWrapper(env)

    print(f"Loading trajectories from {trajectories_path}...")
    with open(trajectories_path, 'rb') as f:
        trajectories = pickle.load(f)
    print(f"  {len(trajectories)} episodes")

    print(f"Loading RSSM from {rssm_path}...")
    ckpt = torch.load(rssm_path, map_location='cpu')
    rssm = RSSM(
        latent_dim=ckpt['latent_dim'],
        feature_dim=ckpt.get('feature_dim', 256),
        hidden_dim=ckpt.get('hidden_dim', 256),
        action_dim=7,
        overshoot_ks=(1, 3, 5),
    ).to(device)
    rssm.load_state_dict(ckpt['rssm_state_dict'])
    rssm.eval()
    print(f"  latent_dim={ckpt['latent_dim']}")

    all_z = []
    all_xy = []
    n_collected = 0
    print("\nReplaying trajectories to get latents + positions...")
    for ep_idx, ep in enumerate(trajectories[:200]):
        obs, _ = env.reset()
        ep_z = []
        ep_xy = []
        for step_idx, step in enumerate(ep):
            o = step['obs'].astype(np.float32) / 8.0
            z, mu = encode_obs_batch(rssm, o[None], device)
            ep_z.append(mu.detach().cpu().numpy()[0])
            ep_xy.append(list(env.unwrapped.agent_pos))
            action = step.get('action', 0)
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        all_z.append(np.stack(ep_z, axis=0))
        all_xy.append(np.array(ep_xy, dtype=np.float32))
        n_collected += len(ep_xy)
        if (ep_idx + 1) % 50 == 0:
            print(f"  {ep_idx+1}/{len(trajectories[:200])} episodes, {n_collected} samples")
    env.close()

    Z = np.concatenate(all_z, axis=0)
    XY = np.concatenate(all_xy, axis=0)
    print(f"\nCollected {Z.shape[0]} samples, latent dim {Z.shape[1]}")

    # Split
    n = Z.shape[0]
    perm = np.random.RandomState(42).permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]
    Z_train, Z_test = Z[train_idx], Z[test_idx]
    XY_train, XY_test = XY[train_idx], XY[test_idx]
    print(f"Train: {len(Z_train)}, Test: {len(Z_test)}")

    # Normalize positions
    xy_mean = XY_train.mean(axis=0)
    xy_std = XY_train.std(axis=0).clip(min=1e-8)
    XY_train_n = (XY_train - xy_mean) / xy_std
    XY_test_n = (XY_test - xy_mean) / xy_std

    # Train linear probe
    probe = LinearProbe(Z.shape[1]).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    batch_size = 1024
    n_epochs = 50

    Z_train_t = torch.FloatTensor(Z_train).to(device)
    XY_train_t = torch.FloatTensor(XY_train_n).to(device)
    Z_test_t = torch.FloatTensor(Z_test).to(device)
    XY_test_t = torch.FloatTensor(XY_test_n).to(device)

    print(f"\nTraining linear probe ({n_epochs} epochs)...")
    for epoch in range(n_epochs):
        perm_epoch = torch.randperm(Z_train_t.shape[0])
        losses = []
        for i in range(0, Z_train_t.shape[0], batch_size):
            idx = perm_epoch[i:i + batch_size]
            pred = probe(Z_train_t[idx])
            loss = F.mse_loss(pred, XY_train_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                pred_test = probe(Z_test_t)
                test_loss = F.mse_loss(pred_test, XY_test_t).item()
            print(f"  Epoch {epoch+1}/{n_epochs}  train_loss={np.mean(losses):.6f}  test_loss={test_loss:.6f}")

    # Final evaluation
    probe.eval()
    with torch.no_grad():
        pred_train = probe(Z_train_t).cpu().numpy()
        pred_test = probe(Z_test_t).cpu().numpy()
        pred_train_xy = pred_train * xy_std + xy_mean
        pred_test_xy = pred_test * xy_std + xy_mean

    train_rmse = np.sqrt(np.mean((pred_train_xy - XY_train) ** 2))
    test_rmse = np.sqrt(np.mean((pred_test_xy - XY_test) ** 2))
    train_rmse_x = np.sqrt(np.mean((pred_train_xy[:, 0] - XY_train[:, 0]) ** 2))
    train_rmse_y = np.sqrt(np.mean((pred_train_xy[:, 1] - XY_train[:, 1]) ** 2))
    test_rmse_x = np.sqrt(np.mean((pred_test_xy[:, 0] - XY_test[:, 0]) ** 2))
    test_rmse_y = np.sqrt(np.mean((pred_test_xy[:, 1] - XY_test[:, 1]) ** 2))

    # Baselines: predicting mean position, or grid center
    mean_pred = np.full_like(XY_test, XY_train.mean(axis=0))
    baseline_rmse = np.sqrt(np.mean((mean_pred - XY_test) ** 2))
    center_pred = np.full_like(XY_test, [2.0, 2.0])
    center_rmse = np.sqrt(np.mean((center_pred - XY_test) ** 2))

    # Position-dependent analysis
    grid_size = 5
    cell_errors = np.zeros((grid_size, grid_size))
    cell_counts = np.zeros((grid_size, grid_size))
    for i in range(len(XY_test)):
        x, y = int(round(XY_test[i, 0])), int(round(XY_test[i, 1]))
        if 0 <= x < grid_size and 0 <= y < grid_size:
            cell_errors[y, x] += np.sqrt(((pred_test_xy[i] - XY_test[i]) ** 2).sum())
            cell_counts[y, x] += 1
    cell_rmse = np.divide(cell_errors, cell_counts, where=cell_counts > 0)

    results = {
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'train_rmse_x': float(train_rmse_x),
        'train_rmse_y': float(train_rmse_y),
        'test_rmse_x': float(test_rmse_x),
        'test_rmse_y': float(test_rmse_y),
        'baseline_mean_rmse': float(baseline_rmse),
        'baseline_center_rmse': float(center_rmse),
        'n_train': int(len(Z_train)),
        'n_test': int(len(Z_test)),
        'cell_rmse': cell_rmse.tolist(),
    }

    print(f"\n{'='*60}")
    print(f"Position-Tracking Results")
    print(f"{'='*60}")
    print(f"  Train RMSE:         {train_rmse:.4f} (x: {train_rmse_x:.4f}, y: {train_rmse_y:.4f})")
    print(f"  Test RMSE:          {test_rmse:.4f} (x: {test_rmse_x:.4f}, y: {test_rmse_y:.4f})")
    print(f"  Baseline (mean):    {baseline_rmse:.4f}")
    print(f"  Baseline (center):  {center_rmse:.4f}")
    print(f"  R² (test):          {1 - test_rmse**2 / baseline_rmse**2:.4f}")
    print()
    print(f"  Grid cell RMSE (5x5):")
    for y in range(grid_size):
        row = '  |' + '|'.join(f" {cell_rmse[y, x]:.2f} " for x in range(grid_size)) + '|'
        print(row)
    print(f"{'='*60}")

    save_path = Path(rssm_path).parent / f"position_probe_{Path(rssm_path).stem}.json"
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rssm', type=str, default='checkpoints/rssm_rollout.pt')
    p.add_argument('--trajectories', type=str, default='data/trajectories.pkl')
    p.add_argument('--device', type=str, default=None)
    args = p.parse_args()
    run_position_probe(
        rssm_path=args.rssm,
        trajectories_path=args.trajectories,
        device_name=args.device,
    )
