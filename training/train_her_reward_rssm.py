"""
Stage 4: Train HER reward predictor on RSSM latents.

The previous HER predictor was trained on the 128-dim VICReg encoder
latents. The RSSM uses 64-dim stochastic latents, so we need a new
predictor with the right input dimension.

Pipeline:
    1. Load trained RSSM.
    2. For each obs, compute the posterior z ~ q(z|o, h) (with h=0 since
       this is single-step encoding).
    3. HER relabel: pair each obs with K fake goals from the same episode.
    4. Train MLP to predict reward(z_t, z_goal) → soft distance.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import argparse
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.rssm import RSSM


def her_relabel(trajectories, n_goals=4, strategy='binary'):
    """
    Binary HER: pair each timestep with K fake goals from the same
    trajectory. Reward is 1 if the fake goal is a terminal goal state
    (achieved the env's actual goal), 0 otherwise.

    This avoids the soft-distance saturation problem: a clear 0/1 signal
    that the predictor can actually learn.
    """
    samples = []
    for ep in trajectories:
        obs_list = [s['obs'] for s in ep]
        rew_list = [s.get('reward', 0) for s in ep]
        L = len(obs_list)
        if L < 2:
            continue
        for t in range(L):
            obs_t = obs_list[t]
            if strategy == 'binary':
                # Mix of (a) the actual goal, (b) random non-goal states
                # Pair 50% with the actual goal, 50% with random states
                # If the actual goal exists in this episode
                actual_goal_idx = None
                for i, r in enumerate(rew_list):
                    if r > 0:
                        actual_goal_idx = i
                        break

                if actual_goal_idx is not None and np.random.random() < 0.5:
                    obs_g = obs_list[actual_goal_idx]
                    samples.append((obs_t, obs_g, 1.0))
                else:
                    # Random non-goal
                    non_goal_idxs = [i for i in range(L) if rew_list[i] <= 0]
                    if non_goal_idxs:
                        g_idx = np.random.choice(non_goal_idxs)
                        obs_g = obs_list[g_idx]
                        samples.append((obs_t, obs_g, 0.0))
            else:
                raise ValueError(strategy)
    return samples


class HERRewardHead(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_t, z_goal):
        z = torch.cat([z_t, z_goal], dim=-1)
        return torch.sigmoid(self.net(z).squeeze(-1))


def encode_obs_with_rssm(rssm, obs_arr, device, batch_size=256):
    """
    Encode observations using the RSSM posterior with h=0.
    Returns z_arr of shape (N, latent_dim).
    """
    z_all = []
    rssm.eval()
    with torch.no_grad():
        for i in range(0, len(obs_arr), batch_size):
            batch = torch.FloatTensor(obs_arr[i:i + batch_size]).to(device) / 8.0
            features = rssm.trunk(batch)
            h = torch.zeros(batch.shape[0], rssm.hidden_dim, device=device)
            post_input = torch.cat([features, h], dim=-1)
            post_params = rssm.posterior_head(post_input)
            mu_q, _ = rssm._split_params(post_params)
            z_all.append(mu_q.cpu().numpy())
    return np.concatenate(z_all, axis=0)


def train_her_on_rssm(
    rssm_path='checkpoints/rssm_rollout.pt',
    trajectories_path='data/trajectories.pkl',
    n_goals=4,
    sigma=0.5,
    batch_size=256,
    num_epochs=15,
    lr=3e-4,
    save_path='checkpoints/reward_predictor_rssm.pt',
    device_name=None,
    binary=True,
):
    if device_name:
        device = torch.device(device_name)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    print(f"Loading RSSM from {rssm_path}...")
    rssm_ckpt = torch.load(rssm_path, map_location=device)
    rssm = RSSM(
        latent_dim=rssm_ckpt['latent_dim'],
        feature_dim=rssm_ckpt['feature_dim'],
        hidden_dim=rssm_ckpt['hidden_dim'],
    ).to(device)
    rssm.load_state_dict(rssm_ckpt['rssm_state_dict'])
    rssm.eval()
    for p in rssm.parameters():
        p.requires_grad = False
    print(f"  recon={rssm_ckpt['recon']:.4f}, kl={rssm_ckpt['kl']:.4f}")

    print(f"Loading trajectories from {trajectories_path}...")
    with open(trajectories_path, 'rb') as f:
        trajectories = pickle.load(f)

    print(f"HER relabelling (binary={binary})...")
    if binary:
        samples = her_relabel(trajectories, strategy='binary')
    else:
        samples = her_relabel(trajectories, n_goals=n_goals, sigma=sigma)
    print(f"Generated {len(samples)} (obs, goal, reward) pairs")

    obs_arr = np.array([s[0] for s in samples], dtype=np.float32)
    goal_arr = np.array([s[1] for s in samples], dtype=np.float32)
    reward_arr = np.array([s[2] for s in samples], dtype=np.float32)
    print(f"Reward stats: mean={reward_arr.mean():.4f}, std={reward_arr.std():.4f}, max={reward_arr.max():.4f}")
    print(f"  binary counts: 0={int((reward_arr == 0).sum())}, 1={int((reward_arr == 1).sum())}")

    print("Encoding obs with RSSM posterior...")
    z_t = encode_obs_with_rssm(rssm, obs_arr, device, batch_size)
    print("Encoding goals with RSSM posterior...")
    z_goal = encode_obs_with_rssm(rssm, goal_arr, device, batch_size)
    print(f"RSSM latents shape: {z_t.shape}")
    sample_idx = np.random.choice(len(z_t), 200, replace=False)
    z_sample = z_t[sample_idx]
    pd = np.mean(np.sqrt(np.sum((z_sample[:, None] - z_sample[None, :]) ** 2, axis=-1)))
    print(f"  z_t pairwise dist (200 sample): {pd:.4f}")

    z_t_t = torch.FloatTensor(z_t)
    z_g_t = torch.FloatTensor(z_goal)
    r_t = torch.FloatTensor(reward_arr)

    predictor = HERRewardHead(latent_dim=rssm_ckpt['latent_dim']).to(device)
    opt = optim.Adam(predictor.parameters(), lr=lr)

    n = len(z_t_t)
    history = []
    best = float('inf')
    print(f"\nTraining HER reward predictor on RSSM latents for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        predictor.train()
        perm = torch.randperm(n)
        sum_loss = 0.0
        sum_correct = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            zt = z_t_t[idx].to(device)
            zg = z_g_t[idx].to(device)
            r = r_t[idx].to(device)

            pred = predictor(zt, zg)
            loss = ((pred - r) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            sum_loss += float(loss.item())
            sum_correct += float(((pred > 0.5).float() == r).float().mean().item())
            n_batches += 1

        avg = sum_loss / n_batches
        avg_acc = sum_correct / n_batches
        history.append(avg)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg:.6f}, Acc: {avg_acc:.4f}")

        if avg < best:
            best = avg
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'predictor_state_dict': predictor.state_dict(),
                'latent_dim': rssm_ckpt['latent_dim'],
                'sigma': sigma,
                'n_goals': n_goals,
                'loss': best,
            }, save_path)
            print(f"  ✓ saved")

    print(f"\nDone. Best loss: {best:.6f}. Saved to {save_path}.")
    with open(save_path.replace('.pt', '_history.json'), 'w') as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--rssm', type=str, default='checkpoints/rssm_rollout.pt')
    p.add_argument('--trajectories', type=str, default='data/trajectories.pkl')
    p.add_argument('--n-goals', type=int, default=4)
    p.add_argument('--sigma', type=float, default=0.5)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--save-path', type=str, default='checkpoints/reward_predictor_rssm.pt')
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--binary', action='store_true', default=True)
    args = p.parse_args()

    train_her_on_rssm(
        rssm_path=args.rssm,
        trajectories_path=args.trajectories,
        n_goals=args.n_goals,
        sigma=args.sigma,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        save_path=args.save_path,
        device_name=args.device,
        binary=args.binary,
    )
