"""
Goal-conditioned reward predictor for HER-style hindsight relabelling.

Predicts reward from (z_t, z_goal) using a soft distance-based signal:
  reward = exp(-||z_t - z_goal||^2 / sigma^2)

This converts the sparse-reward problem into a dense one without any
extra environment interaction: every state in every trajectory can be
paired with any other state in the same trajectory as a "fake goal",
producing dense training signal.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from models.encoder import CNNEncoder


class GoalConditionedRewardPredictor(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256):
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


def her_relabel(trajectories, n_goals=4, strategy='future', sigma=1.0):
    """
    Hindsight relabelling: for each timestep, pair it with K fake goals
    sampled from the same episode.

    Reward = exp(-||z_t - z_goal||^2 / sigma^2) using a soft distance
    proxy in observation space (we don't have z here, so we use the
    mean L2 distance in observation space as a stand-in; the predictor
    is trained on these soft labels).

    Returns list of (obs, goal_obs, reward) tuples.
    """
    samples = []
    for ep in trajectories:
        obs_list = [s['obs'] for s in ep]
        L = len(obs_list)
        for t in range(L):
            obs_t = obs_list[t]
            if strategy == 'future':
                goal_idxs = np.random.randint(t, L, size=n_goals)
            elif strategy == 'final':
                goal_idxs = [L - 1] * n_goals
            elif strategy == 'random':
                goal_idxs = np.random.randint(0, L, size=n_goals)
            else:
                raise ValueError(strategy)
            for g_idx in goal_idxs:
                obs_g = obs_list[g_idx]
                obs_t_flat = obs_t.flatten()
                obs_g_flat = obs_g.flatten()
                dist = np.mean((obs_t_flat - obs_g_flat) ** 2)
                r = float(np.exp(-dist / (sigma ** 2 + 1e-6)))
                samples.append((obs_t, obs_g, r))
    return samples


def train_her_reward_predictor(
    trajectories_path='data/trajectories.pkl',
    encoder_path='checkpoints/world_model_best.pt',
    latent_dim=128,
    batch_size=256,
    num_epochs=20,
    lr=1e-3,
    save_path='checkpoints/reward_predictor_her.pt',
    n_goals=4,
    sigma=1.0,
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

    print(f"HER relabelling with n_goals={n_goals}, sigma={sigma}...")
    samples = her_relabel(trajectories, n_goals=n_goals, strategy='future', sigma=sigma)
    print(f"Generated {len(samples)} (obs, goal, reward) pairs")

    obs_arr = np.array([s[0] for s in samples], dtype=np.float32)
    goal_arr = np.array([s[1] for s in samples], dtype=np.float32)
    reward_arr = np.array([s[2] for s in samples], dtype=np.float32)
    print(f"Reward stats: mean={reward_arr.mean():.4f} std={reward_arr.std():.4f} max={reward_arr.max():.4f}")

    print(f"Loading encoder from {encoder_path}...")
    encoder = CNNEncoder(input_shape=(7, 7, 3), latent_dim=latent_dim).to(device)
    checkpoint = torch.load(encoder_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    def encode_batch(arr):
        out = []
        with torch.no_grad():
            for i in range(0, len(arr), batch_size):
                batch = torch.FloatTensor(arr[i:i + batch_size]).to(device)
                z = encoder(batch)
                out.append(z.cpu().numpy())
        return np.concatenate(out, axis=0)

    print("Encoding all observations...")
    z_t = encode_batch(obs_arr)
    print("Encoding all goal observations...")
    z_goal = encode_batch(goal_arr)
    print(f"Latents shape: {z_t.shape}")

    z_t_t = torch.FloatTensor(z_t)
    z_g_t = torch.FloatTensor(z_goal)
    r_t = torch.FloatTensor(reward_arr)

    predictor = GoalConditionedRewardPredictor(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(predictor.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n = len(z_t_t)
    print(f"\nTraining HER reward predictor for {num_epochs} epochs...")
    best_loss = float('inf')
    for epoch in range(num_epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            zt = z_t_t[idx].to(device)
            zg = z_g_t[idx].to(device)
            r = r_t[idx].to(device)

            pred = predictor(zt, zg)
            loss = criterion(pred, r)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / n_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg:.6f}")

        if avg < best_loss:
            best_loss = avg
            torch.save({
                'predictor_state_dict': predictor.state_dict(),
                'latent_dim': latent_dim,
                'loss': best_loss,
            }, save_path)

    print(f"\nDone. Best loss: {best_loss:.6f}. Saved to {save_path}.")
    return predictor


def extract_goal_observation(trajectories_path, encoder_path, latent_dim=128, device_name=None):
    """Pick the first terminal observation from a successful trajectory and encode it."""
    if device_name:
        device = torch.device(device_name)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    with open(trajectories_path, 'rb') as f:
        trajectories = pickle.load(f)

    for ep in trajectories:
        for step in ep:
            if step.get('reward', 0) > 0:
                goal_obs = step['obs']
                break
        if any(s.get('reward', 0) > 0 for s in ep):
            break
    else:
        raise RuntimeError(f"No successful trajectory in {trajectories_path}")

    encoder = CNNEncoder(input_shape=(7, 7, 3), latent_dim=latent_dim).to(device)
    checkpoint = torch.load(encoder_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()

    with torch.no_grad():
        z_goal = encoder(torch.FloatTensor(goal_obs).unsqueeze(0).to(device))

    return z_goal, goal_obs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectories', type=str, default='data/trajectories.pkl')
    parser.add_argument('--encoder', type=str, default='checkpoints/world_model_best.pt')
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save-path', type=str, default='checkpoints/reward_predictor_her.pt')
    parser.add_argument('--n-goals', type=int, default=4)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    train_her_reward_predictor(
        trajectories_path=args.trajectories,
        encoder_path=args.encoder,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        save_path=args.save_path,
        n_goals=args.n_goals,
        sigma=args.sigma,
        device_name=args.device,
    )
