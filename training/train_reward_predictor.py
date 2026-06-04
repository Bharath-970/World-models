"""
Reward predictor: latent z_t -> predicted scalar reward.

Trained on rewards stored in collected trajectories.
Used by the CEM planner to score hypothetical futures.
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

sys.path.append(str(Path(__file__).parent.parent))

from models.encoder import CNNEncoder


class RewardPredictor(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


def train_reward_predictor(
    trajectories_path='data/trajectories.pkl',
    encoder_path='checkpoints/world_model_best.pt',
    latent_dim=128,
    batch_size=256,
    num_epochs=20,
    lr=1e-3,
    save_path='checkpoints/reward_predictor.pt',
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

    obs_list = []
    reward_list = []
    for episode in trajectories:
        for step in episode:
            obs_list.append(step['obs'])
            reward_list.append(float(step.get('reward', 0.0)))

    obs_arr = np.array(obs_list, dtype=np.float32)
    reward_arr = np.array(reward_list, dtype=np.float32)

    if reward_arr.max() > 0:
        reward_arr = reward_arr / max(reward_arr.max(), 1e-6)

    print(f"Total samples: {len(obs_arr)}")
    print(f"Reward stats: mean={reward_arr.mean():.4f} std={reward_arr.std():.4f} max={reward_arr.max():.4f}")

    print(f"Loading encoder from {encoder_path}...")
    encoder = CNNEncoder(input_shape=(7, 7, 3), latent_dim=latent_dim).to(device)
    checkpoint = torch.load(encoder_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    print("Encoding all observations...")
    latents = []
    with torch.no_grad():
        for i in tqdm(range(0, len(obs_arr), batch_size)):
            batch = torch.FloatTensor(obs_arr[i:i + batch_size]).to(device)
            z = encoder(batch)
            latents.append(z.cpu().numpy())
    latents = np.concatenate(latents, axis=0)
    print(f"Latents shape: {latents.shape}")

    latents_t = torch.FloatTensor(latents)
    rewards_t = torch.FloatTensor(reward_arr)

    predictor = RewardPredictor(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(predictor.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n = len(latents_t)
    print(f"\nTraining reward predictor for {num_epochs} epochs...")
    best_loss = float('inf')
    for epoch in range(num_epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            z_batch = latents_t[idx].to(device)
            r_batch = rewards_t[idx].to(device)

            pred = predictor(z_batch)
            loss = criterion(pred, r_batch)

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
                'reward_mean': float(reward_arr.mean()),
                'reward_std': float(reward_arr.std()),
            }, save_path)

    print(f"\nDone. Best loss: {best_loss:.6f}. Saved to {save_path}.")
    return predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectories', type=str, default='data/trajectories.pkl')
    parser.add_argument('--encoder', type=str, default='checkpoints/world_model_best.pt')
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save-path', type=str, default='checkpoints/reward_predictor.pt')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    train_reward_predictor(
        trajectories_path=args.trajectories,
        encoder_path=args.encoder,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        save_path=args.save_path,
        device_name=args.device,
    )
