"""
Train residual transition model with anti-collapse loss.

The anti-collapse loss has two terms:
1. Standard MSE: predict z_{t+1} accurately
2. Magnitude match: ||pred - z_t|| ≈ ||true - z_t|| (model should
   predict the right change magnitude, not just z_t)

The residual form (z_{t+1} = z_t + scale * Δ) prevents the trivial
identity solution. The magnitude match term ensures the model
actually learns to predict change, not just output z_t.
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

from models.encoder import CNNEncoder
from models.transition_residual import TransitionModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trajectories', type=str, default='data/trajectories.pkl')
    p.add_argument('--encoder', type=str, default='checkpoints/world_model_best.pt')
    p.add_argument('--latent-dim', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--save-path', type=str, default='checkpoints/transition_residual.pt')
    p.add_argument('--magnitude-weight', type=float, default=0.5)
    p.add_argument('--device', type=str, default=None)
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    print(f"Loading encoder from {args.encoder}...")
    encoder = CNNEncoder(input_shape=(7, 7, 3), latent_dim=args.latent_dim).to(device)
    ckpt = torch.load(args.encoder, map_location=device)
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    encoder.eval()
    for p_ in encoder.parameters():
        p_.requires_grad = False

    print(f"Loading trajectories from {args.trajectories}...")
    with open(args.trajectories, 'rb') as f:
        trajectories = pickle.load(f)

    obs_list, obs_next_list, action_list = [], [], []
    for ep in trajectories:
        for i in range(len(ep) - 1):
            obs_list.append(ep[i]['obs'])
            obs_next_list.append(ep[i + 1]['obs'])
            action_list.append(int(ep[i].get('action', 0)))

    obs_arr = np.array(obs_list, dtype=np.float32)
    obs_next_arr = np.array(obs_next_list, dtype=np.float32)
    action_arr = np.array(action_list, dtype=np.int64)

    print(f"Total samples: {len(obs_arr)}")
    print(f"Encoding all observations...")

    def encode(arr):
        out = []
        with torch.no_grad():
            for i in range(0, len(arr), args.batch_size):
                batch = torch.FloatTensor(arr[i:i + args.batch_size]).to(device)
                out.append(encoder(batch).cpu().numpy())
        return np.concatenate(out, axis=0)

    z_t = encode(obs_arr)
    z_tp1 = encode(obs_next_arr)
    print(f"Latents shape: {z_t.shape}")

    z_t_t = torch.FloatTensor(z_t)
    z_tp1_t = torch.FloatTensor(z_tp1)
    a_t = torch.LongTensor(action_arr)

    trans = TransitionModel(latent_dim=args.latent_dim, action_dim=7).to(device)
    opt = optim.Adam(trans.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    n = len(z_t_t)
    best_loss = float('inf')
    print(f"\nTraining residual transition model for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            zt = z_t_t[idx].to(device)
            ztp1_true = z_tp1_t[idx].to(device)
            a = a_t[idx].to(device)

            ztp1_pred = trans(zt, a)

            mse_loss = mse(ztp1_pred, ztp1_true)
            mag_true = (ztp1_true - zt).norm(dim=-1)
            mag_pred = (ztp1_pred - zt).norm(dim=-1)
            mag_loss = mse(mag_pred, mag_true)

            loss = mse_loss + args.magnitude_weight * mag_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / n_batches
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg:.6f}")

        if avg < best_loss:
            best_loss = avg
            Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'transition_state_dict': trans.state_dict(),
                'latent_dim': args.latent_dim,
                'loss': best_loss,
            }, args.save_path)

    print(f"\nDone. Best loss: {best_loss:.6f}. Saved to {args.save_path}.")

    trans.eval()
    with torch.no_grad():
        zt = z_t_t[:1000].to(device)
        ztp1_true = z_tp1_t[:1000].to(device)
        a = a_t[:1000].to(device)
        ztp1_pred = trans(zt, a)
        print(f"\nSanity check on first 1000 samples:")
        print(f"  ||z_t||         = {zt.norm(dim=-1).mean():.4f}")
        print(f"  ||z_t+1 true||  = {ztp1_true.norm(dim=-1).mean():.4f}")
        print(f"  ||z_t+1 pred||  = {ztp1_pred.norm(dim=-1).mean():.4f}")
        print(f"  ||true - z_t||  = {(ztp1_true - zt).norm(dim=-1).mean():.4f}")
        print(f"  ||pred - z_t||  = {(ztp1_pred - zt).norm(dim=-1).mean():.4f}")
        print(f"  ||pred - true|| = {(ztp1_pred - ztp1_true).norm(dim=-1).mean():.4f}")

        per_action = {}
        for aa in range(7):
            mask = a == aa
            if mask.sum() > 0:
                per_action[aa] = {
                    'n': int(mask.sum().item()),
                    'true_change': float((ztp1_true[mask] - zt[mask]).norm(dim=-1).mean()),
                    'pred_change': float((ztp1_pred[mask] - zt[mask]).norm(dim=-1).mean()),
                }
        print(f"  Per-action change magnitude:")
        for aa, stats in per_action.items():
            print(f"    action {aa}: n={stats['n']}, true={stats['true_change']:.4f}, pred={stats['pred_change']:.4f}")


if __name__ == "__main__":
    main()
