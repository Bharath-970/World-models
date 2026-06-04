"""
Train a CNN encoder with a VICReg-style anti-collapse regularizer.

Standard autoencoder reconstruction loss + VICReg terms on the latent:
- Variance: hinge loss forcing each latent dim to have std > 1 across batch
- Covariance: penalty on off-diagonal entries of the latent covariance matrix

This addresses the encoder collapse (latents span 0.01-diameter ball on
random-policy data) without changing the downstream pipeline.
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
from models.decoder import CNNDecoder


def vicreg_loss(z, sim_weight=1.0, var_weight=1.0, cov_weight=1.0, var_target=1.0):
    n, d = z.shape
    std = z.std(dim=0)
    var_loss = torch.mean(torch.relu(var_target - std))
    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / max(n - 1, 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    cov_loss = (off_diag ** 2).sum() / d
    sim_loss = 0.0
    return sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss, {
        'var': float(var_loss),
        'cov': float(cov_loss),
    }


def train_encoder_vicreg(
    trajectories_path='data/trajectories.pkl',
    latent_dim=128,
    batch_size=256,
    num_epochs=20,
    lr=1e-3,
    recon_weight=1.0,
    vicreg_weight=1.0,
    save_path='checkpoints/encoder_vicreg.pt',
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
    for ep in trajectories:
        for step in ep:
            obs_list.append(step['obs'])
    obs_arr = np.array(obs_list, dtype=np.float32)
    print(f"Total observations: {len(obs_arr)}")

    encoder = CNNEncoder(input_shape=(7, 7, 3), latent_dim=latent_dim).to(device)
    decoder = CNNDecoder(latent_dim=latent_dim, output_shape=(7, 7, 3)).to(device)
    opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    mse = nn.MSELoss()

    n = len(obs_arr)
    print(f"\nTraining VICReg encoder for {num_epochs} epochs...")
    best_loss = float('inf')
    for epoch in range(num_epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_vic = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x = torch.FloatTensor(obs_arr[idx]).to(device)
            z = encoder(x)
            x_hat = decoder(z)

            recon = mse(x_hat, x)
            vic, vic_info = vicreg_loss(z, var_weight=1.0, cov_weight=0.1)
            loss = recon_weight * recon + vicreg_weight * vic

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_vic += vic.item()
            n_batches += 1

        avg = epoch_loss / n_batches
        avg_recon = epoch_recon / n_batches
        avg_vic = epoch_vic / n_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg:.4f} (recon={avg_recon:.4f}, vicreg={avg_vic:.4f})")

        if avg < best_loss:
            best_loss = avg
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'latent_dim': latent_dim,
                'loss': best_loss,
            }, save_path)

    print(f"\nDone. Best loss: {best_loss:.4f}. Saved to {save_path}.")

    encoder.eval()
    with torch.no_grad():
        sample = torch.FloatTensor(obs_arr[:1000]).to(device)
        z = encoder(sample)
        print(f"\nLatent stats after training:")
        print(f"  mean norm: {z.norm(dim=-1).mean():.4f}")
        print(f"  per-dim std: mean={z.std(dim=0).mean():.4f}, min={z.std(dim=0).min():.4f}, max={z.std(dim=0).max():.4f}")
        print(f"  pairwise dist: mean={torch.cdist(z, z).mean():.4f}, max={torch.cdist(z, z).max():.4f}")

        x_hat = decoder(z)
        recon_mse = mse(x_hat, sample).item()
        print(f"  recon MSE on 1000 samples: {recon_mse:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectories', type=str, default='data/trajectories.pkl')
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--recon-weight', type=float, default=1.0)
    parser.add_argument('--vicreg-weight', type=float, default=1.0)
    parser.add_argument('--save-path', type=str, default='checkpoints/encoder_vicreg.pt')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    train_encoder_vicreg(
        trajectories_path=args.trajectories,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        recon_weight=args.recon_weight,
        vicreg_weight=args.vicreg_weight,
        save_path=args.save_path,
        device_name=args.device,
    )
