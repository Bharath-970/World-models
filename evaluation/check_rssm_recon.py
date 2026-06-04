"""
Quick visual sanity check: are the reconstructions all-yellow?
"""
import torch
import numpy as np
import pickle
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from models.rssm import RSSM
import torch.nn.functional as F


def kl_to_standard_normal(mu, log_std):
    return 0.5 * (mu ** 2 + torch.exp(2 * log_std) - 1.0 - 2 * log_std).sum(dim=-1)


def main():
    device = 'mps'
    ckpt = torch.load('checkpoints/rssm_ae.pt', map_location=device)
    rssm = RSSM(
        latent_dim=ckpt['latent_dim'],
        feature_dim=ckpt['feature_dim'],
        hidden_dim=ckpt['hidden_dim'],
    ).to(device)
    rssm.load_state_dict(ckpt['rssm_state_dict'])
    rssm.eval()

    with open('data/trajectories.pkl', 'rb') as f:
        trajs = pickle.load(f)

    rng = np.random.default_rng(0)
    obs = []
    chosen = rng.choice(len(trajs), 8, replace=False)
    for ep_idx in chosen:
        ep = trajs[ep_idx]
        t = rng.integers(0, len(ep))
        obs.append(ep[t]['obs'])
    obs = np.array(obs, dtype=np.float32) / 8.0
    x = torch.FloatTensor(obs).to(device)

    with torch.no_grad():
        features = rssm.trunk(x)
        h = torch.zeros(x.shape[0], rssm.hidden_dim, device=device)
        post_input = torch.cat([features, h], dim=-1)
        post_params = rssm.posterior_head(post_input)
        mu_q, log_std_q = rssm._split_params(post_params)
        z = mu_q  # use mean for deterministic decode
        z_sampled = rssm.reparameterize(mu_q, log_std_q)

        x_hat_mean = rssm.decoder(z)
        x_hat_sampled = rssm.decoder(z_sampled)
        recon_mse_mean = float(F.mse_loss(x_hat_mean, x).item())
        recon_mse_sampled = float(F.mse_loss(x_hat_sampled, x).item())
        kl = float(kl_to_standard_normal(mu_q, log_std_q).mean().item())
        z_std_per_dim = float(mu_q.std(dim=0).mean().item())
        z_pair_dist = float(torch.cdist(mu_q, mu_q).mean().item())

    print(f"Recon MSE (mean):  {recon_mse_mean:.4f}")
    print(f"Recon MSE (sample):{recon_mse_sampled:.4f}")
    print(f"KL to N(0,I):      {kl:.4f}")
    print(f"z std per-dim:     {z_std_per_dim:.4f}")
    print(f"z pairwise dist:   {z_pair_dist:.4f}")

    fig, axes = plt.subplots(8, 2, figsize=(3, 12))
    for i in range(8):
        axes[i, 0].imshow(obs[i].clip(0, 1))
        axes[i, 0].set_title('true', fontsize=8)
        axes[i, 1].imshow(x_hat_mean[i].cpu().numpy().clip(0, 1))
        axes[i, 1].set_title('recon', fontsize=8)
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.savefig('evaluation/rssm_ae_recon.png', dpi=120)
    print("Saved evaluation/rssm_ae_recon.png")


if __name__ == "__main__":
    main()
