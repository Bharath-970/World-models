"""
Recurrent State-Space Model (RSSM) — Dreamer-style world model.

Architecture:
    o_t → encoder → features
    h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
    prior:      (μ_p, σ_p) = prior_head(h_t)
    posterior:  (μ_q, σ_q) = posterior_head(features, h_t)
    z_t ~ q(z_t | o_≤t, a_<t)            (uses observation)
    z_t ~ p(z_t | o_<t, a_<t)            (open-loop, no obs)

Why this fixes the open-loop OOD problem:
    The KL(posterior ‖ prior) loss forces the prior to track the
    posterior. When we sample from the prior at inference time (no
    observation available), the rollouts stay in a region the encoder
    has actually produced z's for, so the reward predictor stays valid.

Components:
    - CNN trunk (shared encoder body, no final projection)
    - Prior head: MLP(h_t) → (μ_p, log_std_p)
    - Posterior head: MLP(features, h_t) → (μ_q, log_std_q)
    - Decoder: MLP → deconv to (7,7,3)
    - GRU cell: 1 layer
    - Latent Overshooting heads: MLP(h_t) → (μ_p at t+1, t+3, t+5)

Training losses (always logged separately):
    - recon_loss: MSE(x_hat, x)
    - kl_loss:    KL(q ‖ p) (Dreamer closed-form, summed over latent dims)
    - reward_loss: optional auxiliary (caller adds)
    - overshoot_loss: KL between h_t-derived prior and h_{t+k} posterior
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import CNNEncoder
from .decoder import CNNDecoder


class TrunkEncoder(nn.Module):
    """Shared CNN trunk, outputs a flat feature vector (no latent projection)."""
    def __init__(self, input_shape=(7, 7, 3), feature_dim=256):
        super().__init__()
        self.input_shape = input_shape
        self.feature_dim = feature_dim
        self.conv1 = nn.Conv2d(3, 16, kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0)
        self.flat_size = 64 * 4 * 4
        self.fc = nn.Linear(self.flat_size, feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[-1] == 3 and x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dtype == torch.uint8:
            x = x.float() / 8.0
        elif x.max() > 1.5:
            x = x / 8.0
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def load_from_vicreg(self, vicreg_state_dict):
        """
        Load the conv layers from a VICReg-trained CNNEncoder.
        The VICReg encoder's final fc projects to latent_dim, ours projects to
        feature_dim. They have the same flat_size input so the conv weights
        transfer directly.
        """
        own = self.state_dict()
        loaded = 0
        skipped = 0
        for k, v in vicreg_state_dict.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v)
                loaded += 1
            else:
                skipped += 1
        return loaded, skipped


class RSSM(nn.Module):
    def __init__(
        self,
        input_shape=(7, 7, 3),
        latent_dim=64,
        feature_dim=256,
        hidden_dim=256,
        action_dim=7,
        overshoot_ks=(1, 3, 5),
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.overshoot_ks = overshoot_ks

        self.trunk = TrunkEncoder(input_shape=input_shape, feature_dim=feature_dim)
        self.decoder = CNNDecoder(latent_dim=latent_dim, output_shape=input_shape)

        self.action_embed = nn.Embedding(action_dim, 32)
        self.gru = nn.GRUCell(latent_dim + 32, hidden_dim)

        self.prior_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * latent_dim),
        )
        self.posterior_head = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * latent_dim),
        )

        self.overshoot_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 2 * latent_dim),
            )
            for _ in overshoot_ks
        ])

    @staticmethod
    def _split_params(params):
        mu = params[..., :params.shape[-1] // 2]
        log_std = params[..., params.shape[-1] // 2:]
        return mu, log_std

    def reparameterize(self, mu, log_std):
        std = torch.exp(log_std.clamp(-5.0, 2.0))
        eps = torch.randn_like(std)
        return mu + std * eps

    def init_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        return h, z

    def step(self, h, z_prev, action, features=None):
        """
        One RSSM step.

        If features is provided: use posterior (training with obs).
        If features is None:    use prior     (open-loop rollouts).

        Returns:
            h_new, z_new, prior_params, posterior_params
        """
        a_embed = self.action_embed(action)
        gru_input = torch.cat([z_prev, a_embed], dim=-1)
        h_new = self.gru(gru_input, h)

        prior_params = self.prior_head(h_new)
        mu_p, _ = self._split_params(prior_params)

        if features is not None:
            post_input = torch.cat([features, h_new], dim=-1)
            posterior_params = self.posterior_head(post_input)
            mu_q, log_std_q = self._split_params(posterior_params)
            z_new = self.reparameterize(mu_q, log_std_q)
        else:
            posterior_params = None
            mu_p, log_std_p = self._split_params(prior_params)
            z_new = self.reparameterize(mu_p, log_std_p)

        return h_new, z_new, prior_params, posterior_params

    def rollout_prior(self, h0, z0, actions):
        """
        Open-loop rollouts: h_t, z_t from prior only (no observations).
        Used at inference / planning.

        Args:
            h0: (B, hidden)
            z0: (B, latent)
            actions: (T, B) int64

        Returns:
            hs: (T, B, hidden)
            zs: (T, B, latent)
            prior_mus: (T, B, latent)
            prior_log_stds: (T, B, latent)
        """
        T = actions.shape[0]
        B = h0.shape[0]
        hs, zs, mus, log_stds = [], [], [], []
        h, z = h0, z0
        for t in range(T):
            h, z, prior_params, _ = self.step(h, z, actions[t], features=None)
            mu_p, log_std_p = self._split_params(prior_params)
            hs.append(h)
            zs.append(z)
            mus.append(mu_p)
            log_stds.append(log_std_p)
        return (
            torch.stack(hs, dim=0),
            torch.stack(zs, dim=0),
            torch.stack(mus, dim=0),
            torch.stack(log_stds, dim=0),
        )

    def rollout_overshoot(self, h0, z0, actions):
        """
        Open-loop rollouts with overshooting heads.

        For each k in self.overshoot_ks, predict (mu, log_std) at t+k
        from h_t. Returns stacked tensors of shape (K, T, B, latent).
        """
        T = actions.shape[0]
        K = len(self.overshoot_ks)
        B = h0.shape[0]
        hs_all, zs_all, mus_all, log_stds_all = self.rollout_prior(h0, z0, actions)
        # hs_all: (T, B, hidden)

        K_mus, K_log_stds = [], []
        for k_idx, k in enumerate(self.overshoot_ks):
            h_at_t = hs_all[:-k] if k > 0 else hs_all
            params = self.overshoot_heads[k_idx](h_at_t)
            mu_k, log_std_k = self._split_params(params)
            K_mus.append(mu_k)
            K_log_stds.append(log_std_k)
        # shapes: (K, T-k, B, latent)
        return hs_all, zs_all, mus_all, log_stds_all, K_mus, K_log_stds

    def kl_qp(self, mu_q, log_std_q, mu_p, log_std_p):
        """KL( N(mu_q, sigma_q) || N(mu_p, sigma_p) ) per dim, summed."""
        var_q = torch.exp(2 * log_std_q)
        var_p = torch.exp(2 * log_std_p)
        kl = (
            log_std_p - log_std_q
            + 0.5 * (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-8)
            - 0.5
        )
        return kl.sum(dim=-1)

    def forward_train(self, obs_seq, action_seq):
        """
        Forward pass over a sequence using posterior (training mode).

        Args:
            obs_seq:    (T, B, H, W, C)   float
            action_seq: (T, B)             int64

        Returns dict with all the things needed to compute losses:
            x_hats:    (T, B, H, W, C)
            prior_*:   (T, B, latent)
            post_*:    (T, B, latent)
            zs_post:   (T, B, latent)  sampled from posterior
            hs:        (T, B, hidden)
        """
        T, B = obs_seq.shape[0], obs_seq.shape[1]
        device = obs_seq.device
        h, z_prev = self.init_state(B, device)

        x_hats, prior_mus, prior_log_stds = [], [], []
        post_mus, post_log_stds, zs_post, hs = [], [], [], []

        for t in range(T):
            features = self.trunk(obs_seq[t])
            h, z, prior_params, posterior_params = self.step(
                h, z_prev, action_seq[t], features=features
            )
            mu_p, log_std_p = self._split_params(prior_params)
            mu_q, log_std_q = self._split_params(posterior_params)
            x_hat = self.decoder(z)

            x_hats.append(x_hat)
            prior_mus.append(mu_p)
            prior_log_stds.append(log_std_p)
            post_mus.append(mu_q)
            post_log_stds.append(log_std_q)
            zs_post.append(z)
            hs.append(h)
            z_prev = z

        return {
            'x_hats': torch.stack(x_hats, dim=0),
            'prior_mus': torch.stack(prior_mus, dim=0),
            'prior_log_stds': torch.stack(prior_log_stds, dim=0),
            'post_mus': torch.stack(post_mus, dim=0),
            'post_log_stds': torch.stack(post_log_stds, dim=0),
            'zs_post': torch.stack(zs_post, dim=0),
            'hs': torch.stack(hs, dim=0),
        }


def compute_rssm_losses(out, obs_seq, beta=0.01, overshoot_targets=None, overshoot_weight=0.1):
    """
    Compute the three losses, always returned separately so we can log them.

    Args:
        out: dict from RSSM.forward_train
        obs_seq: (T, B, H, W, C)
        beta: KL weight (start at 0.01 per project convention)
        overshoot_targets: dict from RSSM.rollout_overshoot
            {'hs_prior': (T,B,H), 'zs_prior': (T,B,L),
             'prior_mus': (T,B,L), 'prior_log_stds': (T,B,L),
             'overshoot_mus': list of (T-k, B, L),
             'overshoot_log_stds': list of (T-k, B, L)}
            if None, overshoot loss is skipped
        overshoot_weight: scaling on overshoot loss

    Returns:
        total_loss, info dict
    """
    x_hats = out['x_hats']
    prior_mus = out['prior_mus']
    prior_log_stds = out['prior_log_stds']
    post_mus = out['post_mus']
    post_log_stds = out['post_log_stds']
    zs_post = out['zs_post']

    recon_loss = F.mse_loss(x_hats, obs_seq)

    kl_per_step = 0.5 * (
        2 * (post_log_stds - prior_log_stds)
        + (torch.exp(2 * prior_log_stds) + (prior_mus - post_mus) ** 2)
        / (torch.exp(2 * post_log_stds) + 1e-8)
        - 1.0
    ).sum(dim=-1)
    kl_loss = kl_per_step.mean()

    info = {
        'recon': float(recon_loss.item()),
        'kl': float(kl_loss.item()),
        'kl_per_dim': float((kl_per_step / max(zs_post.shape[-1], 1)).mean().item()),
        'prior_std_mean': float(prior_log_stds.exp().mean().item()),
        'post_std_mean': float(post_log_stds.exp().mean().item()),
    }

    total = recon_loss + beta * kl_loss

    if overshoot_targets is not None:
        post_mus_seq = out['post_mus']            # (T, B, L)
        post_log_stds_seq = out['post_log_stds']  # (T, B, L)
        K_mus = overshoot_targets['overshoot_mus']
        K_log_stds = overshoot_targets['overshoot_log_stds']
        K = len(K_mus)
        ov_loss = 0.0
        for k_idx, k in enumerate([1, 3, 5][:K]):
            mu_k = K_mus[k_idx]                  # (T-k, B, L)
            log_std_k = K_log_stds[k_idx]        # (T-k, B, L)
            mu_target = post_mus_seq[k:]          # (T-k, B, L)
            log_std_target = post_log_stds_seq[k:]
            kl_k = 0.5 * (
                2 * (log_std_target - log_std_k)
                + (torch.exp(2 * log_std_k) + (mu_k - mu_target) ** 2)
                / (torch.exp(2 * log_std_target) + 1e-8)
                - 1.0
            ).sum(dim=-1).mean()
            ov_loss = ov_loss + kl_k
        ov_loss = ov_loss / max(K, 1)
        total = total + overshoot_weight * ov_loss
        info['overshoot'] = float(ov_loss.item())

    return total, info
