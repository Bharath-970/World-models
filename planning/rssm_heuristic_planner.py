"""
Diagnostic: RSSM planner with HEURISTIC reward (pixel-space distance to goal).

Bypasses the learned reward predictor entirely. Computes reward as
exp(-||decoded_pred - obs_goal||^2 / sigma^2) where decoded_pred is the
decoder output of the prior rollout.

This is a controlled ablation to isolate: is the world model good enough
for planning, or is the learned reward predictor the bottleneck?
"""
import torch
import numpy as np
import argparse
import sys
from pathlib import Path
import json
import gymnasium as gym
import minigrid

sys.path.append(str(Path(__file__).parent.parent))

from models.rssm import RSSM


def obs_distance_reward(z_pred, z_goal, decoder, obs_goal, sigma=0.1):
    """
    Heuristic reward: decode both z's, compute MSE in pixel space,
    convert to dense reward via Gaussian.

    obs_goal is the raw obs (7, 7, 3) in [0, 1] (normalized).
    """
    with torch.no_grad():
        x_pred = decoder(z_pred)
        x_goal = decoder(z_goal)
        if x_goal.dim() == 3:
            x_goal = x_goal.unsqueeze(0)
        dist = ((x_pred - x_goal) ** 2).mean(dim=(-1, -2, -3))
        return torch.exp(-dist / (sigma ** 2))


class RSSMHeuristicPlanner:
    """
    CEM planner using the RSSM world model + pixel-space heuristic reward.

    No learned reward predictor — pure world model + decoder.
    """
    def __init__(
        self,
        rssm,
        obs_goal,
        action_dim=7,
        horizon=4,
        num_samples=128,
        num_elites=10,
        num_iterations=4,
        discount=0.99,
        sigma=0.1,
    ):
        self.rssm = rssm
        self.obs_goal = obs_goal
        self.sigma = sigma
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.discount = discount
        self.discounts = (discount ** np.arange(horizon)).astype(np.float32)
        self.device = next(rssm.parameters()).device

        # Encode goal obs once
        with torch.no_grad():
            x_g = torch.FloatTensor(obs_goal).to(self.device)
            if x_g.dim() == 3:
                x_g = x_g.unsqueeze(0)
            if x_g.max() > 1.5:
                x_g = x_g / 8.0
            features_g = self.rssm.trunk(x_g)
            h_g = torch.zeros(1, self.rssm.hidden_dim, device=self.device)
            post_input = torch.cat([features_g, h_g], dim=-1)
            post_params = self.rssm.posterior_head(post_input)
            mu_q, _ = self.rssm._split_params(post_params)
            self.z_goal = mu_q

        self.rssm.eval()

    @torch.no_grad()
    def _initial_state(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        if obs.max() > 1.5:
            obs = obs / 8.0
        features = self.rssm.trunk(obs)
        h = torch.zeros(1, self.rssm.hidden_dim, device=self.device)
        post_input = torch.cat([features, h], dim=-1)
        post_params = self.rssm.posterior_head(post_input)
        mu_q, _ = self.rssm._split_params(post_params)
        z = mu_q
        return h, z

    @torch.no_grad()
    def _rollout_batch(self, h_start, z_start, sampled_actions):
        n = sampled_actions.shape[0]
        h = h_start.expand(n, -1)
        z = z_start.expand(n, -1)
        total_reward = torch.zeros(n, device=self.device)
        for t in range(self.horizon):
            a = sampled_actions[:, t]
            h, z, _, _ = self.rssm.step(h, z, a, features=None)
            r = obs_distance_reward(z, self.z_goal, self.rssm.decoder, self.obs_goal, self.sigma)
            total_reward = total_reward + r * float(self.discounts[t])
        return total_reward

    @torch.no_grad()
    def plan(self, obs, num_steps=1):
        h, z = self._initial_state(obs)
        plan = []
        for _ in range(num_steps):
            best_actions = self._cem_from(h, z)
            action = int(best_actions[0].item())
            plan.append(action)
            a_t = torch.LongTensor([action]).to(self.device)
            h, z, _, _ = self.rssm.step(h, z, a_t, features=None)
        return plan

    @torch.no_grad()
    def _cem_from(self, h_start, z_start):
        action_mean = np.full((self.horizon, self.action_dim), 1.0 / self.action_dim, dtype=np.float32)
        action_std = np.ones((self.horizon, self.action_dim), dtype=np.float32) * 0.5

        for _ in range(self.num_iterations):
            samples = np.random.normal(
                action_mean, action_std,
                size=(self.num_samples, self.horizon, self.action_dim),
            ).astype(np.float32)
            samples = np.maximum(samples, 0.0)
            samples_sum = samples.sum(axis=-1, keepdims=True) + 1e-8
            samples = samples / samples_sum
            sampled_actions = np.argmax(samples, axis=-1).astype(np.int64)
            actions_t = torch.from_numpy(sampled_actions).to(self.device)

            rewards = self._rollout_batch(h_start, z_start, actions_t)
            rewards_np = rewards.cpu().numpy()

            elite_idx = np.argsort(rewards_np)[-self.num_elites:]
            elite_actions = samples[elite_idx]
            action_mean = elite_actions.mean(axis=0)
            action_std = elite_actions.std(axis=0) + 1e-6

        final = np.argmax(action_mean, axis=-1)
        return final


def load_heuristic_planner(
    rssm_path='checkpoints/rssm_rollout.pt',
    trajectories_path='data/trajectories.pkl',
    horizon=4,
    num_samples=128,
    num_elites=10,
    num_iterations=4,
    discount=0.99,
    sigma=0.1,
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

    rssm_ckpt = torch.load(rssm_path, map_location=device)
    rssm = RSSM(
        latent_dim=rssm_ckpt['latent_dim'],
        feature_dim=rssm_ckpt['feature_dim'],
        hidden_dim=rssm_ckpt['hidden_dim'],
    ).to(device)
    rssm.load_state_dict(rssm_ckpt['rssm_state_dict'])

    import pickle
    with open(trajectories_path, 'rb') as f:
        trajectories = pickle.load(f)
    for ep in trajectories:
        for step in ep:
            if step.get('reward', 0) > 0:
                obs_goal = step['obs']
                break
        else:
            continue
        break
    else:
        raise ValueError("No successful trajectory found")

    return RSSMHeuristicPlanner(
        rssm=rssm,
        obs_goal=obs_goal,
        action_dim=7,
        horizon=horizon,
        num_samples=num_samples,
        num_elites=num_elites,
        num_iterations=num_iterations,
        discount=discount,
        sigma=sigma,
    ), device
