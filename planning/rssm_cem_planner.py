"""
RSSM goal-conditioned CEM planner.

Operates in the RSSM's latent space. The transition model is the
RSSM prior (h_t, z_t, a_t) → (h_{t+1}, z_{t+1}).

Initial state:
    h_0 = 0
    z_0 = q(z_0 | o_0)              (posterior with h=0)

Planning:
    For each candidate action sequence, roll out via the prior and
    score with the goal-conditioned reward predictor at each step.

CEM: sample action distributions, fit to elites, resample.
"""
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.rssm import RSSM


class RSSMCEMPlanner:
    def __init__(
        self,
        rssm,
        reward_predictor,
        z_goal,
        action_dim=7,
        horizon=5,
        num_samples=128,
        num_elites=10,
        num_iterations=4,
        discount=0.99,
    ):
        self.rssm = rssm
        self.reward = reward_predictor
        self.z_goal = z_goal
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.discount = discount
        self.discounts = (discount ** np.arange(horizon)).astype(np.float32)
        self.device = next(rssm.parameters()).device

        self.rssm.eval()
        self.reward.eval()

    @torch.no_grad()
    def _initial_state(self, obs):
        """
        obs: (7, 7, 3) or (1, 7, 7, 3) numpy or tensor in [0, 255] (uint8).
        Returns (h, z) for rollout.
        """
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
        """
        sampled_actions: (num_samples, horizon) int64
        Returns: (num_samples,) total discounted reward.
        """
        n = sampled_actions.shape[0]
        h = h_start.expand(n, -1)
        z = z_start.expand(n, -1)
        z_g = self.z_goal.expand(n, -1) if self.z_goal.dim() == 2 else self.z_goal.unsqueeze(0).expand(n, -1)
        total_reward = torch.zeros(n, device=self.device)
        for t in range(self.horizon):
            a = sampled_actions[:, t]
            h, z, _, _ = self.rssm.step(h, z, a, features=None)
            r = self.reward(z, z_g)
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


def extract_rssm_goal_observation(trajectories_path, rssm_path, latent_dim, device_name=None):
    """
    Pick a successful terminal observation and encode it through the RSSM
    posterior to get z_goal.
    """
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
    rssm.eval()

    import pickle
    with open(trajectories_path, 'rb') as f:
        trajectories = pickle.load(f)

    for ep in trajectories:
        for step in ep:
            if step.get('reward', 0) > 0:
                goal_obs = step['obs']
                break
        else:
            continue
        break
    else:
        raise ValueError("No successful trajectory found")

    with torch.no_grad():
        x = torch.FloatTensor(goal_obs).to(device) / 8.0
        if x.dim() == 3:
            x = x.unsqueeze(0)
        features = rssm.trunk(x)
        h = torch.zeros(1, rssm.hidden_dim, device=device)
        post_input = torch.cat([features, h], dim=-1)
        post_params = rssm.posterior_head(post_input)
        mu_q, _ = rssm._split_params(post_params)
        z_goal = mu_q

    return z_goal, goal_obs


def load_rssm_planner(
    rssm_path='checkpoints/rssm_rollout.pt',
    reward_predictor_path='checkpoints/reward_predictor_rssm_bin.pt',
    trajectories_path='data/trajectories.pkl',
    horizon=5,
    num_samples=128,
    num_elites=10,
    num_iterations=4,
    discount=0.99,
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

    from training.train_her_reward_rssm import HERRewardHead
    rp_ckpt = torch.load(reward_predictor_path, map_location=device)
    reward = HERRewardHead(latent_dim=rp_ckpt['latent_dim']).to(device)
    reward.load_state_dict(rp_ckpt['predictor_state_dict'])

    z_goal, _ = extract_rssm_goal_observation(
        trajectories_path, rssm_path, rssm_ckpt['latent_dim'], str(device),
    )

    planner = RSSMCEMPlanner(
        rssm=rssm,
        reward_predictor=reward,
        z_goal=z_goal,
        action_dim=7,
        horizon=horizon,
        num_samples=num_samples,
        num_elites=num_elites,
        num_iterations=num_iterations,
        discount=discount,
    )
    return planner, device
