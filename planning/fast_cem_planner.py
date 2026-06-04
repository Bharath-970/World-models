"""
Vectorized CEM planner operating in latent space.

Batches all CEM samples and rollout through the transition model in one pass,
giving ~100x speedup over the per-sample loop in cem_planner.py.
"""
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.encoder import CNNEncoder
from models.transition import TransitionModel
from training.train_reward_predictor import RewardPredictor


class FastCEMPlanner:
    def __init__(
        self,
        encoder,
        transition,
        reward_predictor,
        action_dim=7,
        horizon=5,
        num_samples=128,
        num_elites=10,
        num_iterations=4,
        discount=0.99,
    ):
        self.encoder = encoder
        self.transition = transition
        self.reward = reward_predictor
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.discount = discount
        self.discounts = (discount ** np.arange(horizon)).astype(np.float32)
        self.device = next(encoder.parameters()).device

        for m in (encoder, transition, reward_predictor):
            m.eval()

    @torch.no_grad()
    def _rollout_batch(self, z_start, sampled_actions):
        """
        Roll out the world model for `num_samples` parallel action sequences.

        z_start: (1, latent_dim)
        sampled_actions: (num_samples, horizon) int64
        returns: (num_samples,) total discounted reward
        """
        n = sampled_actions.shape[0]
        z = z_start.expand(n, -1)
        rewards = torch.zeros(n, device=self.device)
        for t in range(self.horizon):
            a = sampled_actions[:, t]
            z = self.transition(z, a)
            r = self.reward(z)
            rewards = rewards + r * self.discounts[t]
        return rewards

    @torch.no_grad()
    def plan(self, obs, num_steps=1):
        if isinstance(obs, np.ndarray):
            obs_t = torch.FloatTensor(obs).to(self.device)
        else:
            obs_t = obs.to(self.device)
        if obs_t.dim() == 3:
            obs_t = obs_t.unsqueeze(0)

        z_current = self.encoder(obs_t)
        plan = []

        for _ in range(num_steps):
            best_actions = self._cem_from(z_current)
            action = int(best_actions[0].item())
            plan.append(action)
            a_t = torch.LongTensor([action]).to(self.device)
            z_current = self.transition(z_current, a_t)
        return plan

    @torch.no_grad()
    def _cem_from(self, z_start):
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

            rewards = self._rollout_batch(z_start, actions_t)
            rewards_np = rewards.cpu().numpy()

            elite_idx = np.argsort(rewards_np)[-self.num_elites:]
            elite_actions = samples[elite_idx]
            action_mean = elite_actions.mean(axis=0)
            action_std = elite_actions.std(axis=0) + 1e-6

        final = np.argmax(action_mean, axis=-1)
        return final


def load_fast_planner(model_dir='checkpoints', latent_dim=128, horizon=5,
                      num_samples=128, num_elites=10, num_iterations=4,
                      device_name=None):
    if device_name:
        device = torch.device(device_name)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    encoder = CNNEncoder(input_shape=(7, 7, 3), latent_dim=latent_dim).to(device)
    transition = TransitionModel(latent_dim=latent_dim, action_dim=7).to(device)
    reward = RewardPredictor(latent_dim=latent_dim).to(device)

    model_dir = Path(model_dir)
    wm = torch.load(model_dir / 'world_model_best.pt', map_location=device)
    encoder.load_state_dict(wm['encoder_state_dict'])
    transition.load_state_dict(wm['transition_state_dict'])

    rp = torch.load(model_dir / 'reward_predictor.pt', map_location=device)
    reward.load_state_dict(rp['predictor_state_dict'])

    planner = FastCEMPlanner(
        encoder=encoder,
        transition=transition,
        reward_predictor=reward,
        horizon=horizon,
        num_samples=num_samples,
        num_elites=num_elites,
        num_iterations=num_iterations,
    )
    return planner, device
