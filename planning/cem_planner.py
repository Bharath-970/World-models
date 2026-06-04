"""
Cross-Entropy Method (CEM) planner operating in latent space.

Plans action sequences by sampling, rolling out in the world model,
scoring with a learned reward predictor, and refitting to elites.
"""
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.encoder import CNNEncoder
from models.transition import TransitionModel
from training.train_reward_predictor import RewardPredictor


class CEMPlanner:
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

        self.encoder.eval()
        self.transition.eval()
        self.reward.eval()

    @torch.no_grad()
    def plan(self, obs, num_steps=1):
        """
        Plan a short action sequence starting from observation.
        Returns the first `num_steps` actions (1 by default).
        """
        device = next(self.encoder.parameters()).device
        if isinstance(obs, np.ndarray):
            obs_t = torch.FloatTensor(obs).to(device)
        else:
            obs_t = obs.to(device)
        if obs_t.dim() == 3:
            obs_t = obs_t.unsqueeze(0)

        z_current = self.encoder(obs_t)
        plan = []

        for _ in range(num_steps):
            best_actions = self._cem_from(z_current)
            action = int(best_actions[0].item())
            plan.append(action)

            action_t = torch.LongTensor([action]).to(device)
            z_current = self.transition(z_current, action_t)

        return plan

    @torch.no_grad()
    def _cem_from(self, z_start):
        device = next(self.encoder.parameters()).device
        action_mean = np.full((self.horizon, self.action_dim), 1.0 / self.action_dim)
        action_std = np.ones((self.horizon, self.action_dim)) * 0.5

        for _ in range(self.num_iterations):
            samples = np.random.normal(
                action_mean, action_std,
                size=(self.num_samples, self.horizon, self.action_dim),
            )
            samples = np.maximum(samples, 0.0)
            samples = samples / (samples.sum(axis=-1, keepdims=True) + 1e-8)
            sampled_actions = np.argmax(samples, axis=-1)

            scores = np.zeros(self.num_samples, dtype=np.float32)
            for s in range(self.num_samples):
                z = z_start.repeat(self.horizon, 1) if z_start.dim() == 2 else z_start
                if z.dim() == 2 and z.shape[0] == 1:
                    z = z.expand(self.horizon, -1)
                a = torch.LongTensor(sampled_actions[s]).to(device)
                z_traj = self.transition(z, a)
                r = self.reward(z_traj).cpu().numpy()
                discounts = (self.discount ** np.arange(self.horizon))
                scores[s] = float((r * discounts).sum())

            elite_idx = np.argsort(scores)[-self.num_elites:]
            elite_actions = samples[elite_idx]
            action_mean = elite_actions.mean(axis=0)
            action_std = elite_actions.std(axis=0) + 1e-6

        final_mean = np.argmax(action_mean, axis=-1)
        return final_mean

    @torch.no_grad()
    def evaluate_sequence(self, z_start, actions):
        device = next(self.encoder.parameters()).device
        z = z_start.expand(len(actions), -1)
        a = torch.LongTensor(actions).to(device)
        z_traj = self.transition(z, a)
        r = self.reward(z_traj).cpu().numpy()
        return float(r.sum())


def load_planner(model_dir='checkpoints', latent_dim=128, horizon=5,
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

    planner = CEMPlanner(
        encoder=encoder,
        transition=transition,
        reward_predictor=reward,
        horizon=horizon,
        num_samples=num_samples,
        num_elites=num_elites,
        num_iterations=num_iterations,
    )
    return planner, device
