"""
Goal-conditioned CEM planner operating in latent space.

Uses a GoalConditionedRewardPredictor trained with HER. At each step,
the planner is given z_goal (encoded actual env goal) and rolls out
trajectories in latent space, scoring with the goal-conditioned reward.
"""
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.encoder import CNNEncoder
from models.transition import TransitionModel
from training.train_reward_predictor_her import GoalConditionedRewardPredictor


class GoalConditionedCEMPlanner:
    def __init__(
        self,
        encoder,
        transition,
        reward_predictor,
        z_goal,
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
        self.z_goal = z_goal
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
    def _rollout_batch(self, z_start, sampled_actions, z_goal):
        n = sampled_actions.shape[0]
        z = z_start.expand(n, -1)
        z_g = z_goal.expand(n, -1) if z_goal.dim() == 2 else z_goal.unsqueeze(0).expand(n, -1)
        rewards = torch.zeros(n, device=self.device)
        for t in range(self.horizon):
            a = sampled_actions[:, t]
            z = self.transition(z, a)
            r = self.reward(z, z_g)
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

            rewards = self._rollout_batch(z_start, actions_t, self.z_goal)
            rewards_np = rewards.cpu().numpy()

            elite_idx = np.argsort(rewards_np)[-self.num_elites:]
            elite_actions = samples[elite_idx]
            action_mean = elite_actions.mean(axis=0)
            action_std = elite_actions.std(axis=0) + 1e-6

        final = np.argmax(action_mean, axis=-1)
        return final


def load_goal_conditioned_planner(
    model_dir='checkpoints',
    trajectories_path='data/trajectories.pkl',
    latent_dim=128,
    horizon=5,
    num_samples=128,
    num_elites=10,
    num_iterations=4,
    device_name=None,
    her_predictor_path=None,
):
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
    reward = GoalConditionedRewardPredictor(latent_dim=latent_dim).to(device)

    model_dir = Path(model_dir)
    wm = torch.load(model_dir / 'world_model_best.pt', map_location=device)
    encoder.load_state_dict(wm['encoder_state_dict'])
    transition.load_state_dict(wm['transition_state_dict'])

    rp_path = Path(her_predictor_path) if her_predictor_path else (model_dir / 'reward_predictor_her.pt')
    rp = torch.load(rp_path, map_location=device)
    reward.load_state_dict(rp['predictor_state_dict'])

    from training.train_reward_predictor_her import extract_goal_observation
    z_goal, _ = extract_goal_observation(trajectories_path, model_dir / 'world_model_best.pt', latent_dim, str(device))

    planner = GoalConditionedCEMPlanner(
        encoder=encoder,
        transition=transition,
        reward_predictor=reward,
        z_goal=z_goal,
        horizon=horizon,
        num_samples=num_samples,
        num_elites=num_elites,
        num_iterations=num_iterations,
    )
    return planner, device
