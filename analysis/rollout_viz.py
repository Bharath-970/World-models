"""
Rollout visualization figure.

Shows open-loop RSSM predictions vs ground truth for qualitative comparison.
"""
import torch
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import ImgObsWrapper
from models.rssm import RSSM


def make_rollout_figure(rssm, obs_seq, act_seq, env_name='Empty-5x5',
                        save_path='rollouts_comparison.png'):
    """
    obs_seq: (T, 7, 7, 3) ground truth observations
    act_seq: (T,) actions
    """
    rssm.eval()
    device = next(rssm.parameters()).device
    T = len(obs_seq)

    # Ground truth
    gt_imgs = [(obs_seq[t] * 8).astype(np.uint8) for t in range(T)]

    # Open-loop RSSM rollout
    obs_t = torch.FloatTensor(obs_seq[0:1]).to(device)
    pred_imgs = [gt_imgs[0]]

    with torch.no_grad():
        h, z = rssm.init_state(1, device)
        features = rssm.trunk(obs_t)
        h, z, _, _ = rssm.step(h, z, torch.zeros(1, dtype=torch.long, device=device), features)

        # Store decoded obs at step 0 (posterior)
        x_hat = rssm.decoder(z)
        pred_img = (x_hat[0].cpu().numpy() * 8).clip(0, 8).astype(np.uint8)
        pred_imgs.append(pred_img)

        for t in range(1, T):
            a = torch.LongTensor([int(act_seq[t - 1])]).to(device)
            h, z, _, _ = rssm.step(h, z, a, features=None)
            x_hat = rssm.decoder(z)
            pred_img = (x_hat[0].cpu().numpy() * 8).clip(0, 8).astype(np.uint8)
            pred_imgs.append(pred_img)

    # Create comparison grid
    n_cols = min(T, 8)
    fig, axes = plt.subplots(3, n_cols, figsize=(2.5 * n_cols, 5.5))

    row_labels = ['Ground Truth', 'RSSM Pred', 'Error']
    for row in range(3):
        for col in range(n_cols):
            ax = axes[row, col]
            if row == 0:
                ax.imshow(gt_imgs[col])
                ax.set_title(f'Step {col}')
            elif row == 1:
                ax.imshow(pred_imgs[col])
            else:
                diff = np.abs(gt_imgs[col].astype(float) - pred_imgs[col].astype(float))
                ax.imshow(diff, cmap='hot', vmin=0, vmax=8)
            ax.axis('off')

        axes[row, 0].set_ylabel(row_labels[row], fontsize=12, fontweight='bold')

    plt.suptitle(f'RSSM Open-Loop Rollouts ({env_name})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved rollouts figure to {save_path}")


def run_rollout_viz(rssm_path='checkpoints/rssm_rollout.pt',
                    trajectories_path='data/trajectories.pkl',
                    save_dir='evaluation',
                    device_name='cuda',
                    num_examples=3):
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading RSSM from {rssm_path}...")
    ckpt = torch.load(rssm_path, map_location='cpu')
    rssm = RSSM(
        latent_dim=ckpt['latent_dim'],
        feature_dim=ckpt.get('feature_dim', 256),
        hidden_dim=ckpt.get('hidden_dim', 256),
        action_dim=7,
        overshoot_ks=(1, 3, 5),
    ).to(device)
    rssm.load_state_dict(ckpt['rssm_state_dict'])
    rssm.eval()
    print(f"  latent_dim={ckpt['latent_dim']}")

    print(f"Loading trajectories from {trajectories_path}...")
    with open(trajectories_path, 'rb') as f:
        trajectories = pickle.load(f)
    print(f"  {len(trajectories)} episodes")

    env = gym.make('MiniGrid-Empty-5x5-v0', render_mode=None)
    env = ImgObsWrapper(env)

    rssm_label = Path(rssm_path).stem
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Find successful trajectories with at least 5 steps
    good_eps = [ep for ep in trajectories if len(ep) >= 5]

    for ex_idx in range(min(num_examples, len(good_eps))):
        ep = good_eps[ex_idx]
        ep = ep[:min(len(ep), 8)]  # cap at 8 steps for figure

        # Replay to get consistent observations
        obs_list = []
        obs, _ = env.reset()
        obs_list.append(obs.astype(np.float32) / 8.0)
        for step in ep[:-1]:
            action = step.get('action', 0)
            obs, _, terminated, truncated, _ = env.step(action)
            obs_list.append(obs.astype(np.float32) / 8.0)
            if terminated or truncated:
                break
        T = min(len(obs_list), 8)

        obs_arr = np.stack(obs_list[:T], axis=0)
        act_arr = np.array([int(s.get('action', 0)) for s in ep[:T - 1]])

        save_path = save_dir / f'rollouts_{rssm_label}_ex{ex_idx}.png'
        make_rollout_figure(rssm, obs_arr, act_arr,
                            env_name=f'Empty-5x5 (ex {ex_idx})',
                            save_path=str(save_path))

    env.close()
    print(f"\nDone. {num_examples} rollouts saved to {save_dir}/")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--rssm', type=str, default='checkpoints/rssm_rollout.pt')
    p.add_argument('--trajectories', type=str, default='data/trajectories.pkl')
    p.add_argument('--save-dir', type=str, default='evaluation')
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--num-examples', type=int, default=3)
    args = p.parse_args()

    run_rollout_viz(
        rssm_path=args.rssm,
        trajectories_path=args.trajectories,
        save_dir=args.save_dir,
        device_name=args.device,
        num_examples=args.num_examples,
    )
