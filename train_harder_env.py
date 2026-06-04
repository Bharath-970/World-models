"""
Train world model on a harder environment.

Tests if the model generalizes to more complex tasks.
"""
import subprocess
import sys
from pathlib import Path


def build_python_command(script_path, *args):
    """Build a subprocess command that reuses the active Python interpreter."""
    return [sys.executable, script_path, *[str(arg) for arg in args]]


def run_command(command):
    """Run a subprocess command and return the completed process."""
    return subprocess.run(command, check=False)


def train_harder_env(
    env_name='MiniGrid-FourRooms-v0',
    episodes=2000,
    epochs=50,
    trajectories_path='data/trajectories_hard.pkl',
    save_dir='checkpoints_hard',
    results_dir='evaluation/results_hard',
    resume_from=None,
    skip_collection=False,
):
    """
    Train world model on a harder environment.
    
    Args:
        env_name: Name of the harder environment
        episodes: Number of episodes to collect
        epochs: Number of training epochs
    """
    print("="*60)
    print(f"Training World Model on Harder Environment")
    print(f"Environment: {env_name}")
    print("="*60)
    
    trajectories_file = Path(trajectories_path)
    should_collect = not skip_collection and not (resume_from and trajectories_file.exists())

    # Step 1: Collect trajectories
    if should_collect:
        print("\n[1/3] Collecting trajectories...")
        cmd = build_python_command(
            "training/collect_trajectories.py",
            "--env", env_name,
            "--episodes", episodes,
            "--save-path", trajectories_path,
        )
        result = run_command(cmd)
    else:
        print("\n[1/3] Reusing existing trajectories...")
        if not trajectories_file.exists():
            print(f"✗ Expected trajectories at {trajectories_path}, but none were found")
            sys.exit(1)
        result = subprocess.CompletedProcess(args=[], returncode=0)
    
    if result.returncode != 0:
        print("✗ Failed to collect trajectories")
        sys.exit(1)
    
    # Step 2: Train world model
    print("\n[2/3] Training world model...")
    cmd = build_python_command(
        "training/train_world_model.py",
        "--trajectories", trajectories_path,
        "--epochs", epochs,
        "--save-dir", save_dir,
    )
    if resume_from:
        cmd.extend(["--resume-from", str(resume_from)])
    result = run_command(cmd)
    
    if result.returncode != 0:
        print("✗ Failed to train world model")
        sys.exit(1)
    
    # Step 3: Visualize
    print("\n[3/3] Visualizing results...")
    cmd = build_python_command(
        "evaluation/visualize_latent.py",
        "--model", str(Path(save_dir) / "world_model_best.pt"),
        "--trajectories", trajectories_path,
        "--save-dir", results_dir,
    )
    result = run_command(cmd)
    
    if result.returncode != 0:
        print("✗ Failed to visualize")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✓ Training on harder environment complete!")
    print("="*60)
    print(f"\nResults saved to:")
    print(f"  - Model: checkpoints_hard/")
    print(f"  - Visualizations: evaluation/results_hard/")
    print(f"\nCompare with easy environment results in:")
    print(f"  - evaluation/results/ (easy)")
    print(f"  - evaluation/results_hard/ (hard)")


if __name__ == "__main__":
    # You can change the environment here
    # Options: MiniGrid-FourRooms-v0, MiniGrid-DoorKey-v0, MiniGrid-Memory-v0
    train_harder_env(
        env_name='MiniGrid-FourRooms-v0',
        episodes=2000,
        epochs=50,
        resume_from='checkpoints_hard/checkpoint_epoch_10.pt',
    )
