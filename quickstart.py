"""
Quick start script for Phase 1 MVP.

Runs complete pipeline: collect data → train model → visualize.
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    
    if result.returncode != 0:
        print(f"\n✗ Failed: {description}")
        sys.exit(1)
    
    print(f"\n✓ Success: {description}")


def main():
    print("="*60)
    print("World Models - Phase 1 MVP")
    print("="*60)
    
    # Step 1: Collect trajectories
    run_command(
        "python training/collect_trajectories.py --env MiniGrid-Empty-5x5-v0 --episodes 1000",
        "Step 1: Collecting trajectories"
    )
    
    # Step 2: Train world model
    run_command(
        "python training/train_world_model.py --epochs 50 --batch-size 64 --lr 1e-3",
        "Step 2: Training world model"
    )
    
    # Step 3: Visualize latent space
    run_command(
        "python evaluation/visualize_latent.py",
        "Step 3: Visualizing latent space"
    )
    
    print("\n" + "="*60)
    print("✓ Phase 1 MVP Complete!")
    print("="*60)
    print("\nResults:")
    print("  - Trajectories: data/trajectories.pkl")
    print("  - Model: checkpoints/world_model_final.pt")
    print("  - Visualizations: evaluation/results/")
    print("\nNext steps:")
    print("  1. Check visualizations in evaluation/results/")
    print("  2. Verify prediction loss < 0.1")
    print("  3. Check latent space shows structure")
    print("  4. Move to Phase 2 (visual validation)")


if __name__ == "__main__":
    main()
