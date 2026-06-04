"""
Test script to verify installation and basic functionality.
"""
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

print("Testing World Models Installation...")
print("="*60)

# Test 1: PyTorch
print("\n1. Testing PyTorch...")
try:
    x = torch.randn(2, 3)
    print(f"   ✓ PyTorch {torch.__version__}")
    print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
    if torch.backends.mps.is_available():
        print(f"   ✓ MPS available (Apple Silicon)")
except Exception as e:
    print(f"   ✗ PyTorch failed: {e}")
    sys.exit(1)

# Test 2: Gymnasium + MiniGrid
print("\n2. Testing Gymnasium + MiniGrid...")
try:
    import gymnasium as gym
    import minigrid
    env = gym.make('MiniGrid-Empty-5x5-v0', render_mode='rgb_array')
    obs, info = env.reset()
    print(f"   ✓ Gymnasium {gym.__version__}")
    print(f"   ✓ MiniGrid loaded")
    print(f"   ✓ Observation shape: {obs['image'].shape}")
    env.close()
except Exception as e:
    print(f"   ✗ Gymnasium/MiniGrid failed: {e}")
    sys.exit(1)

# Test 3: Models
print("\n3. Testing model architectures...")
try:
    from models.encoder import CNNEncoder
    from models.transition import TransitionModel
    from models.decoder import CNNDecoder
    
    encoder = CNNEncoder(input_shape=(7, 7, 3), latent_dim=128)
    transition = TransitionModel(latent_dim=128, action_dim=7)
    decoder = CNNDecoder(latent_dim=128, output_shape=(7, 7, 3))
    
    # Test forward pass
    obs = torch.randn(2, 7, 7, 3)
    z = encoder(obs)
    action = torch.randint(0, 7, (2,))
    z_next = transition(z, action)
    obs_recon = decoder(z)
    
    print(f"   ✓ Encoder: {obs.shape} → {z.shape}")
    print(f"   ✓ Transition: (z, action) → z_next {z_next.shape}")
    print(f"   ✓ Decoder: {z.shape} → {obs_recon.shape}")
    
    # Count parameters
    enc_params = sum(p.numel() for p in encoder.parameters())
    trans_params = sum(p.numel() for p in transition.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    
    print(f"   ✓ Encoder params: {enc_params:,}")
    print(f"   ✓ Transition params: {trans_params:,}")
    print(f"   ✓ Decoder params: {dec_params:,}")
    
except Exception as e:
    print(f"   ✗ Models failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Visualization libraries
print("\n4. Testing visualization libraries...")
try:
    import matplotlib
    import sklearn
    print(f"   ✓ Matplotlib {matplotlib.__version__}")
    print(f"   ✓ Scikit-learn {sklearn.__version__}")
except Exception as e:
    print(f"   ✗ Visualization libraries failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✓ All tests passed!")
print("="*60)
print("\nReady to run Phase 1:")
print("  python quickstart.py")
print("\nOr run steps individually:")
print("  python training/collect_trajectories.py")
print("  python training/train_world_model.py")
print("  python evaluation/visualize_latent.py")
