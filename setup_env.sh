#!/bin/bash

echo "Setting up World Models environment..."

# Create conda environment
conda create -n world-models python=3.11 -y

# Activate environment
source activate world-models || conda activate world-models

# Install PyTorch (adjust for your system)
# For macOS with MPS:
conda install pytorch torchvision -c pytorch -y

# For Linux with CUDA:
# conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
pip install -r requirements.txt

echo ""
echo "Environment setup complete!"
echo "To activate: conda activate world-models"
echo ""
echo "Testing installation..."
python -c "import torch; import gymnasium; import minigrid; print('✓ All packages installed successfully')"
