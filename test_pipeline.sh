#!/bin/bash
#SBATCH --job-name=test_pipeline
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mem=32GB
#SBATCH --output=test_pipeline_%j.out

source ~/augenblick/complete_augenblick_setup.sh

echo "Testing imports..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
cd ~/augenblick/src/NeuS2
python -c "import sys; sys.path.insert(0, '.'); import pyngp; print('NeuS2: OK')"
cd ~/augenblick
python -c "from src.vggt import *; print('VGG-T: OK')" 2>/dev/null || echo "VGG-T: Check import path"

echo "Environment ready for pipeline!"
