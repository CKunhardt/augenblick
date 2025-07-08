#!/bin/bash
#SBATCH --job-name=test_neus2
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --mem=16GB

# Load environment
source ~/augenblick/setup_env.sh

# Test NeuS2 import
cd ~/augenblick/src/NeuS2
python -c "import sys; sys.path.insert(0, '.'); import pyngp as ngp; print('NeuS2 imported successfully')"

# Run your actual pipeline here
# python ~/augenblick/src/pipeline/run_pipeline.py ...
