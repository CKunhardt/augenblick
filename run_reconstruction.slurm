#!/bin/bash
#SBATCH --job-name=3d_reconstruction
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --output=reconstruction_%j.out
#SBATCH --error=reconstruction_%j.err

# Print job info
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules
module load conda/25.3.1
module load cuda/12.4.1

# Activate conda environment
conda activate augenblick

# Set CUDA paths
export CUDA_HOME=/apps/compilers/cuda/12.4.1
export PATH=/apps/compilers/cuda/12.4.1/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/cuda/12.4.1/lib64:$LD_LIBRARY_PATH

# Set Python paths
export PYTHONPATH=$PYTHONPATH:~/augenblick/src:~/augenblick/src/NeuS2

# Change to project directory
cd ~/augenblick

# Check GPU availability
echo "Checking GPU..."
nvidia-smi

# Run the complete pipeline
echo "Starting 3D reconstruction pipeline..."

# Process images from scratch
python run_complete_pipeline.py \
    ~/scratch/images_from_dropbox/noscale \
    --output_dir ~/scratch/reconstruction_$(date +%Y%m%d_%H%M%S) \
    --neus2_steps 30000

echo "Job completed on $(date)"
