#!/bin/bash
#SBATCH --job-name=neuralangelo_b200
#SBATCH --output=neuralangelo_%j.out
#SBATCH --error=neuralangelo_%j.err
#SBATCH --time=08:00:00              # 8 hours (adjust based on dataset size)
#SBATCH --partition=hpg-b200         # B200 GPU partition
#SBATCH --gres=gpu:b200:1            # Request 1 B200 GPU
#SBATCH --nodes=1                    # Single node
#SBATCH --ntasks=1                   # Single task
#SBATCH --cpus-per-task=16           # 16 CPUs for data loading
#SBATCH --mem=256G                   # 256GB system memory
#SBATCH --mail-type=BEGIN,END,FAIL   # Email notifications
#SBATCH --mail-user=jhennessy7.gatech@gatech.edu  # Update with your email

# Print job information
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "=========================================="

# Print GPU information
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Set up environment
echo "Loading modules..."
module purge
module load pytorch/2.7
module load cuda/12.8.1

# Verify modules
module list

# Activate virtual environment for VGGT
echo "Activating VGGT virtual environment..."
source /home/jhennessy7.gatech/neuralangelo_b200_env/bin/activate

# Verify Python environment
echo "Python environment:"
which python
python --version
echo ""

# Change to augenblick directory
cd /home/jhennessy7.gatech/augenblick

# Set non-interactive mode for batch processing
export NEURALANGELO_BATCH_MODE=1

# Parse command line arguments passed to sbatch
# Usage: sbatch run_neuralangelo.sbatch [--new-dataset --images /path --masks /path] [--use-existing /path]

if [[ "$1" == "--new-dataset" ]]; then
    # Running with new dataset
    echo "Processing new dataset..."
    # Use yes to automatically answer prompts
    yes | ./launch_neuralangelo.sh --dataset "${SLURM_JOB_ID}_dataset" "$@"
    
elif [[ "$1" == "--use-existing" ]]; then
    # Using existing processed data
    echo "Using existing processed data..."
    yes | ./launch_neuralangelo.sh "$@"
    
else
    # Default: use existing scale data
    echo "Using default existing data..."
    yes | ./launch_neuralangelo.sh
fi

# Print completion information
echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=========================================="
