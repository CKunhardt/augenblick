#!/bin/bash
#SBATCH --job-name=nerf_full_pipeline
#SBATCH --output=pipeline_%j.out
#SBATCH --error=pipeline_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=hpg-b200         # B200 GPU partition
#SBATCH --gres=gpu:b200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jhennessy7.gatech@gatech.edu

# This script properly manages environments for both VGGT and Neuralangelo

echo "=========================================="
echo "Full Neuralangelo Pipeline"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "=========================================="

# Function to setup VGGT environment
setup_vggt_env() {
    echo "Setting up VGGT environment..."
    # Deactivate any current environment
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate 2>/dev/null || true
    fi
    
    # Load base modules for VGGT
    module purge
    module load cuda/12.8.1
    
    # Activate VGGT virtual environment
    source /home/jhennessy7.gatech/neuralangelo_b200_env/bin/activate
    
    echo "VGGT Environment:"
    echo "  Python: $(which python)"
    echo "  Python version: $(python --version)"
    echo "  Virtual env: $VIRTUAL_ENV"
    
    # Verify VGGT packages
    python -c "import vggt; print('  VGGT available')" 2>/dev/null || echo "  Warning: VGGT not found"
}

# Function to setup Neuralangelo environment
setup_neuralangelo_env() {
    echo "Setting up Neuralangelo environment..."
    # Deactivate virtual environment
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate
    fi
    
    # Load modules for Neuralangelo
    module purge
    module load pytorch/2.7
    module load cuda/12.8.1
    
    echo "Neuralangelo Environment:"
    echo "  Python: $(which python)"
    echo "  Python version: $(python --version)"
    module list | grep -E "(pytorch|cuda)"
}

# Parse arguments
PIPELINE_MODE="$1"
shift

cd /home/jhennessy7.gatech/augenblick

case "$PIPELINE_MODE" in
    "--new-dataset")
        echo "Mode: Processing new dataset with VGGT + Neuralangelo"
        
        # First run VGGT pipeline with proper environment
        setup_vggt_env
        
        # Create temporary script that handles environment switching
        cat > /tmp/pipeline_${SLURM_JOB_ID}.sh << 'EOF'
#!/bin/bash
# This subscript manages environment switching

# Function to run VGGT part
run_vggt_part() {
    source /home/jhennessy7.gatech/neuralangelo_b200_env/bin/activate
    cd /home/jhennessy7.gatech/augenblick
    
    # Run only the VGGT pipeline part
    python prep_crop.py "$@"
    python vggt_batch_final.py "$@"
    python scale_transforms_to_original.py "$@"
}

# Function to run Neuralangelo part
run_neuralangelo_part() {
    deactivate 2>/dev/null || true
    module purge
    module load pytorch/2.7 cuda/12.8.1
    cd /home/jhennessy7.gatech/augenblick
    
    # Continue with Neuralangelo setup and training
    yes | ./launch_neuralangelo.sh "$@"
}

# Export functions
export -f run_vggt_part
export -f run_neuralangelo_part
EOF
        
        chmod +x /tmp/pipeline_${SLURM_JOB_ID}.sh
        
        # Run the full pipeline with proper environment management
        yes | ./launch_neuralangelo.sh --dataset "${SLURM_JOB_ID}_dataset" "$@"
        
        # Cleanup
        rm -f /tmp/pipeline_${SLURM_JOB_ID}.sh
        ;;
        
    "--use-existing")
        echo "Mode: Using existing data (Neuralangelo only)"
        setup_neuralangelo_env
        yes | ./launch_neuralangelo.sh "$@"
        ;;
        
    *)
        echo "Mode: Default (using existing scale data)"
        setup_neuralangelo_env
        yes | ./launch_neuralangelo.sh
        ;;
esac

# Print completion information
echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=========================================="

# If job completed successfully, show where outputs are
if [ $? -eq 0 ]; then
    echo "Success! Check outputs in:"
    find /blue/arthur.porto-biocosmos/jhennessy7.gatech/scratch -name "*${SLURM_JOB_ID}*" -type d 2>/dev/null | head -5
fi
