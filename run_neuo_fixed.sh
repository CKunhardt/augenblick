#!/bin/bash

# Neuralangelo B200 Training Pipeline (COLMAP) - Standalone Version
# This script runs the Neuralangelo training pipeline without SLURM

# Configuration variables (modify these as needed)
export USER_BASE="/blue/arthur.porto-biocosmos/jhennessy7.gatech"
export SCRATCH_DIR="${USER_BASE}/scratch"
export LOG_DIR="${SCRATCH_DIR}/neuralangelo_logs"
export VENV_PATH="/home/jhennessy7.gatech/neuralangelo_b200_env"
export NEURALANGELO_PATH="/home/jhennessy7.gatech/augenblick/src/neuralangelo"
export SCRIPT_DIR="/home/jhennessy7.gatech/augenblick"
export ORGANIZED_DIR="${SCRATCH_DIR}/scale_organized"

# GPU configuration
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Training parameters
MAX_STEPS=25000
MESH_RESOLUTION=8192
BLOCK_RES=256
GPU_ID=0

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Create timestamped directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
WORK_DIR="${SCRATCH_DIR}/skull_neuro_colmap_${TIMESTAMP}"
mkdir -p "${WORK_DIR}"

# Set up logging
LOG_FILE="${WORK_DIR}/training.log"
exec 1> >(tee "${LOG_FILE}")
exec 2>&1

echo "=========================================="
echo "Neuralangelo B200 Training Pipeline (COLMAP)"
echo "=========================================="
echo "Hostname: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: ${WORK_DIR}"
echo "=========================================="

# Check if running in appropriate environment
if ! command -v module &> /dev/null; then
    echo "WARNING: Module system not available. Make sure required software is in PATH."
else
    # Load modules
    echo "Loading modules..."
    module load pytorch/2.8 2>/dev/null || module load pytorch/2.7 2>/dev/null || echo "PyTorch module not found"
    module load cuda/12.8.1 2>/dev/null || echo "CUDA module not found"
    module load colmap/3.11 2>/dev/null || echo "COLMAP module not found"
    
    echo "Loaded modules:"
    module list 2>&1
fi

# Activate virtual environment
if [ -f "${VENV_PATH}/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "${VENV_PATH}/bin/activate"
else
    echo "ERROR: Virtual environment not found at ${VENV_PATH}"
    echo "Please create it or update VENV_PATH in this script"
    exit 1
fi

# Check Python environment
echo ""
echo "Python environment:"
echo "Python: $(which python)"
python -c 'import torch; print(f"PyTorch version: {torch.__version__}")'
python -c 'import torch; print(f"CUDA available: {torch.cuda.is_available()}")'
python -c 'import torch; print(f"CUDA version: {torch.version.cuda}")'
echo ""

# Detect GPU type
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    echo "Detected GPU: ${GPU_NAME}"
else
    echo "WARNING: nvidia-smi not found. Cannot detect GPU type."
    GPU_NAME="Unknown"
fi

# Check if organized data exists
if [ ! -d "${ORGANIZED_DIR}" ]; then
    echo "ERROR: Organized data not found at ${ORGANIZED_DIR}"
    echo "Please run data organization first or update ORGANIZED_DIR in this script"
    exit 1
fi

# Create combined input directory
echo ""
echo "Setting up work directory..."
mkdir -p "${WORK_DIR}/input_combined"

# Copy/link images and masks to combined directory
echo "Creating combined input directory..."
for img in "${ORGANIZED_DIR}/images/"*.jpg; do
    if [ -f "$img" ]; then
        base=$(basename "$img" .jpg)
        ln -sf "$img" "${WORK_DIR}/input_combined/${base}.jpg"
        if [ -f "${ORGANIZED_DIR}/masks/${base}.png" ]; then
            ln -sf "${ORGANIZED_DIR}/masks/${base}.png" "${WORK_DIR}/input_combined/${base}.mask.png"
        fi
    fi
done

# Also create separate symlinks
ln -sfn "${ORGANIZED_DIR}/images" "${WORK_DIR}/images"
ln -sfn "${ORGANIZED_DIR}/masks" "${WORK_DIR}/masks"

# Count frames
IMG_COUNT=$(ls -1 "${WORK_DIR}/input_combined/"*.jpg 2>/dev/null | wc -l)
MASK_COUNT=$(ls -1 "${WORK_DIR}/input_combined/"*.mask.png 2>/dev/null | wc -l)

echo "✓ Created combined input directory"
echo "  Images: ${IMG_COUNT} files"
echo "  Masks: ${MASK_COUNT} files"

if [ ${IMG_COUNT} -eq 0 ]; then
    echo "ERROR: No images found in input directory"
    exit 1
fi

# Copy required scripts
echo ""
echo "Copying required scripts..."

# Main reconstruction script
if [ -f "${SCRIPT_DIR}/masked_reconstruction_vggt.py" ]; then
    cp "${SCRIPT_DIR}/masked_reconstruction_vggt.py" "${WORK_DIR}/"
    echo "✓ Copied reconstruction script"
else
    echo "ERROR: masked_reconstruction_vggt.py not found at ${SCRIPT_DIR}"
    exit 1
fi

# Training monitor script
if [ -f "${SCRIPT_DIR}/training_monitor.py" ]; then
    cp "${SCRIPT_DIR}/training_monitor.py" "${WORK_DIR}/"
    chmod +x "${WORK_DIR}/training_monitor.py"
    echo "✓ Copied training monitor"
fi

# Copy config files and supporting scripts
for file in stage*.yaml *.json prep_crop.py scale_transforms_to_original.py convert_transforms_to_neuralangelo.py; do
    if [ -f "${SCRIPT_DIR}/${file}" ]; then
        cp "${SCRIPT_DIR}/${file}" "${WORK_DIR}/"
        echo "✓ Copied ${file}"
    fi
done

# Create subdirectories
mkdir -p "${WORK_DIR}/logs"
mkdir -p "${WORK_DIR}/checkpoints"
mkdir -p "${WORK_DIR}/meshes"

# Determine config template
CONFIG_TEMPLATE=""
if [[ "${GPU_NAME}" == *"B200"* ]]; then
    if [ -f "${NEURALANGELO_PATH}/projects/neuralangelo/configs/b200_template.yaml" ]; then
        CONFIG_TEMPLATE="${NEURALANGELO_PATH}/projects/neuralangelo/configs/b200_template.yaml"
        echo "Using B200 template configuration"
    fi
fi

# Change to work directory
cd "${WORK_DIR}"

echo ""
echo "=========================================="
echo "Starting training with COLMAP"
echo "=========================================="
echo "Note: Using COLMAP for pose estimation"
echo "Work directory: ${WORK_DIR}"
echo ""

# Build command
CMD="python ./masked_reconstruction_vggt.py"
CMD="${CMD} ${WORK_DIR}/input_combined"
CMD="${CMD} ${WORK_DIR}/output"
CMD="${CMD} --gpu ${GPU_ID}"
CMD="${CMD} --max_steps ${MAX_STEPS}"
CMD="${CMD} --mesh_resolution ${MESH_RESOLUTION}"
CMD="${CMD} --block_res ${BLOCK_RES}"
CMD="${CMD} --skip-vggt"
CMD="${CMD} --use-module-colmap"
CMD="${CMD} --no-depth"

if [ -n "${CONFIG_TEMPLATE}" ]; then
    CMD="${CMD} --config-template ${CONFIG_TEMPLATE}"
fi

echo "Running command:"
echo "${CMD}"
echo ""

# Start training
${CMD} &
TRAIN_PID=$!
echo "Training started with PID: ${TRAIN_PID}"

# Wait a bit for training to start
sleep 60

# Start monitoring in background if available
if [ -f "${WORK_DIR}/training_monitor.py" ]; then
    echo ""
    echo "Starting training monitor..."
    python "${WORK_DIR}/training_monitor.py" "${WORK_DIR}/output" --interval 120 > "${WORK_DIR}/monitor.log" 2>&1 &
    MONITOR_PID=$!
    echo "Monitor started with PID: ${MONITOR_PID}"
fi

# Function to handle interruption
cleanup() {
    echo ""
    echo "Received interrupt signal. Cleaning up..."
    
    # Kill training process
    if [ -n "$TRAIN_PID" ] && kill -0 $TRAIN_PID 2>/dev/null; then
        kill -TERM $TRAIN_PID
        echo "Sent TERM signal to training process"
    fi
    
    # Kill monitor process
    if [ -n "$MONITOR_PID" ] && kill -0 $MONITOR_PID 2>/dev/null; then
        kill -TERM $MONITOR_PID
        echo "Sent TERM signal to monitor process"
    fi
    
    echo "Cleanup complete. Exiting."
    exit 1
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for training to complete
wait $TRAIN_PID
TRAIN_EXIT_CODE=$?

# Kill monitor if still running
if [ -n "$MONITOR_PID" ] && kill -0 $MONITOR_PID 2>/dev/null; then
    kill $MONITOR_PID
fi

# Check if training completed successfully
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    
    # List generated files
    echo ""
    echo "Generated files:"
    echo "Meshes:"
    ls -lh output/meshes/*.ply 2>/dev/null || ls -lh output/*.ply 2>/dev/null || echo "  No meshes found"
    echo ""
    echo "Checkpoints:"
    ls -lh output/logs/checkpoints/*.pth 2>/dev/null || echo "  No checkpoints found"
else
    echo ""
    echo "❌ Training failed with error code $TRAIN_EXIT_CODE"
fi

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "Output directory: ${WORK_DIR}"
echo "Log file: ${LOG_FILE}"
echo "=========================================="

# Create completion marker
touch "${WORK_DIR}/COMPLETED_$(date +%s)"

# Deactivate virtual environment
deactivate 2>/dev/null || true
