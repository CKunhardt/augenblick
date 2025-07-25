#!/bin/bash
#SBATCH --job-name=neura_b200
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=160G
#SBATCH --time=6:00:00
#SBATCH --output=/blue/arthur.porto-biocosmos/jhennessy7.gatech/scratch/neuralangelo_logs/slurm-%j.out
#SBATCH --error=/blue/arthur.porto-biocosmos/jhennessy7.gatech/scratch/neuralangelo_logs/slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jhennessy7@gatech.edu

# Ensure log directory exists
mkdir -p /blue/arthur.porto-biocosmos/jhennessy7.gatech/scratch/neuralangelo_logs

# Create timestamped directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
WORK_DIR="/blue/arthur.porto-biocosmos/jhennessy7.gatech/scratch/skull_neuro_${TIMESTAMP}"
mkdir -p "${WORK_DIR}"

# Also redirect to work directory for convenience
exec 1> >(tee "${WORK_DIR}/training.log")
exec 2>&1

echo "=========================================="
echo "Neuralangelo B200 Training Pipeline"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Hostname: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: ${WORK_DIR}"
echo "=========================================="

# Load modules
module load pytorch/2.7
module load cuda/12.8.1

# Check if trimesh is available for mesh validation
python3 -c "import trimesh" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing trimesh for mesh validation..."
    pip install trimesh
fi

# Activate virtual environment
source /home/jhennessy7.gatech/neuralangelo_b200_env/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Detect GPU type
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
echo "Detected GPU: ${GPU_NAME}"

# Set Neuralangelo path (used by the script)
export NEURALANGELO_PATH="/home/jhennessy7.gatech/augenblick/src/neuralangelo"

# Check/Install tiny-cuda-nn
python -c "import tinycudann" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing tiny-cuda-nn..."
    pip install ninja
    
    # Clone to work directory instead of home
    if [ ! -d "${WORK_DIR}/tiny-cuda-nn" ]; then
        git clone https://github.com/NVlabs/tiny-cuda-nn.git "${WORK_DIR}/tiny-cuda-nn"
    fi
    
    # Install from work directory
    cd "${WORK_DIR}/tiny-cuda-nn/bindings/torch"
    pip install -e .
    
    # Test installation
    cd "${WORK_DIR}"
    python -c "import tinycudann; print('tiny-cuda-nn installed successfully')"
else
    echo "tiny-cuda-nn already installed"
fi

# Copy necessary files to work directory
echo ""
echo "Setting up work directory..."

# Use existing organized data
ORGANIZED_DIR="/blue/arthur.porto-biocosmos/jhennessy7.gatech/scratch/scale_organized"

if [ ! -d "${ORGANIZED_DIR}" ]; then
    echo "ERROR: Organized data not found at ${ORGANIZED_DIR}"
    exit 1
fi

# Create a combined input directory that masked_reconstruction_vggt.py expects
mkdir -p "${WORK_DIR}/input_combined"

# Copy/link images and masks to combined directory
echo "Creating combined input directory..."
for img in "${ORGANIZED_DIR}/images/"*.jpg; do
    if [ -f "$img" ]; then
        base=$(basename "$img" .jpg)
        # Create symlinks in combined directory
        ln -sf "$img" "${WORK_DIR}/input_combined/${base}.jpg"
        # Also link the mask with .mask.png naming convention
        if [ -f "${ORGANIZED_DIR}/masks/${base}.png" ]; then
            ln -sf "${ORGANIZED_DIR}/masks/${base}.png" "${WORK_DIR}/input_combined/${base}.mask.png"
        fi
    fi
done

# Also create separate symlinks for other scripts that might need them
ln -sfn "${ORGANIZED_DIR}/images" "${WORK_DIR}/images"
ln -sfn "${ORGANIZED_DIR}/masks" "${WORK_DIR}/masks"

# Copy camera info if it exists
if [ -f "${ORGANIZED_DIR}/camera_info.txt" ]; then
    cp "${ORGANIZED_DIR}/camera_info.txt" "${WORK_DIR}/"
fi

# Count frames
IMG_COUNT=$(ls -1 "${WORK_DIR}/input_combined/"*.jpg 2>/dev/null | wc -l)
MASK_COUNT=$(ls -1 "${WORK_DIR}/input_combined/"*.mask.png 2>/dev/null | wc -l)

echo "✓ Created combined input directory"
echo "  Images: ${IMG_COUNT} files"
echo "  Masks: ${MASK_COUNT} files"

# Copy pipeline scripts
SCRIPT_DIR="/home/jhennessy7.gatech/augenblick"
if [ -f "${SCRIPT_DIR}/masked_reconstruction_vggt.py" ]; then
    cp "${SCRIPT_DIR}/masked_reconstruction_vggt.py" "${WORK_DIR}/"
    echo "✓ Copied reconstruction script"
else
    echo "ERROR: masked_reconstruction_vggt.py not found"
    exit 1
fi

# Copy training monitor script
if [ -f "${SCRIPT_DIR}/training_monitor.py" ]; then
    cp "${SCRIPT_DIR}/training_monitor.py" "${WORK_DIR}/"
    chmod +x "${WORK_DIR}/training_monitor.py"
    echo "✓ Copied training monitor"
fi

# Copy stage config files
for stage_config in stage1_coarse.yaml stage2_mid.yaml stage3_fine.yaml; do
    if [ -f "${SCRIPT_DIR}/${stage_config}" ]; then
        cp "${SCRIPT_DIR}/${stage_config}" "${WORK_DIR}/"
        echo "✓ Copied ${stage_config}"
    fi
done

# Copy any supporting scripts
for script in prep_crop.py vggt_batch_processor.py scale_transforms_to_original.py convert_transforms_to_neuralangelo.py; do
    if [ -f "${SCRIPT_DIR}/${script}" ]; then
        cp "${SCRIPT_DIR}/${script}" "${WORK_DIR}/"
        echo "✓ Copied ${script}"
    fi
done

# Create subdirectories
mkdir -p "${WORK_DIR}/logs"
mkdir -p "${WORK_DIR}/checkpoints"
mkdir -p "${WORK_DIR}/meshes"

# Save run information
cat > "${WORK_DIR}/run_info.txt" << EOF
Neuralangelo B200 Training Run
==============================
Job ID: ${SLURM_JOB_ID}
Timestamp: ${TIMESTAMP}
Hostname: $(hostname)
GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)
Start time: $(date)
Python: $(which python)
PyTorch: $(python -c "import torch; print(torch.__version__)")
CUDA: $(python -c "import torch; print(torch.version.cuda)")

Parameters:
- Training mode: Staged (3 stages)
- Stage 1: 2k iterations (coarse)
- Stage 2: 10k iterations (mid)
- Stage 3: 20k iterations (fine)
- Mesh resolution: 16384
- Block resolution: 512

Directories:
- Work dir: ${WORK_DIR}
- Logs: ${WORK_DIR}/logs
- Checkpoints: ${WORK_DIR}/checkpoints
- Meshes: ${WORK_DIR}/meshes
EOF

echo ""
echo "Work directory setup complete!"
echo ""

# Change to work directory
cd "${WORK_DIR}"

# Run the pipeline to generate VGGT output
echo "Starting VGGT processing..."
echo ""

# Determine which VGGT script to use (updated name)
VGGT_SCRIPT_PATH=""
for script in \
    "${SCRIPT_DIR}/vggt_batch_processor.py" \
    "${HOME}/augenblick/vggt_batch_processor.py" \
    "${HOME}/augenblick/src/vggt/vggt_batch_processor.py"; do
    if [ -f "$script" ]; then
        VGGT_SCRIPT_PATH="$script"
        echo "Found VGGT script: $script"
        break
    fi
done

# Run initial processing with VGGT
echo "Running masked_reconstruction_vggt.py for VGGT processing..."

python ./masked_reconstruction_vggt.py \
    "${WORK_DIR}/input_combined" \
    "${WORK_DIR}/output" \
    --gpu 0 \
    --skip-colmap \
    --use-vggt-script \
    --vggt-script "${VGGT_SCRIPT_PATH}" \
    --use-depth \
    --max_steps 0 &  # Set to 0 to skip training in this step

# Save the PID
VGGT_PID=$!
echo "VGGT processing started with PID: ${VGGT_PID}"

# Wait for VGGT to complete
wait $VGGT_PID
VGGT_EXIT_CODE=$?

if [ $VGGT_EXIT_CODE -ne 0 ]; then
    echo "❌ VGGT processing failed!"
    exit 1
fi

# Find the neuralangelo data directory created by VGGT
NEURALANGELO_DATA_DIR="${WORK_DIR}/output/vggt/neuralangelo_data"
if [ ! -d "${NEURALANGELO_DATA_DIR}" ]; then
    # Try alternative locations
    NEURALANGELO_DATA_DIR=$(find "${WORK_DIR}/output" -name "neuralangelo_data" -type d | head -1)
    if [ -z "$NEURALANGELO_DATA_DIR" ]; then
        echo "ERROR: Could not find neuralangelo_data directory!"
        exit 1
    fi
fi

echo "✓ Found Neuralangelo data at: ${NEURALANGELO_DATA_DIR}"

# Update stage configs with correct data root
echo ""
echo "Updating stage configurations..."

update_stage_config() {
    local config_file=$1
    local data_root=$2
    
    if [ -f "${config_file}" ]; then
        # Use Python to update YAML preserving structure
        python3 << EOF
import yaml
import sys

with open('${config_file}', 'r') as f:
    config = yaml.safe_load(f)

# Update data root
config['data']['root'] = '${data_root}'

# Ensure depth supervision is enabled if depth maps exist
depth_dir = '${data_root}/depth_maps'
import os
if os.path.exists(depth_dir) and len(os.listdir(depth_dir)) > 0:
    if 'model' not in config:
        config['model'] = {}
    if 'depth_supervision' not in config['model']:
        config['model']['depth_supervision'] = {}
    config['model']['depth_supervision']['enabled'] = True
    config['model']['depth_supervision']['weight'] = 0.1
    print(f"✓ Enabled depth supervision for {os.path.basename('${config_file}')}")

# Write updated config
with open('${config_file}', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
EOF
    fi
}

# Update all stage configs
for stage_config in stage1_coarse.yaml stage2_mid.yaml stage3_fine.yaml; do
    if [ -f "${WORK_DIR}/${stage_config}" ]; then
        update_stage_config "${WORK_DIR}/${stage_config}" "${NEURALANGELO_DATA_DIR}"
        echo "✓ Updated ${stage_config}"
    fi
done

# Change to Neuralangelo directory for training
cd "${NEURALANGELO_PATH}"

echo ""
echo "=========================================="
echo "Starting Staged Neuralangelo Training"
echo "=========================================="

# Function to find latest checkpoint
find_latest_checkpoint() {
    local ckpt_dir="${WORK_DIR}/logs/checkpoints"
    if [ -d "$ckpt_dir" ]; then
        ls -t "$ckpt_dir"/*.pth 2>/dev/null | head -1
    fi
}

# Stage 1: Coarse (2k iterations)
echo ""
echo "=== Stage 1: Coarse Reconstruction (2k iterations) ==="
echo "Resolution: 512x768, Fast convergence"

torchrun --nproc_per_node=1 --master_port=29501 train.py \
    --logdir "${WORK_DIR}/logs" \
    --config "${WORK_DIR}/stage1_coarse.yaml" \
    --show_pbar

STAGE1_EXIT=$?
if [ $STAGE1_EXIT -ne 0 ]; then
    echo "❌ Stage 1 failed!"
    exit 1
fi

echo "✓ Stage 1 complete"
LATEST_CKPT=$(find_latest_checkpoint)
echo "Latest checkpoint: ${LATEST_CKPT}"

# Extract Stage 1 mesh
if [ -n "$LATEST_CKPT" ]; then
    echo "Extracting Stage 1 mesh (low resolution)..."
    STAGE1_MESH="${WORK_DIR}/meshes/mesh_stage1_coarse.ply"
    
    torchrun --nproc_per_node=1 --master_port=29511 \
        projects/neuralangelo/scripts/extract_mesh.py \
        --config "${WORK_DIR}/stage1_coarse.yaml" \
        --checkpoint "${LATEST_CKPT}" \
        --output_file "${STAGE1_MESH}" \
        --resolution 2048 \
        --block_res 128
    
    if [ -f "${STAGE1_MESH}" ]; then
        echo "✓ Stage 1 mesh extracted: $(du -h "${STAGE1_MESH}" | cut -f1)"
        
        # Validate mesh has faces
        FACE_COUNT=$(python3 -c "
import sys
try:
    import trimesh
    mesh = trimesh.load('${STAGE1_MESH}')
    print(len(mesh.faces))
except Exception as e:
    print('0')
" 2>/dev/null)
        
        if [ "$FACE_COUNT" = "0" ] || [ -z "$FACE_COUNT" ]; then
            echo "❌ ERROR: Stage 1 mesh has 0 faces!"
            echo "This indicates the reconstruction failed. Possible issues:"
            echo "  - Camera poses may be incorrect"
            echo "  - Mask/image alignment problems"
            echo "  - Training parameters need adjustment"
            echo "  - SDF initialization failed"
            echo ""
            echo "Stopping pipeline to prevent wasted computation."
            exit 1
        else
            echo "✓ Mesh validation passed: ${FACE_COUNT} faces"
        fi
    else
        echo "❌ ERROR: Stage 1 mesh extraction failed!"
        exit 1
    fi
fi

# Stage 2: Mid (10k iterations total)
echo ""
echo "=== Stage 2: Mid-Resolution Reconstruction (10k iterations) ==="
echo "Resolution: 1040x1560, Refining geometry"

if [ -n "$LATEST_CKPT" ]; then
    torchrun --nproc_per_node=1 --master_port=29502 train.py \
        --logdir "${WORK_DIR}/logs" \
        --config "${WORK_DIR}/stage2_mid.yaml" \
        --checkpoint "${LATEST_CKPT}" \
        --show_pbar
else
    echo "WARNING: No checkpoint found from Stage 1, starting fresh"
    torchrun --nproc_per_node=1 --master_port=29502 train.py \
        --logdir "${WORK_DIR}/logs" \
        --config "${WORK_DIR}/stage2_mid.yaml" \
        --show_pbar
fi

STAGE2_EXIT=$?
if [ $STAGE2_EXIT -ne 0 ]; then
    echo "❌ Stage 2 failed!"
    exit 1
fi

echo "✓ Stage 2 complete"
LATEST_CKPT=$(find_latest_checkpoint)
echo "Latest checkpoint: ${LATEST_CKPT}"

# Extract Stage 2 mesh
if [ -n "$LATEST_CKPT" ]; then
    echo "Extracting Stage 2 mesh (medium resolution)..."
    STAGE2_MESH="${WORK_DIR}/meshes/mesh_stage2_mid.ply"
    
    torchrun --nproc_per_node=1 --master_port=29512 \
        projects/neuralangelo/scripts/extract_mesh.py \
        --config "${WORK_DIR}/stage2_mid.yaml" \
        --checkpoint "${LATEST_CKPT}" \
        --output_file "${STAGE2_MESH}" \
        --resolution 8192 \
        --block_res 256
    
    if [ -f "${STAGE2_MESH}" ]; then
        echo "✓ Stage 2 mesh extracted: $(du -h "${STAGE2_MESH}" | cut -f1)"
        
        # Validate mesh has faces
        FACE_COUNT=$(python3 -c "
import sys
try:
    import trimesh
    mesh = trimesh.load('${STAGE2_MESH}')
    print(len(mesh.faces))
except Exception as e:
    print('0')
" 2>/dev/null)
        
        if [ "$FACE_COUNT" = "0" ] || [ -z "$FACE_COUNT" ]; then
            echo "❌ ERROR: Stage 2 mesh has 0 faces!"
            echo "Stage 1 passed but Stage 2 failed. Possible issues:"
            echo "  - Learning rate may be too high"
            echo "  - Resolution jump too large"
            echo "  - Checkpoint loading failed"
            echo ""
            echo "Stopping pipeline to prevent wasted computation."
            exit 1
        else
            echo "✓ Mesh validation passed: ${FACE_COUNT} faces"
        fi
    else
        echo "❌ ERROR: Stage 2 mesh extraction failed!"
        exit 1
    fi
fi

# Stage 3: Fine (20k iterations total)
echo ""
echo "=== Stage 3: Fine Detail Reconstruction (20k iterations) ==="
echo "Resolution: 4160x6240 (full), Capturing fine anatomical details"

if [ -n "$LATEST_CKPT" ]; then
    torchrun --nproc_per_node=1 --master_port=29503 train.py \
        --logdir "${WORK_DIR}/logs" \
        --config "${WORK_DIR}/stage3_fine.yaml" \
        --checkpoint "${LATEST_CKPT}" \
        --show_pbar
else
    echo "WARNING: No checkpoint found from Stage 2, starting fresh"
    torchrun --nproc_per_node=1 --master_port=29503 train.py \
        --logdir "${WORK_DIR}/logs" \
        --config "${WORK_DIR}/stage3_fine.yaml" \
        --show_pbar
fi

STAGE3_EXIT=$?
if [ $STAGE3_EXIT -ne 0 ]; then
    echo "❌ Stage 3 failed!"
    exit 1
fi

echo "✓ Stage 3 complete"
LATEST_CKPT=$(find_latest_checkpoint)
echo "Latest checkpoint: ${LATEST_CKPT}"

# Extract Stage 3 mesh (final high-resolution)
if [ -n "$LATEST_CKPT" ]; then
    echo "Extracting Stage 3 mesh (high resolution)..."
    STAGE3_MESH="${WORK_DIR}/meshes/mesh_stage3_fine.ply"
    
    torchrun --nproc_per_node=1 --master_port=29513 \
        projects/neuralangelo/scripts/extract_mesh.py \
        --config "${WORK_DIR}/stage3_fine.yaml" \
        --checkpoint "${LATEST_CKPT}" \
        --output_file "${STAGE3_MESH}" \
        --resolution 16384 \
        --block_res 512
    
    if [ -f "${STAGE3_MESH}" ]; then
        echo "✓ Stage 3 mesh extracted: $(du -h "${STAGE3_MESH}" | cut -f1)"
        
        # Validate mesh has faces
        FACE_COUNT=$(python3 -c "
import sys
try:
    import trimesh
    mesh = trimesh.load('${STAGE3_MESH}')
    print(len(mesh.faces))
except Exception as e:
    print('0')
" 2>/dev/null)
        
        if [ "$FACE_COUNT" = "0" ] || [ -z "$FACE_COUNT" ]; then
            echo "❌ ERROR: Stage 3 mesh has 0 faces!"
            echo "Previous stages passed but final stage failed. Possible issues:"
            echo "  - Memory constraints at full resolution"
            echo "  - Numerical instability at high resolution"
            echo "  - Need to adjust fine-tuning parameters"
            echo ""
            echo "Consider using Stage 2 mesh as final result."
            exit 1
        else
            echo "✓ Mesh validation passed: ${FACE_COUNT} faces"
            echo "✅ Final mesh successfully generated with ${FACE_COUNT} faces"
        fi
    else
        echo "❌ ERROR: Stage 3 mesh extraction failed!"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "All training stages completed successfully!"
echo "=========================================="

# Summary of extracted meshes
echo ""
echo "Extracted Meshes Summary:"
echo "------------------------"
if [ -f "${WORK_DIR}/meshes/mesh_stage1_coarse.ply" ]; then
    echo "Stage 1 (Coarse):  $(du -h "${WORK_DIR}/meshes/mesh_stage1_coarse.ply" | cut -f1)"
fi
if [ -f "${WORK_DIR}/meshes/mesh_stage2_mid.ply" ]; then
    echo "Stage 2 (Mid):     $(du -h "${WORK_DIR}/meshes/mesh_stage2_mid.ply" | cut -f1)"
fi
if [ -f "${WORK_DIR}/meshes/mesh_stage3_fine.ply" ]; then
    echo "Stage 3 (Fine):    $(du -h "${WORK_DIR}/meshes/mesh_stage3_fine.ply" | cut -f1)"
fi
echo ""

# Create a symlink to the final mesh for convenience
if [ -f "${WORK_DIR}/meshes/mesh_stage3_fine.ply" ]; then
    ln -sf "mesh_stage3_fine.ply" "${WORK_DIR}/meshes/final_mesh.ply"
    echo "✓ Created symlink: final_mesh.ply -> mesh_stage3_fine.ply"
fi

# Run final validation
if [ -f "${WORK_DIR}/training_monitor.py" ]; then
    echo ""
    echo "Running final validation..."
    python "${WORK_DIR}/training_monitor.py" "${WORK_DIR}" --mode validate
fi

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "End time: $(date)"
echo "Output directory: ${WORK_DIR}"
echo "Training stages: 3 (coarse → mid → fine)"
echo "Total iterations: 20,000"
echo "=========================================="

# Create completion marker
touch "${WORK_DIR}/COMPLETED_STAGED_${SLURM_JOB_ID}"
