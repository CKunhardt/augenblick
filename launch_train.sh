#!/bin/bash
# Staged Neuralangelo training launch script for NVIDIA B200

# Get the absolute path of the directory where this script is located
WORK_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Paths
NEURALANGELO_PATH="/home/jhennessy7.gatech/augenblick/src/neuralangelo"
BASE_DIR="/blue/arthur.porto-biocosmos/jhennessy7.gatech/scratch"

# B200-specific environment settings
export CUDA_VISIBLE_DEVICES=0

# AGGRESSIVE MEMORY MANAGEMENT
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6"
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="9.0"

# Force garbage collection
export PYTORCH_NO_CUDA_MEMORY_CACHING=0
export PYTORCH_CUDA_ALLOC_SYNC=1

# Set distributed environment variables for single GPU
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

# Add Neuralangelo to Python path
export PYTHONPATH="${NEURALANGELO_PATH}:${PYTHONPATH}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
USE_CHECKPOINT=false
CHECKPOINT_PATH=""
EXTRACT_MESH_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            USE_CHECKPOINT=true
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --resume)
            USE_CHECKPOINT=true
            CHECKPOINT_PATH="auto"
            shift
            ;;
        --extract-mesh)
            EXTRACT_MESH_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --checkpoint PATH  Resume from specific checkpoint"
            echo "  --resume          Resume from latest checkpoint"
            echo "  --extract-mesh    Extract mesh from latest checkpoint and exit"
            echo "  --help            Show this help message"
            echo ""
            echo "Default: Start fresh training from stage 1"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}=========================================="
echo "Neuralangelo 3-Stage Training - B200"
echo "==========================================${NC}"
echo "Working directory: ${WORK_DIR}"
echo ""

# Function to check if checkpoint exists and get iteration
get_checkpoint_info() {
    local checkpoint_path=$1
    if [ -f "$checkpoint_path" ]; then
        # Extract iteration from filename
        if [[ $checkpoint_path =~ iteration_([0-9]+) ]]; then
            echo "${BASH_REMATCH[1]}"
        else
            echo "0"
        fi
    else
        echo "-1"  # File not found
    fi
}

# Function to find latest checkpoint
find_latest_checkpoint() {
    local latest_checkpoint=""
    local latest_iter=0
    
    # Look for checkpoints in logs directory
    for ckpt in ${WORK_DIR}/logs/*checkpoint*.pt ${WORK_DIR}/logs/*.pth; do
        if [ -f "$ckpt" ]; then
            iter=$(get_checkpoint_info "$ckpt")
            if [ $iter -gt $latest_iter ]; then
                latest_iter=$iter
                latest_checkpoint=$ckpt
            fi
        fi
    done
    
    if [ -n "$latest_checkpoint" ]; then
        echo "$latest_checkpoint"
    else
        echo ""
    fi
}

# Function to determine current stage based on iteration
get_stage_from_iteration() {
    local iter=$1
    if [ $iter -lt 2000 ]; then
        echo "1"
    elif [ $iter -lt 10000 ]; then
        echo "2"
    else
        echo "3"
    fi
}

# Function to run a specific stage
run_stage() {
    local stage=$1
    local resume_checkpoint=$2
    local config_file=""
    local train_script=""
    local stage_name=""
    local target_iter=""
    
    case $stage in
        1)
            config_file="stage1_coarse.yaml"
            train_script="train_stage1.py"
            stage_name="Stage 1: Coarse (512×768)"
            target_iter="2000"
            ;;
        2)
            config_file="stage2_mid.yaml"
            train_script="train_stage2.py"
            stage_name="Stage 2: Mid-Resolution (1024×1536)"
            target_iter="10000"
            ;;
        3)
            config_file="stage3_fine.yaml"
            train_script="train_stage3.py"
            stage_name="Stage 3: Fine Detail (2080×3120)"
            target_iter="20000"
            ;;
        *)
            echo -e "${RED}Invalid stage: $stage${NC}"
            exit 1
            ;;
    esac
    
    echo -e "\n${GREEN}▶ Starting ${stage_name}${NC}"
    echo -e "${YELLOW}Target iterations: ${target_iter}${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Check if config exists
    if [ ! -f "${WORK_DIR}/logs/${config_file}" ]; then
        echo -e "${RED}ERROR: Config file not found: ${WORK_DIR}/logs/${config_file}${NC}"
        echo "Please create the stage configuration files first."
        exit 1
    fi
    
    # Check if training script exists
    if [ ! -f "${WORK_DIR}/${train_script}" ]; then
        echo -e "${RED}ERROR: Training script not found: ${WORK_DIR}/${train_script}${NC}"
        echo "Please create the training scripts first."
        exit 1
    fi
    
    cd ${NEURALANGELO_PATH}
    
    # Build command
    local cmd="python ${WORK_DIR}/${train_script} --config ${WORK_DIR}/logs/${config_file} --logdir ${WORK_DIR}/logs --local_rank 0 --show_pbar"
    
    # Add checkpoint if resuming
    if [ -n "$resume_checkpoint" ]; then
        # Fix checkpoint iteration if needed
        if [ -f "${WORK_DIR}/fix_checkpoint_iter.py" ]; then
            echo -e "${YELLOW}Fixing checkpoint iteration...${NC}"
            python ${WORK_DIR}/fix_checkpoint_iter.py "$resume_checkpoint"
        fi
        cmd="$cmd --checkpoint $resume_checkpoint"
        echo -e "${BLUE}Resuming from checkpoint: $(basename $resume_checkpoint)${NC}"
    fi
    
    # Run training
    echo -e "${GREEN}Running: $cmd${NC}\n"
    eval $cmd 2>&1 | tee ${WORK_DIR}/logs/train_stage${stage}.log
    
    # Extract mesh after stage completes
    echo -e "\n${YELLOW}Extracting preview mesh for ${stage_name}...${NC}"
    
    # Find the latest checkpoint for this stage
    local stage_checkpoint=$(find_latest_checkpoint)
    if [ -z "$stage_checkpoint" ]; then
        echo -e "${YELLOW}Warning: No checkpoint found for mesh extraction${NC}"
        return
    fi
    
    # Different resolutions for different stages
    local mesh_resolution=512
    local block_resolution=128
    
    case $stage in
        1)
            mesh_resolution=256
            block_resolution=64
            ;;
        2)
            mesh_resolution=512
            block_resolution=128
            ;;
        3)
            mesh_resolution=768
            block_resolution=192
            ;;
    esac
    
    # Extract mesh
    cd ${NEURALANGELO_PATH}
    python projects/neuralangelo/scripts/extract_mesh.py \
        --config ${WORK_DIR}/logs/${config_file} \
        --checkpoint $stage_checkpoint \
        --output_file ${WORK_DIR}/logs/mesh_stage${stage}_iter${target_iter}.ply \
        --resolution $mesh_resolution \
        --block_res $block_resolution \
        --threshold 0.0
    
    # Also save a latest symlink
    if [ -f "${WORK_DIR}/logs/mesh_stage${stage}_iter${target_iter}.ply" ]; then
        ln -sf mesh_stage${stage}_iter${target_iter}.ply ${WORK_DIR}/logs/mesh_stage${stage}_latest.ply
        echo -e "${GREEN}✓ Mesh saved: mesh_stage${stage}_iter${target_iter}.ply${NC}"
        echo -e "${GREEN}  Resolution: ${mesh_resolution}³${NC}"
    else
        echo -e "${RED}Failed to extract mesh${NC}"
    fi
}

# Function to ensure transforms.json has correct scale and Neuralangelo parameters
ensure_transforms_json() {
    local transforms_path="${WORK_DIR}/transforms.json"
    
    # Check if transforms.json exists and has sphere parameters
    if [ -f "$transforms_path" ]; then
        if python -c "import json; data=json.load(open('$transforms_path')); exit(0 if 'sphere_center' in data else 1)" 2>/dev/null; then
            echo -e "${GREEN}✓ transforms.json already has Neuralangelo parameters${NC}"
            return
        fi
    fi
    
    # First choice: Use the already-scaled transforms from the pipeline
    local scaled_transforms="${BASE_DIR}/scale_neuralangelo_fullres/transforms.json"
    if [ -f "$scaled_transforms" ]; then
        echo -e "${YELLOW}Using pre-scaled transforms from pipeline...${NC}"
        cp $scaled_transforms $transforms_path
        
        # Add Neuralangelo sphere parameters
        python -c "
import json

with open('$transforms_path', 'r') as f:
    data = json.load(f)

# Add Neuralangelo parameters from config
center = [0.6389583298477574, 0.6039861034655917, 0.5979484190111575]
scale = 0.8946571020905045

data['sphere_center'] = center
data['sphere_radius'] = scale
data['aabb_range'] = [
    [-scale + center[0], scale + center[0]],
    [-scale + center[1], scale + center[1]], 
    [-scale + center[2], scale + center[2]]
]

# Update image paths to point to correct location
for frame in data.get('frames', []):
    if 'file_path' in frame:
        # Make sure paths point to the actual images
        filename = frame['file_path'].split('/')[-1]
        frame['file_path'] = '${BASE_DIR}/scale_organized/images/' + filename

with open('$transforms_path', 'w') as f:
    json.dump(data, f, indent=2)
print('✓ Added Neuralangelo parameters to scaled transforms')
print(f'  Sphere center: {center}')
print(f'  Sphere radius: {scale}')
"
    else
        # Fallback: create minimal transforms
        echo -e "${RED}Warning: No scaled transforms found!${NC}"
        echo -e "${YELLOW}Creating minimal transforms.json...${NC}"
        python -c "
import json

transforms = {
    'sphere_center': [0.6389583298477574, 0.6039861034655917, 0.5979484190111575],
    'sphere_radius': 0.8946571020905045,
    'aabb_range': [
        [-0.2557, 1.5336],
        [-0.2907, 1.4987],
        [-0.2967, 1.4926]
    ],
    'camera_model': 'OPENCV'
}

with open('$transforms_path', 'w') as f:
    json.dump(transforms, f, indent=2)
print('Created minimal transforms.json')
"
    fi
}

# Function to extract mesh with proper settings
extract_mesh() {
    local checkpoint_path=$1
    local output_name=$2
    local resolution=$3
    local config_stage=$4
    
    # Default values
    resolution=${resolution:-1024}
    config_stage=${config_stage:-3}
    
    local config_file="stage${config_stage}_fine.yaml"
    if [ $config_stage -eq 1 ]; then
        config_file="stage1_coarse.yaml"
    elif [ $config_stage -eq 2 ]; then
        config_file="stage2_mid.yaml"
    fi
    
    # Ensure transforms.json exists
    ensure_transforms_json
    
    echo -e "${YELLOW}Extracting mesh...${NC}"
    echo "  Checkpoint: $(basename $checkpoint_path)"
    echo "  Resolution: ${resolution}³"
    echo "  Config: $config_file"
    
    cd ${NEURALANGELO_PATH}
    python ${NEURALANGELO_PATH}/projects/neuralangelo/scripts/extract_mesh.py \
        --config ${WORK_DIR}/logs/${config_file} \
        --checkpoint $checkpoint_path \
        --output_file ${WORK_DIR}/${output_name} \
        --resolution $resolution \
        --block_res $((resolution/8)) \
        --single_gpu
    
    if [ -f "${WORK_DIR}/${output_name}" ]; then
        echo -e "${GREEN}✅ Mesh saved to: ${output_name}${NC}"
        # Print mesh statistics
        python -c "
import trimesh
mesh = trimesh.load('${WORK_DIR}/${output_name}')
print(f'  Vertices: {len(mesh.vertices):,}')
print(f'  Faces: {len(mesh.faces):,}')
print(f'  Watertight: {mesh.is_watertight}')
" 2>/dev/null || true
    else
        echo -e "${RED}Failed to extract mesh${NC}"
    fi
}

# Handle mesh extraction only mode
if [ "$EXTRACT_MESH_ONLY" = true ]; then
    echo -e "${BLUE}=========================================="
    echo "Mesh Extraction Mode"
    echo "==========================================${NC}"
    
    # Find latest checkpoint
    latest_checkpoint=$(find_latest_checkpoint)
    if [ -z "$latest_checkpoint" ]; then
        echo -e "${RED}ERROR: No checkpoint found${NC}"
        exit 1
    fi
    
    # Get iteration
    current_iter=$(get_checkpoint_info "$latest_checkpoint")
    current_stage=$(get_stage_from_iteration $current_iter)
    
    echo -e "${GREEN}Found checkpoint at iteration ${current_iter} (Stage ${current_stage})${NC}"
    echo ""
    echo "Extraction options:"
    echo "1) Quick preview (512³)"
    echo "2) Standard quality (768³)"
    echo "3) High quality (1024³)"
    echo "4) Ultra quality (1536³) - Warning: High memory usage"
    echo "5) Custom resolution"
    read -p "Select quality (1-5): " quality_choice
    
    case $quality_choice in
        1) resolution=512 ;;
        2) resolution=768 ;;
        3) resolution=1024 ;;
        4) resolution=1536 ;;
        5) 
            read -p "Enter resolution (e.g., 896): " resolution
            ;;
        *) resolution=768 ;;
    esac
    
    output_name="skull_iter${current_iter}_res${resolution}.ply"
    extract_mesh "$latest_checkpoint" "$output_name" "$resolution" "$current_stage"
    exit 0
fi

# Main execution logic
if [ "$USE_CHECKPOINT" = true ]; then
    # User wants to resume from checkpoint
    echo -e "${YELLOW}Checkpoint mode activated${NC}"
    
    # Find checkpoint
    if [ "$CHECKPOINT_PATH" = "auto" ]; then
        CHECKPOINT_PATH=$(find_latest_checkpoint)
        if [ -z "$CHECKPOINT_PATH" ]; then
            echo -e "${RED}ERROR: No checkpoint found to resume from${NC}"
            exit 1
        fi
        echo -e "${GREEN}Found latest checkpoint: $(basename $CHECKPOINT_PATH)${NC}"
    else
        if [ ! -f "$CHECKPOINT_PATH" ]; then
            echo -e "${RED}ERROR: Checkpoint not found: $CHECKPOINT_PATH${NC}"
            exit 1
        fi
    fi
    
    # Get iteration and stage
    current_iter=$(get_checkpoint_info "$CHECKPOINT_PATH")
    current_stage=$(get_stage_from_iteration $current_iter)
    
    echo -e "${BLUE}Checkpoint at iteration: ${current_iter}${NC}"
    echo -e "${BLUE}Current stage: ${current_stage}${NC}"
    
    # Ask what to do
    echo ""
    echo "Options:"
    echo "1) Continue training from iteration $current_iter"
    echo "2) Start fresh anyway"
    echo "3) Exit"
    read -p "Enter choice (1-3): " choice
    
    case $choice in
        1)
            # Determine which stage to run based on iteration
            if [ $current_iter -ge 20000 ]; then
                echo -e "${GREEN}Training is complete!${NC}"
                echo "Extracting final mesh..."
                cd ${NEURALANGELO_PATH}
                python projects/neuralangelo/scripts/extract_mesh.py \
                    --config ${WORK_DIR}/logs/stage3_fine.yaml \
                    --checkpoint $CHECKPOINT_PATH \
                    --output_file ${WORK_DIR}/skull_final_b200.ply \
                    --resolution 1024 \
                    --block_res 256
                echo -e "${GREEN}✅ Mesh saved to: skull_final_b200.ply${NC}"
                exit 0
            elif [ $current_iter -ge 10000 ]; then
                run_stage 3 "$CHECKPOINT_PATH"
            elif [ $current_iter -ge 2000 ]; then
                run_stage 2 "$CHECKPOINT_PATH"
            else
                run_stage 1 "$CHECKPOINT_PATH"
            fi
            ;;
        2)
            USE_CHECKPOINT=false
            ;;
        3)
            exit 0
            ;;
    esac
fi

# Default: Fresh training
if [ "$USE_CHECKPOINT" = false ]; then
    echo -e "${GREEN}Starting fresh 3-stage training pipeline${NC}"
    echo ""
    echo "This will run all 3 stages sequentially:"
    echo "  Stage 1: 0-2k iterations (512×768) - ~15 minutes"
    echo "  Stage 2: 2k-10k iterations (1024×1536) - ~45 minutes"
    echo "  Stage 3: 10k-20k iterations (2080×3120) - ~2-3 hours"
    echo ""
    echo -e "${YELLOW}Total estimated time: ~4 hours${NC}"
    echo ""
    read -p "Start training? (y/n): " confirm
    
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "Training cancelled."
        exit 0
    fi
    
    # Clear any existing checkpoints
    if [ -n "$(find_latest_checkpoint)" ]; then
        echo -e "${YELLOW}Found existing checkpoints.${NC}"
        read -p "Delete them and start fresh? (y/n): " delete_confirm
        if [ "$delete_confirm" = "y" ] || [ "$delete_confirm" = "Y" ]; then
            echo "Removing old checkpoints..."
            rm -f ${WORK_DIR}/logs/*.pt ${WORK_DIR}/logs/*.pth
            rm -f ${WORK_DIR}/logs/latest*
        else
            echo "Keeping existing checkpoints. Use --resume to continue from them."
            exit 0
        fi
    fi
    
    echo -e "\n${GREEN}Starting complete 3-stage training pipeline...${NC}"
    
    # Stage 1
    run_stage 1 ""
    
    # Find checkpoint from stage 1
    stage1_checkpoint=$(find_latest_checkpoint)
    if [ -z "$stage1_checkpoint" ]; then
        echo -e "${RED}ERROR: Stage 1 checkpoint not found!${NC}"
        exit 1
    fi
    
    # Stage 2
    run_stage 2 "$stage1_checkpoint"
    
    # Find checkpoint from stage 2
    stage2_checkpoint=$(find_latest_checkpoint)
    if [ -z "$stage2_checkpoint" ]; then
        echo -e "${RED}ERROR: Stage 2 checkpoint not found!${NC}"
        exit 1
    fi
    
    # Stage 3
    run_stage 3 "$stage2_checkpoint"
    
    # Final mesh extraction
    echo -e "\n${YELLOW}Extracting final high-resolution mesh...${NC}"
    final_checkpoint=$(find_latest_checkpoint)
    
    # Extract at multiple resolutions
    extract_mesh "$final_checkpoint" "skull_final_b200_768.ply" 768 3
    extract_mesh "$final_checkpoint" "skull_final_b200_1024.ply" 1024 3
    
    # Create a convenience symlink
    ln -sf skull_final_b200_1024.ply ${WORK_DIR}/skull_final_b200.ply
    
    echo -e "\n${GREEN}🎉 Training complete!${NC}"
    echo -e "${GREEN}Final meshes saved:${NC}"
    echo -e "${GREEN}  - skull_final_b200_768.ply (standard)${NC}"
    echo -e "${GREEN}  - skull_final_b200_1024.ply (high quality)${NC}"
    echo -e "${GREEN}  - skull_final_b200.ply (symlink to 1024)${NC}"
fi

# Show final summary
echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Training Summary${NC}"
echo -e "${BLUE}============================================${NC}"
echo "Logs: ${WORK_DIR}/logs/train_stage*.log"
echo "Checkpoints: ${WORK_DIR}/logs/"
echo "Preview meshes: ${WORK_DIR}/logs/mesh_stage*.ply"
echo "Final mesh: ${WORK_DIR}/skull_final_b200.ply"
echo ""
echo "To monitor GPU usage in another terminal:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "To view training progress:"
echo "  tail -f ${WORK_DIR}/logs/train_stage*.log"
echo -e "${BLUE}============================================${NC}"
