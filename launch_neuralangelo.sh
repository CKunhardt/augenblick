#!/bin/bash
# Complete Neuralangelo pipeline: VGGT processing → Training setup → Launch
# Can process new datasets or use existing processed data

# Load required modules
echo "Loading required modules..."
module load pytorch/2.7
module load cuda/12.8.1

# Verify modules loaded
echo "Loaded modules:"
module list 2>&1 | grep -E "(pytorch|cuda)"
echo ""

# Get the directory where this script is located (should be augenblick)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Base paths
SCRATCH_DIR="/blue/arthur.porto-biocosmos/jhennessy7.gatech/scratch"
NEURALANGELO_PATH="/home/jhennessy7.gatech/augenblick/src/neuralangelo"

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
RUN_VGGT=false
RAW_IMAGES=""
RAW_MASKS=""
USE_EXISTING=""
RESUME_FROM=""
DATASET_NAME=""
WORK_DIR=""
STRIDE=2
CAMERAS_PER_GROUP=46

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_NAME="$2"
            shift 2
            ;;
        --images)
            RAW_IMAGES="$2"
            RUN_VGGT=true
            shift 2
            ;;
        --masks)
            RAW_MASKS="$2"
            shift 2
            ;;
        --stride)
            STRIDE="$2"
            shift 2
            ;;
        --cameras-per-group)
            CAMERAS_PER_GROUP="$2"
            shift 2
            ;;
        --use-existing)
            USE_EXISTING="$2"
            WORK_DIR="$2"
            shift 2
            ;;
        --resume-from)
            RESUME_FROM="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "For new dataset (runs VGGT pipeline):"
            echo "  --dataset NAME       Dataset name (required for new data)"
            echo "  --images DIR         Directory with raw images"
            echo "  --masks DIR          Directory with masks"
            echo "  --stride N           Sample every Nth frame (default: 2)"
            echo "  --cameras-per-group N Camera group size (default: 46)"
            echo ""
            echo "For existing processed data:"
            echo "  --use-existing DIR   Use existing neuralangelo directory"
            echo "  --resume-from DIR    Resume from existing directory"
            echo ""
            echo "Examples:"
            echo "  # Process new dataset"
            echo "  $0 --dataset skull_v2 --images /path/to/images --masks /path/to/masks"
            echo ""
            echo "  # Use existing processed data"
            echo "  $0 --use-existing ${SCRATCH_DIR}/neuralangelo_b200_20250721_144006"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}============================================"
echo "Neuralangelo Complete Pipeline"
echo "============================================${NC}"

# Determine working directory
if [ -n "$USE_EXISTING" ] || [ -n "$RESUME_FROM" ]; then
    # Using existing directory
    if [ -n "$USE_EXISTING" ]; then
        WORK_DIR="$USE_EXISTING"
    else
        WORK_DIR="${SCRATCH_DIR}/neuralangelo_b200_${TIMESTAMP}"
    fi
elif [ -n "$DATASET_NAME" ]; then
    # New dataset - create directory
    WORK_DIR="${SCRATCH_DIR}/${DATASET_NAME}_neuralangelo_${TIMESTAMP}"
else
    # Default name
    WORK_DIR="${SCRATCH_DIR}/neuralangelo_b200_${TIMESTAMP}"
fi

echo "Working directory: ${WORK_DIR}"
echo ""

# Function to run VGGT pipeline
run_vggt_pipeline() {
    local images_dir="$1"
    local masks_dir="$2"
    local output_base="${WORK_DIR}/vggt_pipeline"
    
    echo -e "${BLUE}=========================================="
    echo "Running VGGT Pipeline"
    echo "==========================================${NC}"
    echo "Raw images: ${images_dir}"
    echo "Raw masks: ${masks_dir}"
    echo ""
    
    # Activate VGGT virtual environment if not already active
    if [ -z "$VIRTUAL_ENV" ] || [ "$VIRTUAL_ENV" != "/home/jhennessy7.gatech/neuralangelo_b200_env" ]; then
        echo "Activating VGGT virtual environment..."
        source /home/jhennessy7.gatech/neuralangelo_b200_env/bin/activate
        echo "Python: $(which python)"
        echo "Python version: $(python --version)"
    fi
    
    # Create pipeline directories
    local cropped_dir="${output_base}/cropped"
    local vggt_output="${output_base}/vggt_output"
    local fullres_output="${output_base}/fullres_transforms"
    
    mkdir -p "$cropped_dir" "$vggt_output" "$fullres_output"
    
    # Step 1: Crop images to focus on object
    echo -e "${YELLOW}Step 1: Cropping images to object region...${NC}"
    cd ${SCRIPT_DIR}
    
    if [ ! -d "${cropped_dir}/images" ]; then
        python prep_crop.py \
            "${images_dir}" \
            "${masks_dir}" \
            --stride ${STRIDE} \
            --cameras-per-group ${CAMERAS_PER_GROUP} \
            --out ${cropped_dir} \
            --make-square
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Cropping complete${NC}"
        else
            echo -e "${RED}✗ Cropping failed${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}✓ Cropped images already exist${NC}"
    fi
    
    # Check crop results
    CROP_COUNT=$(ls ${cropped_dir}/images/*.jpg 2>/dev/null | wc -l || echo "0")
    echo "  Cropped images: ${CROP_COUNT}"
    echo ""
    
    # Step 2: Process through VGGT for camera poses
    echo -e "${YELLOW}Step 2: Running VGGT for camera pose estimation...${NC}"
    
    if [ ! -f "${vggt_output}/transforms.json" ]; then
        python vggt_batch_final.py \
            ${cropped_dir}/images \
            --output_dir ${vggt_output} \
            --metadata ${cropped_dir}/metadata.json
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ VGGT processing complete${NC}"
        else
            echo -e "${RED}✗ VGGT processing failed${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}✓ VGGT output already exists${NC}"
    fi
    
    # Verify VGGT results
    if [ -f "${vggt_output}/transforms.json" ]; then
        NULL_COUNT=$(grep -c "null" ${vggt_output}/transforms.json | head -1 || echo "0")
        echo "  Frames with missing poses: ${NULL_COUNT}"
    fi
    echo ""
    
    # Step 3: Scale transforms back to original resolution
    echo -e "${YELLOW}Step 3: Scaling transforms to original resolution...${NC}"
    
    if [ ! -f "${fullres_output}/transforms.json" ]; then
        python scale_transforms_to_original.py \
            ${vggt_output}/transforms.json \
            ${cropped_dir}/metadata.json \
            ${images_dir} \
            --output_dir ${fullres_output}
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Transform scaling complete${NC}"
        else
            echo -e "${RED}✗ Transform scaling failed${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}✓ Scaled transforms already exist${NC}"
    fi
    
    # Copy final transforms to work directory
    cp "${fullres_output}/transforms.json" "${WORK_DIR}/transforms_source.json"
    
    # Create symlinks to original images and masks
    ln -sfn "${images_dir}" "${WORK_DIR}/images_fullres"
    ln -sfn "${masks_dir}" "${WORK_DIR}/masks_fullres"
    
    echo -e "${GREEN}✓ VGGT pipeline complete!${NC}"
    echo ""
}

# Function to setup working directory
setup_working_directory() {
    local source_dir=$1
    local target_dir=$2
    
    echo -e "${YELLOW}Setting up Neuralangelo working directory...${NC}"
    
    # Create directory structure
    mkdir -p "${target_dir}/logs"
    
    # Copy essential scripts
    echo "Copying training scripts..."
    cp -v "${source_dir}/launch_training.sh" "${target_dir}/" 2>/dev/null || cp -v "${source_dir}/run_full_pipeline.sh" "${target_dir}/launch_training.sh"
    cp -v "${source_dir}/staged_train.py" "${target_dir}/"
    
    # Copy stage configs
    echo "Copying stage configurations..."
    for config in stage1_coarse.yaml stage2_mid.yaml stage3_fine.yaml; do
        if [ -f "${source_dir}/${config}" ]; then
            cp -v "${source_dir}/${config}" "${target_dir}/"
        else
            echo -e "${YELLOW}Warning: ${config} not found in source directory${NC}"
        fi
    done
    
    # Copy helper scripts if they exist
    echo "Copying helper scripts..."
    for script in fix_checkpoint_iter.py scale_transforms_intrinsics.py merge_transforms.py; do
        [ -f "${source_dir}/${script}" ] && cp -v "${source_dir}/${script}" "${target_dir}/"
    done
    
    # Make scripts executable
    chmod +x "${target_dir}/launch_training.sh" 2>/dev/null
    chmod +x "${target_dir}/staged_train.py" 2>/dev/null
    
    # If not running VGGT, link to existing data
    if [ "$RUN_VGGT" = false ]; then
        # Try to find existing transforms
        local scaled_transforms="${SCRATCH_DIR}/scale_neuralangelo_fullres/transforms.json"
        if [ -f "$scaled_transforms" ] && [ ! -f "${target_dir}/transforms_source.json" ]; then
            cp "$scaled_transforms" "${target_dir}/transforms_source.json"
            echo "  Copied existing transforms.json"
        fi
        
        # Link to existing images/masks if not already linked
        if [ ! -e "${target_dir}/images_fullres" ]; then
            local fullres_images="${SCRATCH_DIR}/scale_organized/images"
            [ -d "$fullres_images" ] && ln -sfn "$fullres_images" "${target_dir}/images_fullres"
        fi
        
        if [ ! -e "${target_dir}/masks_fullres" ]; then
            local fullres_masks="${SCRATCH_DIR}/scale_organized/masks"
            [ -d "$fullres_masks" ] && ln -sfn "$fullres_masks" "${target_dir}/masks_fullres"
        fi
    fi
}

# Main execution logic
if [ -n "$USE_EXISTING" ]; then
    # Use existing directory
    if [ ! -d "$WORK_DIR" ]; then
        echo -e "${RED}ERROR: Directory does not exist: $WORK_DIR${NC}"
        exit 1
    fi
    echo -e "${GREEN}Using existing directory: $WORK_DIR${NC}"
    
elif [ -n "$RESUME_FROM" ]; then
    # Resume from existing directory
    if [ ! -d "$RESUME_FROM" ]; then
        echo -e "${RED}ERROR: Resume directory does not exist: $RESUME_FROM${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Creating new directory and copying from: $RESUME_FROM${NC}"
    
    # Create new directory and copy
    mkdir -p "${WORK_DIR}/logs"
    cp -r "${RESUME_FROM}"/* "${WORK_DIR}/"
    
    # Update with any new scripts
    setup_working_directory "$SCRIPT_DIR" "$WORK_DIR"
    
    echo -e "${GREEN}Ready to resume in new directory: $WORK_DIR${NC}"
    
else
    # Fresh start
    echo -e "${GREEN}Creating new directory: $WORK_DIR${NC}"
    mkdir -p "${WORK_DIR}/logs"
    
    # Run VGGT pipeline if requested
    if [ "$RUN_VGGT" = true ]; then
        if [ -z "$RAW_IMAGES" ] || [ ! -d "$RAW_IMAGES" ]; then
            echo -e "${RED}ERROR: Images directory not found: $RAW_IMAGES${NC}"
            exit 1
        fi
        
        if [ -z "$RAW_MASKS" ] || [ ! -d "$RAW_MASKS" ]; then
            echo -e "${RED}ERROR: Masks directory not found: $RAW_MASKS${NC}"
            exit 1
        fi
        
        run_vggt_pipeline "$RAW_IMAGES" "$RAW_MASKS"
    fi
    
    # Setup working directory
    setup_working_directory "$SCRIPT_DIR" "$WORK_DIR"
    
    # Create info file
    cat > "${WORK_DIR}/run_info.txt" << EOF
Neuralangelo Training Run
========================
Created: $(date)
Timestamp: ${TIMESTAMP}
Dataset: ${DATASET_NAME:-default}
GPU: NVIDIA B200
Script launched from: ${SCRIPT_DIR}
Working directory: ${WORK_DIR}

Modules loaded:
- pytorch/2.7
- cuda/12.8.1

Pipeline settings:
- VGGT pipeline run: ${RUN_VGGT}
- Stride: ${STRIDE}
- Cameras per group: ${CAMERAS_PER_GROUP}

Data sources:
- Images: ${RAW_IMAGES:-existing}
- Masks: ${RAW_MASKS:-existing}
- Transforms: ${WORK_DIR}/transforms_source.json

Training stages:
- Stage 1: 512×768 (0-2k iterations)
- Stage 2: 2080×3120 (2k-10k iterations)
- Stage 3: 4160×6240 (10k-20k iterations)
EOF
    
    echo -e "${GREEN}✓ Setup complete!${NC}"
fi

# Check required files
echo ""
echo "Checking required files..."
missing_files=0

for file in staged_train.py stage1_coarse.yaml stage2_mid.yaml stage3_fine.yaml; do
    if [ ! -f "${WORK_DIR}/${file}" ]; then
        echo -e "${RED}✗ Missing: ${file}${NC}"
        missing_files=$((missing_files + 1))
    else
        echo -e "${GREEN}✓ Found: ${file}${NC}"
    fi
done

# Check data files
if [ ! -f "${WORK_DIR}/transforms_source.json" ]; then
    echo -e "${RED}✗ Missing: transforms_source.json${NC}"
    missing_files=$((missing_files + 1))
else
    echo -e "${GREEN}✓ Found: transforms_source.json${NC}"
fi

if [ $missing_files -gt 0 ]; then
    echo -e "\n${RED}ERROR: Missing required files!${NC}"
    exit 1
fi

# Update YAML configs with correct paths
echo -e "\n${YELLOW}Updating config paths...${NC}"
for config in stage1_coarse.yaml stage2_mid.yaml stage3_fine.yaml; do
    if [ -f "${WORK_DIR}/${config}" ]; then
        # Update data root path
        sed -i "s|root:.*|root: ${WORK_DIR}|g" "${WORK_DIR}/${config}"
        echo "  Updated paths in ${config}"
    fi
done

# Summary
echo ""
echo -e "${BLUE}============================================"
echo "Pipeline Complete!"
echo "============================================${NC}"
echo "Working directory: ${WORK_DIR}"
echo ""

if [ "$RUN_VGGT" = true ]; then
    echo "VGGT Pipeline Results:"
    echo "  Cropped images: ${CROP_COUNT}"
    echo "  Camera poses: ✓"
    echo "  Scaled transforms: ✓"
    echo ""
fi

echo "To start training:"
echo -e "${GREEN}cd ${WORK_DIR}${NC}"
echo -e "${GREEN}./launch_training.sh${NC}"
echo ""
echo "To resume training:"
echo -e "${GREEN}cd ${WORK_DIR}${NC}"
echo -e "${GREEN}./launch_training.sh --resume${NC}"
echo ""
echo "To monitor progress:"
echo -e "${GREEN}tail -f ${WORK_DIR}/logs/train_stage*.log${NC}"
echo ""

# Ask if user wants to start training now (skip in batch mode)
if [ -n "$SLURM_JOB_ID" ]; then
    # Running in SLURM - automatically start training
    echo -e "${GREEN}Running in batch mode - automatically starting training${NC}"
    cd "$WORK_DIR"
    ./launch_training.sh
else
    # Interactive mode - ask user
    read -p "Start Neuralangelo training now? (y/n): " start_now
    
    if [ "$start_now" = "y" ] || [ "$start_now" = "Y" ]; then
        cd "$WORK_DIR"
        ./launch_training.sh
    fi
fi
