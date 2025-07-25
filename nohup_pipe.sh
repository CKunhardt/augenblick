#!/bin/bash

# Default values
GPU_INDEX=0
MAX_STEPS=50000
MESH_RESOLUTION=2048
BLOCK_RES=128
PYTHON_SCRIPT="./masked_reconstruction_vggt.py"
VGGT_PATH="${VGGT_PATH:-}"
SKIP_COLMAP=""
CONFIG_TEMPLATE=""

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir> [options]"
    echo ""
    echo "Required arguments:"
    echo "  input_dir      Directory containing images and masks"
    echo "  output_dir     Output directory for all results"
    echo ""
    echo "Optional arguments:"
    echo "  --gpu INDEX           GPU index to use (default: 0)"
    echo "  --max_steps STEPS     Training steps (default: 50000)"
    echo "  --mesh_resolution RES Mesh resolution (default: 2048)"
    echo "  --block_res SIZE      Block size for mesh extraction (default: 128, use 256 for B200)"
    echo "  --python_script PATH  Path to the Python script (default: ./masked_reconstruction_vggt.py)"
    echo "  --vggt_path PATH      Path to VGG-T source (default: from VGGT_PATH env var)"
    echo "  --skip-colmap         Skip COLMAP step and use VGG-T output directly"
    echo "  --config-template PATH Path to YAML config template"
    echo ""
    echo "Pipeline flow: Images → VGG-T → [COLMAP] → Neuralangelo → Mesh"
    echo "Note: COLMAP step is optional with --skip-colmap flag"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/images /path/to/output"
    echo "  $0 /path/to/images /path/to/output --gpu 1 --max_steps 30000"
    echo "  $0 /path/to/images /path/to/output --mesh_resolution 8192 --block_res 256  # B200 8k mode"
    echo "  $0 /path/to/images /path/to/output --config-template configs/b200_template.yaml"
    echo ""
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
shift 2

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_INDEX="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --mesh_resolution)
            MESH_RESOLUTION="$2"
            shift 2
            ;;
        --block_res)
            BLOCK_RES="$2"
            shift 2
            ;;
        --python_script)
            PYTHON_SCRIPT="$2"
            shift 2
            ;;
        --vggt_path)
            VGGT_PATH="$2"
            shift 2
            ;;
        --skip-colmap)
            SKIP_COLMAP="--skip-colmap"
            shift
            ;;
        --config-template)
            CONFIG_TEMPLATE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build the command
CMD="python $PYTHON_SCRIPT \"$INPUT_DIR\" \"$OUTPUT_DIR\" --gpu $GPU_INDEX --max_steps $MAX_STEPS --mesh_resolution $MESH_RESOLUTION --block_res $BLOCK_RES"

if [ -n "$VGGT_PATH" ]; then
    CMD="$CMD --vggt-path \"$VGGT_PATH\""
fi

if [ -n "$SKIP_COLMAP" ]; then
    CMD="$CMD $SKIP_COLMAP"
fi

if [ -n "$CONFIG_TEMPLATE" ]; then
    CMD="$CMD --config-template \"$CONFIG_TEMPLATE\""
fi

# Create log file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/pipeline_vggt_${TIMESTAMP}.log"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Show what we're running
echo "Running masked reconstruction pipeline with VGG-T"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "GPU: $GPU_INDEX"
echo "Max steps: $MAX_STEPS"
echo "Mesh resolution: $MESH_RESOLUTION"
echo "Block resolution: $BLOCK_RES"
echo "VGG-T path: ${VGGT_PATH:-[using default]}"
echo "Skip COLMAP: ${SKIP_COLMAP:-No}"
echo "Config template: ${CONFIG_TEMPLATE:-[using default]}"
echo "Log file: $LOG_FILE"
echo ""
echo "Command: $CMD"
echo ""

# Run with nohup
nohup bash -c "$CMD" > "$LOG_FILE" 2>&1 &
PID=$!

echo "Started process with PID: $PID"
echo "Monitor with: tail -f $LOG_FILE"
echo "Kill with: kill $PID"
