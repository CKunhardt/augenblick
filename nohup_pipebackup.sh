#!/bin/bash

# Nohup runner for the masked reconstruction pipeline with VGG-T
# Usage: ./run_pipeline_nohup.sh <input_dir> <output_dir> [options]

# Function to display usage
usage() {
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
    echo ""
    echo "Pipeline flow: Images → VGG-T → [COLMAP] → Neuralangelo → Mesh"
    echo "Note: COLMAP step is optional with --skip-colmap flag"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/images /path/to/output"
    echo "  $0 /path/to/images /path/to/output --gpu 1 --max_steps 30000"
    echo "  $0 /path/to/images /path/to/output --mesh_resolution 8192 --block_res 256  # B200 8k mode"
    echo "  $0 /path/to/images /path/to/output --vggt_path /home/user/vgg-t --skip-colmap"
    echo ""
    exit 1
}

# Check if at least 2 arguments are provided
if [ $# -lt 2 ]; then
    echo "Error: Missing required arguments"
    usage
fi

# Required arguments
INPUT_DIR="$1"
OUTPUT_DIR="$2"
shift 2

# Default values
GPU_INDEX=0
MAX_STEPS=50000
MESH_RESOLUTION=2048
BLOCK_RES=128
PYTHON_SCRIPT="./masked_reconstruction_vggt.py"
VGGT_PATH="${VGGT_PATH:-}"
SKIP_COLMAP=false
CONFIG_TEMPLATE="/home/jhennessy7.gatech/augenblick/src/neuralangelo/projects/neuralangelo/configs/base.yaml"

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
            SKIP_COLMAP=true
            shift
            ;;
	--config-template)
            CONFIG_TEMPLATE="--config-template $2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate input directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist"
    exit 1
fi

# Validate Python script
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Generate a unique log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/pipeline_vggt_${TIMESTAMP}.log"
PID_FILE="${OUTPUT_DIR}/pipeline_vggt_${TIMESTAMP}.pid"

# Prepare the command
COMMAND="CUDA_VISIBLE_DEVICES=$GPU_INDEX python $PYTHON_SCRIPT \"$INPUT_DIR\" \"$OUTPUT_DIR\" --gpu $GPU_INDEX --max_steps $MAX_STEPS --mesh_resolution $MESH_RESOLUTION --block_res $BLOCK_RES"

# Add skip-colmap flag if set
if [ "$SKIP_COLMAP" = true ]; then
    COMMAND="$COMMAND --skip-colmap"
fi

# Add VGG-T path to environment if specified
if [ -n "$VGGT_PATH" ]; then
    export VGGT_PATH="$VGGT_PATH"
fi

# Auto-detect high-performance mode for B200
if [ $MESH_RESOLUTION -ge 4096 ] && [ $BLOCK_RES -eq 128 ]; then
    echo "⚠ High resolution mode detected (resolution=$MESH_RESOLUTION)"
    echo "  For B200 GPU, consider using --block_res 256 for better performance"
    echo ""
fi

# Display information
echo "=============================================="
echo "Masked Reconstruction Pipeline with VGG-T"
echo "=============================================="
if [ "$SKIP_COLMAP" = true ]; then
    echo "Pipeline: Images → VGG-T → Neuralangelo (COLMAP skipped)"
else
    echo "Pipeline: Images → VGG-T → COLMAP → Neuralangelo"
fi
echo ""
echo "Input directory:    $INPUT_DIR"
echo "Output directory:   $OUTPUT_DIR"
echo "GPU index:          $GPU_INDEX"
echo "Max steps:          $MAX_STEPS"
echo "Mesh resolution:    $MESH_RESOLUTION"
echo "Block resolution:   $BLOCK_RES"
echo "Python script:      $PYTHON_SCRIPT"
echo "VGG-T path:         ${VGGT_PATH:-<not set, will skip VGG-T>}"
echo "Skip COLMAP:        $SKIP_COLMAP"
echo "Log file:           $LOG_FILE"
echo "PID file:           $PID_FILE"
echo "Command:            $COMMAND"
echo ""

# Check for dependencies
echo "Checking dependencies..."
if [ -n "$VGGT_PATH" ] && [ -d "$VGGT_PATH" ]; then
    echo "✓ VGG-T source found at $VGGT_PATH"
else
    echo "⚠ VGG-T source not found. Pipeline will skip VGG-T initialization."
    echo "  Set VGGT_PATH environment variable or use --vggt_path to enable VGG-T"
fi

# Check if COLMAP is available (only if not skipping)
if [ "$SKIP_COLMAP" = false ]; then
    if command -v colmap &> /dev/null; then
        echo "✓ COLMAP found in PATH"
    elif module list 2>&1 | grep -q colmap; then
        echo "✓ COLMAP available via module system"
    else
        echo "⚠ COLMAP not found. Will attempt to use module system during execution."
    fi
else
    echo "ℹ COLMAP step will be skipped"
    if [ -z "$VGGT_PATH" ] || [ ! -d "$VGGT_PATH" ]; then
        echo "⚠ WARNING: Both VGG-T and COLMAP are disabled/unavailable!"
        echo "  The pipeline needs at least one method for camera pose estimation."
        echo "  Either enable VGG-T by setting VGGT_PATH or remove --skip-colmap flag."
    fi
fi

# GPU memory check for high resolution modes
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    
    if [ $MESH_RESOLUTION -ge 8192 ]; then
        echo ""
        echo "⚠ Ultra-high resolution mode ($MESH_RESOLUTION)"
        echo "  Recommended settings for B200: --block_res 256"
        echo "  This will require significant GPU memory"
    fi
fi

echo ""

# Ask for confirmation
read -p "Do you want to start the pipeline? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Pipeline cancelled."
    exit 0
fi

# Start the pipeline with nohup
echo "Starting pipeline in background..."
if [ -n "$VGGT_PATH" ]; then
    nohup bash -c "export VGGT_PATH='$VGGT_PATH'; $COMMAND" > "$LOG_FILE" 2>&1 &
else
    nohup bash -c "$COMMAND" > "$LOG_FILE" 2>&1 &
fi
PIPELINE_PID=$!

# Save the PID
echo $PIPELINE_PID > "$PID_FILE"

echo "Pipeline started successfully!"
echo "Process ID: $PIPELINE_PID"
echo "Log file: $LOG_FILE"
echo "PID file: $PID_FILE"
echo ""
echo "Pipeline stages:"
echo "  1. Organize dataset"
echo "  2. Run VGG-T (if available)"
if [ "$SKIP_COLMAP" = false ]; then
    echo "  3. Run COLMAP (with VGG-T init if available)"
else
    echo "  3. Skip COLMAP (using VGG-T output directly)"
fi
echo "  4. Convert to Neuralangelo format"
echo "  5. Train Neuralangelo"
echo "  6. Extract mesh (resolution=$MESH_RESOLUTION, block_res=$BLOCK_RES)"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To check if still running:"
echo "  ps -p $PIPELINE_PID"
echo ""
echo "To stop the pipeline:"
echo "  kill $PIPELINE_PID"
echo "  # or"
echo "  kill \$(cat $PID_FILE)"
echo ""
echo "The pipeline will continue running even if you disconnect from SSH."
echo "Check the log file for progress and results."

# Optional: Show the first few lines of the log to confirm it started
sleep 2
if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "First few lines of output:"
    echo "----------------------------------------"
    head -10 "$LOG_FILE"
    echo "----------------------------------------"
    echo "Use 'tail -f $LOG_FILE' to see live output"
fi

# Create a status checking script
STATUS_SCRIPT="${OUTPUT_DIR}/check_status_${TIMESTAMP}.sh"
cat > "$STATUS_SCRIPT" << EOF
#!/bin/bash
# Status checker for pipeline PID $PIPELINE_PID

PID=$PIPELINE_PID
LOG_FILE="$LOG_FILE"
SKIP_COLMAP=$SKIP_COLMAP
MESH_RESOLUTION=$MESH_RESOLUTION
BLOCK_RES=$BLOCK_RES

if ps -p \$PID > /dev/null; then
    echo "Pipeline is RUNNING (PID: \$PID)"
    echo ""
    echo "Recent log entries:"
    tail -20 "\$LOG_FILE" | grep -E "(Step|complete|error|failed)"
    
    # Check GPU usage if nvidia-smi is available
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "GPU Usage:"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
    fi
else
    echo "Pipeline has FINISHED (PID: \$PID)"
    echo ""
    echo "Final status:"
    tail -50 "\$LOG_FILE" | grep -E "(COMPLETED|FAILED|Final|Error)"
fi

echo ""
echo "Configuration:"
echo "  Mesh resolution: \$MESH_RESOLUTION"
echo "  Block resolution: \$BLOCK_RES"

echo ""
echo "Output directory contents:"
ls -la "$OUTPUT_DIR"

if [ -f "$OUTPUT_DIR/final_mesh.ply" ]; then
    echo ""
    echo "✓ Final mesh generated!"
    ls -lh "$OUTPUT_DIR/final_mesh.ply"
fi

# Check for intermediate outputs
echo ""
echo "Intermediate outputs:"
if [ -d "$OUTPUT_DIR/vggt/sparse" ]; then
    echo "✓ VGG-T sparse reconstruction found"
fi
if [ "\$SKIP_COLMAP" = "false" ] && [ -d "$OUTPUT_DIR/colmap/sparse/0" ]; then
    echo "✓ COLMAP sparse reconstruction found"
fi
if [ -f "$OUTPUT_DIR/neuralangelo/transforms.json" ]; then
    echo "✓ Neuralangelo transforms.json found"
fi
if [ -d "$OUTPUT_DIR/logs/checkpoints" ]; then
    echo "✓ Training checkpoints found:"
    ls -1 "$OUTPUT_DIR/logs/checkpoints/" | tail -5
fi

# Calculate elapsed time
if [ -f "$PID_FILE" ]; then
    START_TIME=\$(stat -c %Y "$PID_FILE" 2>/dev/null || stat -f %B "$PID_FILE" 2>/dev/null)
    if [ -n "\$START_TIME" ]; then
        CURRENT_TIME=\$(date +%s)
        ELAPSED=\$((CURRENT_TIME - START_TIME))
        echo ""
        echo "Elapsed time: \$(printf '%02d:%02d:%02d' \$((ELAPSED/3600)) \$((ELAPSED%3600/60)) \$((ELAPSED%60)))"
    fi
fi
EOF

chmod +x "$STATUS_SCRIPT"
echo ""
echo "Status checking script created: $STATUS_SCRIPT"
echo "Run it anytime to check pipeline status"#!/bin/bash

# Nohup runner for the masked reconstruction pipeline with VGG-T
# Usage: ./run_pipeline_nohup.sh <input_dir> <output_dir> [options]

# Function to display usage
usage() {
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
    echo ""
    echo "Pipeline flow: Images → VGG-T → [COLMAP] → Neuralangelo → Mesh"
    echo "Note: COLMAP step is optional with --skip-colmap flag"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/images /path/to/output"
    echo "  $0 /path/to/images /path/to/output --gpu 1 --max_steps 30000"
    echo "  $0 /path/to/images /path/to/output --mesh_resolution 8192 --block_res 256  # B200 8k mode"
    echo "  $0 /path/to/images /path/to/output --vggt_path /home/user/vgg-t --skip-colmap"
    echo ""
    exit 1
}

# Check if at least 2 arguments are provided
if [ $# -lt 2 ]; then
    echo "Error: Missing required arguments"
    usage
fi

# Required arguments
INPUT_DIR="$1"
OUTPUT_DIR="$2"
shift 2

# Default values
GPU_INDEX=0
MAX_STEPS=50000
MESH_RESOLUTION=2048
BLOCK_RES=128
PYTHON_SCRIPT="./masked_reconstruction_vggt.py"
VGGT_PATH="${VGGT_PATH:-}"
SKIP_COLMAP=false

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
            SKIP_COLMAP=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate input directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist"
    exit 1
fi

# Validate Python script
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Generate a unique log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${OUTPUT_DIR}/pipeline_vggt_${TIMESTAMP}.log"
PID_FILE="${OUTPUT_DIR}/pipeline_vggt_${TIMESTAMP}.pid"

# Prepare the command
COMMAND="python $PYTHON_SCRIPT \"$INPUT_DIR\" \"$OUTPUT_DIR\" --gpu $GPU_INDEX --max_steps $MAX_STEPS --mesh_resolution $MESH_RESOLUTION --block_res $BLOCK_RES"

# Add skip-colmap flag if set
if [ "$SKIP_COLMAP" = true ]; then
    COMMAND="$COMMAND --skip-colmap"
fi

# Add VGG-T path to environment if specified
if [ -n "$VGGT_PATH" ]; then
    export VGGT_PATH="$VGGT_PATH"
fi

# Auto-detect high-performance mode for B200
if [ $MESH_RESOLUTION -ge 4096 ] && [ $BLOCK_RES -eq 128 ]; then
    echo "⚠ High resolution mode detected (resolution=$MESH_RESOLUTION)"
    echo "  For B200 GPU, consider using --block_res 256 for better performance"
    echo ""
fi

# Display information
echo "=============================================="
echo "Masked Reconstruction Pipeline with VGG-T"
echo "=============================================="
if [ "$SKIP_COLMAP" = true ]; then
    echo "Pipeline: Images → VGG-T → Neuralangelo (COLMAP skipped)"
else
    echo "Pipeline: Images → VGG-T → COLMAP → Neuralangelo"
fi
echo ""
echo "Input directory:    $INPUT_DIR"
echo "Output directory:   $OUTPUT_DIR"
echo "GPU index:          $GPU_INDEX"
echo "Max steps:          $MAX_STEPS"
echo "Mesh resolution:    $MESH_RESOLUTION"
echo "Block resolution:   $BLOCK_RES"
echo "Python script:      $PYTHON_SCRIPT"
echo "VGG-T path:         ${VGGT_PATH:-<not set, will skip VGG-T>}"
echo "Skip COLMAP:        $SKIP_COLMAP"
echo "Log file:           $LOG_FILE"
echo "PID file:           $PID_FILE"
echo "Command:            $COMMAND"
echo ""

# Check for dependencies
echo "Checking dependencies..."
if [ -n "$VGGT_PATH" ] && [ -d "$VGGT_PATH" ]; then
    echo "✓ VGG-T source found at $VGGT_PATH"
else
    echo "⚠ VGG-T source not found. Pipeline will skip VGG-T initialization."
    echo "  Set VGGT_PATH environment variable or use --vggt_path to enable VGG-T"
fi

# Check if COLMAP is available (only if not skipping)
if [ "$SKIP_COLMAP" = false ]; then
    if command -v colmap &> /dev/null; then
        echo "✓ COLMAP found in PATH"
    elif module list 2>&1 | grep -q colmap; then
        echo "✓ COLMAP available via module system"
    else
        echo "⚠ COLMAP not found. Will attempt to use module system during execution."
    fi
else
    echo "ℹ COLMAP step will be skipped"
    if [ -z "$VGGT_PATH" ] || [ ! -d "$VGGT_PATH" ]; then
        echo "⚠ WARNING: Both VGG-T and COLMAP are disabled/unavailable!"
        echo "  The pipeline needs at least one method for camera pose estimation."
        echo "  Either enable VGG-T by setting VGGT_PATH or remove --skip-colmap flag."
    fi
fi

# GPU memory check for high resolution modes
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    
    if [ $MESH_RESOLUTION -ge 8192 ]; then
        echo ""
        echo "⚠ Ultra-high resolution mode ($MESH_RESOLUTION)"
        echo "  Recommended settings for B200: --block_res 256"
        echo "  This will require significant GPU memory"
    fi
fi

echo ""

# Ask for confirmation
read -p "Do you want to start the pipeline? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Pipeline cancelled."
    exit 0
fi

# Start the pipeline with nohup
echo "Starting pipeline in background..."
if [ -n "$VGGT_PATH" ]; then
    nohup bash -c "export VGGT_PATH='$VGGT_PATH'; $COMMAND" > "$LOG_FILE" 2>&1 &
else
    nohup bash -c "$COMMAND" > "$LOG_FILE" 2>&1 &
fi
PIPELINE_PID=$!

# Save the PID
echo $PIPELINE_PID > "$PID_FILE"

echo "Pipeline started successfully!"
echo "Process ID: $PIPELINE_PID"
echo "Log file: $LOG_FILE"
echo "PID file: $PID_FILE"
echo ""
echo "Pipeline stages:"
echo "  1. Organize dataset"
echo "  2. Run VGG-T (if available)"
if [ "$SKIP_COLMAP" = false ]; then
    echo "  3. Run COLMAP (with VGG-T init if available)"
else
    echo "  3. Skip COLMAP (using VGG-T output directly)"
fi
echo "  4. Convert to Neuralangelo format"
echo "  5. Train Neuralangelo"
echo "  6. Extract mesh (resolution=$MESH_RESOLUTION, block_res=$BLOCK_RES)"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To check if still running:"
echo "  ps -p $PIPELINE_PID"
echo ""
echo "To stop the pipeline:"
echo "  kill $PIPELINE_PID"
echo "  # or"
echo "  kill \$(cat $PID_FILE)"
echo ""
echo "The pipeline will continue running even if you disconnect from SSH."
echo "Check the log file for progress and results."

# Optional: Show the first few lines of the log to confirm it started
sleep 2
if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "First few lines of output:"
    echo "----------------------------------------"
    head -10 "$LOG_FILE"
    echo "----------------------------------------"
    echo "Use 'tail -f $LOG_FILE' to see live output"
fi

# Create a status checking script
STATUS_SCRIPT="${OUTPUT_DIR}/check_status_${TIMESTAMP}.sh"
cat > "$STATUS_SCRIPT" << EOF
#!/bin/bash
# Status checker for pipeline PID $PIPELINE_PID

PID=$PIPELINE_PID
LOG_FILE="$LOG_FILE"
SKIP_COLMAP=$SKIP_COLMAP
MESH_RESOLUTION=$MESH_RESOLUTION
BLOCK_RES=$BLOCK_RES

if ps -p \$PID > /dev/null; then
    echo "Pipeline is RUNNING (PID: \$PID)"
    echo ""
    echo "Recent log entries:"
    tail -20 "\$LOG_FILE" | grep -E "(Step|complete|error|failed)"
    
    # Check GPU usage if nvidia-smi is available
    if command -v nvidia-smi &> /dev/null; then
        echo ""
        echo "GPU Usage:"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
    fi
else
    echo "Pipeline has FINISHED (PID: \$PID)"
    echo ""
    echo "Final status:"
    tail -50 "\$LOG_FILE" | grep -E "(COMPLETED|FAILED|Final|Error)"
fi

echo ""
echo "Configuration:"
echo "  Mesh resolution: \$MESH_RESOLUTION"
echo "  Block resolution: \$BLOCK_RES"

echo ""
echo "Output directory contents:"
ls -la "$OUTPUT_DIR"

if [ -f "$OUTPUT_DIR/final_mesh.ply" ]; then
    echo ""
    echo "✓ Final mesh generated!"
    ls -lh "$OUTPUT_DIR/final_mesh.ply"
fi

# Check for intermediate outputs
echo ""
echo "Intermediate outputs:"
if [ -d "$OUTPUT_DIR/vggt/sparse" ]; then
    echo "✓ VGG-T sparse reconstruction found"
fi
if [ "\$SKIP_COLMAP" = "false" ] && [ -d "$OUTPUT_DIR/colmap/sparse/0" ]; then
    echo "✓ COLMAP sparse reconstruction found"
fi
if [ -f "$OUTPUT_DIR/neuralangelo/transforms.json" ]; then
    echo "✓ Neuralangelo transforms.json found"
fi
if [ -d "$OUTPUT_DIR/logs/checkpoints" ]; then
    echo "✓ Training checkpoints found:"
    ls -1 "$OUTPUT_DIR/logs/checkpoints/" | tail -5
fi

# Calculate elapsed time
if [ -f "$PID_FILE" ]; then
    START_TIME=\$(stat -c %Y "$PID_FILE" 2>/dev/null || stat -f %B "$PID_FILE" 2>/dev/null)
    if [ -n "\$START_TIME" ]; then
        CURRENT_TIME=\$(date +%s)
        ELAPSED=\$((CURRENT_TIME - START_TIME))
        echo ""
        echo "Elapsed time: \$(printf '%02d:%02d:%02d' \$((ELAPSED/3600)) \$((ELAPSED%3600/60)) \$((ELAPSED%60)))"
    fi
fi
EOF

chmod +x "$STATUS_SCRIPT"
echo ""
echo "Status checking script created: $STATUS_SCRIPT"
echo "Run it anytime to check pipeline status"
