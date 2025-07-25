#!/usr/bin/env bash

set -euo pipefail

# Configuration
BLUE_ROOT="/blue/arthur.porto-biocosmos/jhennessy7.gatech"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="$BLUE_ROOT/neural_stable_${TIMESTAMP}"

# Random port to avoid conflicts
RANDOM_PORT=$((29501 + RANDOM % 100))

echo "=============================================="
echo "Neuralangelo Stable Training"
echo "Output: $OUT_ROOT"
echo "Port: $RANDOM_PORT"
echo "=============================================="

# Create directories
mkdir -p "$OUT_ROOT"/{data,logs,meshes}

# Copy data
echo "Copying data..."
SOURCE_DATA="$HOME/scratch/full_pipeline_output/neuralangelo"
cp -r "$SOURCE_DATA/images" "$OUT_ROOT/data/"
cp "$SOURCE_DATA/transforms"*.json "$OUT_ROOT/data/"

# Copy working config
cp "$SOURCE_DATA/config_b200_optimal.yaml" "$OUT_ROOT/config.yaml"
sed -i "s|root: ./|root: $OUT_ROOT/data|g" "$OUT_ROOT/config.yaml"

echo "Data ready: $(ls -1 "$OUT_ROOT/data/images" | wc -l) images"
echo "Config: config_b200_optimal.yaml (150k iterations)"

# Navigate to neuralangelo
cd "$HOME/augenblick/src/neuralangelo"

# Set environment
export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=localhost
export MASTER_PORT=$RANDOM_PORT

echo ""
echo "Starting training with torchrun..."
echo "This will take 8-12 hours for 150k iterations"
echo ""

# Run without timeout - let it run fully
torchrun --nproc_per_node=1 \
	    --master_addr=$MASTER_ADDR \
	        --master_port=$MASTER_PORT \
		    train.py \
		        --config "$OUT_ROOT/config.yaml" \
			    --logdir "$OUT_ROOT/logs" \
			        > "$OUT_ROOT/training.log" 2>&1 &

TRAIN_PID=$!
echo "Training PID: $TRAIN_PID"

# Wait longer to check if it's really running
echo "Waiting for training to initialize..."
sleep 30

if ps -p $TRAIN_PID > /dev/null 2>&1; then
	    echo ""
	        echo "=========================================="
		    echo "✓ Training is running successfully!"
		        echo "=========================================="
			    echo ""
			        echo "Output directory: $OUT_ROOT"
				    echo "Process ID: $TRAIN_PID"
				        echo ""
					    echo "Commands:"
					        echo "  Monitor log:    tail -f $OUT_ROOT/training.log"
						    echo "  Check GPU:      nvidia-smi"
						        echo "  Check process:  ps -p $TRAIN_PID"
							    echo "  Kill if needed: kill $TRAIN_PID"
							        echo ""
								    echo "Training will save checkpoints every 10k iterations"
								        echo "You can safely detach tmux now (Ctrl+b d)"
									    
									    # Show first few iterations to confirm it's training
									        echo ""
										    echo "First few lines of training:"
										        tail -20 "$OUT_ROOT/training.log" | grep -E "iter|loss|Epoch" || echo "Waiting for iterations to start..."
										else
											    echo "Training process ended. Checking log..."
											        tail -50 "$OUT_ROOT/training.log"
fi

# Save session info
cat > "$OUT_ROOT/session_info.txt" << EOF
Neuralangelo Training Session
============================
Started: $(date)
PID: $TRAIN_PID
Host: $(hostname)
GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
Port: $RANDOM_PORT
Config: config_b200_optimal.yaml (150k iterations)
Data: $SOURCE_DATA (138 images)

To resume monitoring:
  tail -f $OUT_ROOT/training.log

To extract mesh after completion:
  cd $HOME/augenblick/src/neuralangelo
  python projects/neuralangelo/scripts/extract_mesh.py \\
    --config $OUT_ROOT/config.yaml \\
    --checkpoint $OUT_ROOT/logs/checkpoint_latest.pth \\
    --output_path $OUT_ROOT/meshes \\
    --resolution 1024
EOF

echo ""
echo "Session info saved to: $OUT_ROOT/session_info.txt"
