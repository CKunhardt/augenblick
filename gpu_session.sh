#!/bin/bash
# Quick GPU session launcher with automatic setup

echo "Requesting GPU node..."

# Default values
PARTITION="${1:-hpg-turin}"
TIME="${2:-02:00:00}"
CPUS="${3:-8}"
MEM="${4:-32GB}"

echo "Configuration:"
echo "  Partition: $PARTITION"
echo "  Time: $TIME"
echo "  CPUs: $CPUS"
echo "  Memory: $MEM"

# Create a temporary setup script that runs on the GPU node
cat > /tmp/gpu_init_$$.sh << 'EOF'
#!/bin/bash
# This runs ON the GPU node after allocation

# Source the setup script if it exists
if [ -f ~/augenblick/setup_gpu_env.sh ]; then
    source ~/augenblick/setup_gpu_env.sh
else
    echo "Setup script not found. Please run:"
    echo "  source ~/augenblick/setup_gpu_env.sh"
fi

# Start interactive shell
exec bash
EOF

chmod +x /tmp/gpu_init_$$.sh

# Request GPU node and run setup
srun --partition=$PARTITION \
     --gpus=1 \
     --cpus-per-task=$CPUS \
     --mem=$MEM \
     --time=$TIME \
     --pty /tmp/gpu_init_$$.sh

# Clean up
rm -f /tmp/gpu_init_$$.sh
