#!/bin/bash
#SBATCH --job-name=rebuild_neus2
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mem=32GB
#SBATCH --output=rebuild_neus2_%j.out

set -e

echo "=== Rebuilding NeuS2 on GPU node ==="
echo "Current node: $(hostname)"
echo "Running as: $(whoami)"
echo ""

# Load CUDA
module purge
module load cuda/12.4.1

# Activate conda
source ~/.bashrc
conda activate augenblick

# CUDA sanity check
echo "Using nvcc: $(which nvcc)"
nvcc --version

# Clean previous build
cd ~/augenblick/src/NeuS2
rm -rf build/
mkdir build && cd build

# Rebuild with explicit CUDA include path (for cublas_v2.h)
cmake .. \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DNGP_BUILD_WITH_GUI=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_CUDA_COMPILER=$(which nvcc) \
  -DCMAKE_CUDA_FLAGS="-I$HOME/dummy_gl -I/apps/compilers/cuda/12.4.1/include" \
  -DCMAKE_CXX_FLAGS="-I$HOME/dummy_gl -I/apps/compilers/cuda/12.4.1/include" \
  -DCUDA_TOOLKIT_ROOT_DIR=/apps/compilers/cuda/12.4.1

make pyngp -j$(nproc)

# Link and test import
cd ..
ln -sf build/pyngp.cpython-310-x86_64-linux-gnu.so pyngp.so
export LD_LIBRARY_PATH=/apps/compilers/cuda/12.4.1/lib64:$LD_LIBRARY_PATH

echo ""
echo "Testing pyngp import..."
python -c "import sys; sys.path.insert(0, '.'); import pyngp as ngp; print('✓ NeuS2 import successful!')" || echo "✗ NeuS2 import failed"

echo ""
echo "=== Done ==="

