#!/bin/bash

# Complete Augenblick Setup Script
# This script sets up the entire environment and builds everything from scratch

echo "=== Complete Augenblick Setup ==="
echo "This script will:"
echo "1. Set up the conda environment"
echo "2. Install all dependencies"
echo "3. Initialize submodules"
echo "4. Build NeuS2 (requires GPU node)"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
AUGENBLICK_DIR="$SCRIPT_DIR"

# Function to check if we're on a GPU node
check_gpu_node() {
    if command -v nvidia-smi &> /dev/null; then
        echo "✓ GPU detected"
        return 0
    else
        echo "✗ No GPU detected - you're likely on a login node"
        echo "To build NeuS2, please run this script on a GPU node:"
        echo "  srun --partition=hpg-turin --gpus=1 --time=01:00:00 --pty bash"
        echo "  source $0"
        return 1
    fi
}

# Load required modules
echo "Loading modules..."
module load conda/25.3.1
module load cuda/12.4.1

# Initialize conda
eval "$(/apps/conda/25.3.1/bin/conda shell.bash hook)"

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "^augenblick "; then
    echo "Creating augenblick conda environment..."
    cd "$AUGENBLICK_DIR"
    conda env create -n augenblick -f environment.yml
fi

# Activate the environment
echo "Activating augenblick environment..."
conda activate augenblick

# Set CUDA environment variables
export CUDA_HOME=/apps/compilers/cuda/12.4.1
export CUDA_PATH=/apps/compilers/cuda/12.4.1
export CUDACXX=/apps/compilers/cuda/12.4.1/bin/nvcc
export PATH=/apps/compilers/cuda/12.4.1/bin:$PATH
export LD_LIBRARY_PATH=/apps/compilers/cuda/12.4.1/lib64:$LD_LIBRARY_PATH

# Ensure Python path includes conda env
export PATH=/blue/arthur.porto-biocosmos/jhennessy7.gatech/.conda/envs/augenblick/bin:$PATH

# Install PyTorch with CUDA support
echo "Checking PyTorch installation..."
python -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing PyTorch with CUDA 11.8 support..."
    pip install torch==2.3.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "✓ PyTorch already installed"
fi

# Change to project directory
cd "$AUGENBLICK_DIR"

# Initialize git submodules
echo "Initializing git submodules..."
git submodule update --init --recursive

# Clone VGG-T if needed
if [ ! -d "src/vggt" ] || [ -z "$(ls -A src/vggt 2>/dev/null)" ]; then
    echo "Cloning VGG-T..."
    cd src
    git clone https://github.com/facebookresearch/vggt
    cd ..
fi

# Build NeuS2 if on GPU node
if check_gpu_node; then
    echo ""
    echo "=== Building NeuS2 ==="
    cd "$AUGENBLICK_DIR/src/NeuS2"
    
    # Create dummy GL headers
    mkdir -p $HOME/dummy_gl/GL
    echo "// Dummy GL header for headless build" > $HOME/dummy_gl/GL/gl.h
    echo "#ifndef __gl_h_" >> $HOME/dummy_gl/GL/gl.h
    echo "#define __gl_h_" >> $HOME/dummy_gl/GL/gl.h
    echo "typedef unsigned int GLenum;" >> $HOME/dummy_gl/GL/gl.h
    echo "typedef unsigned int GLuint;" >> $HOME/dummy_gl/GL/gl.h
    echo "#endif" >> $HOME/dummy_gl/GL/gl.h
    
    # Clean and build
    echo "Cleaning previous build..."
    rm -rf build/
    
    echo "Configuring CMake..."
    cmake . -B build \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DNGP_BUILD_WITH_GUI=OFF \
      -DCMAKE_CUDA_COMPILER=/apps/compilers/cuda/12.4.1/bin/nvcc \
      -DCUDA_TOOLKIT_ROOT_DIR=/apps/compilers/cuda/12.4.1 \
      -DCMAKE_CUDA_ARCHITECTURES=80 \
      -DCMAKE_CXX_FLAGS="-I$HOME/dummy_gl" \
      -DCMAKE_CUDA_FLAGS="-I$HOME/dummy_gl"
    
    echo "Building Python bindings..."
    cmake --build build --target pyngp -j$(nproc)
    
    # Create symlink
    if [ -f "build/pyngp.cpython-310-x86_64-linux-gnu.so" ]; then
        ln -sf build/pyngp.cpython-310-x86_64-linux-gnu.so pyngp.so
        echo "✓ NeuS2 built successfully"
        
        # Test import
        echo "Testing NeuS2 import..."
        cd "$AUGENBLICK_DIR/src/NeuS2"
        python -c "import sys; sys.path.insert(0, '.'); import pyngp as ngp; print('✓ NeuS2 import successful!')" || echo "✗ NeuS2 import failed"
    else
        echo "✗ NeuS2 build failed"
    fi
    
    cd "$AUGENBLICK_DIR"
fi

# Set Python paths
export PYTHONPATH=$PYTHONPATH:$AUGENBLICK_DIR/src:$AUGENBLICK_DIR/src/NeuS2:$AUGENBLICK_DIR/src/vggt

# Create convenience scripts
echo "Creating convenience scripts..."

# Create run_on_gpu.sh
cat > "$AUGENBLICK_DIR/run_on_gpu.sh" << 'EOF'
#!/bin/bash
# Request GPU node and run command
srun --partition=hpg-turin --gpus=1 --time=01:00:00 --pty bash -c "source ~/augenblick/complete_augenblick_setup.sh && $*"
EOF
chmod +x "$AUGENBLICK_DIR/run_on_gpu.sh"

# Create test_pipeline.sh
cat > "$AUGENBLICK_DIR/test_pipeline.sh" << 'EOF'
#!/bin/bash
#SBATCH --job-name=test_pipeline
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mem=32GB
#SBATCH --output=test_pipeline_%j.out

source ~/augenblick/complete_augenblick_setup.sh

echo "Testing imports..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
cd ~/augenblick/src/NeuS2
python -c "import sys; sys.path.insert(0, '.'); import pyngp; print('NeuS2: OK')"
cd ~/augenblick
python -c "from src.vggt import *; print('VGG-T: OK')" 2>/dev/null || echo "VGG-T: Check import path"

echo "Environment ready for pipeline!"
EOF
chmod +x "$AUGENBLICK_DIR/test_pipeline.sh"

# Final status report
echo ""
echo "=== Setup Summary ==="
echo "Project directory: $AUGENBLICK_DIR"
echo "Python: $(which python)"
echo "CUDA: $CUDA_HOME"
echo ""

# Check status
python -c "import torch; print(f'✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')" 2>/dev/null || echo "✗ PyTorch not working"

if [ -f "$AUGENBLICK_DIR/src/NeuS2/pyngp.so" ]; then
    echo "✓ NeuS2 built"
else
    echo "✗ NeuS2 not built (run on GPU node)"
fi

if [ -d "$AUGENBLICK_DIR/src/vggt" ]; then
    echo "✓ VGG-T cloned"
else
    echo "✗ VGG-T not found"
fi

echo ""
echo "=== Next Steps ==="
if ! check_gpu_node &>/dev/null; then
    echo "To complete NeuS2 build:"
    echo "  ./run_on_gpu.sh"
    echo ""
    echo "Or manually:"
    echo "  srun --partition=hpg-turin --gpus=1 --time=01:00:00 --pty bash"
    echo "  source $0"
else
    echo "Setup complete! To test the pipeline:"
    echo "  sbatch test_pipeline.sh"
fi

echo ""
echo "To use this environment in future sessions:"
echo "  source $0"
