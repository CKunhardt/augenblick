#!/bin/bash
# Automatic GPU environment setup for Augenblick pipeline

echo "=== Augenblick GPU Environment Setup ==="
echo "Running on: $(hostname)"
echo "Date: $(date)"

# --- GPU CHECK ---
check_gpu() {
    if nvidia-smi &>/dev/null; then
        echo "✓ GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
        return 0
    else
        echo "✗ No GPU detected - this script should be run on a GPU node"
        return 1
    fi
}

if ! check_gpu; then
    echo "Please run on a GPU node: srun --partition=hpg-turin --gpus=1 --time=02:00:00 --pty bash"
    exit 1
fi

# --- SETUP PATHS ---
AUGENBLICK_DIR="$HOME/augenblick"
NEUS2_DIR="$AUGENBLICK_DIR/src/NeuS2"
CACHED_SO="$AUGENBLICK_DIR/build_artifacts/pyngp.so"
NEUS2_SO="$NEUS2_DIR/pyngp.so"
NEUS2_BUILD="$NEUS2_DIR/instant-ngp/build/pyngp.cpython-310-x86_64-linux-gnu.so"

echo "[INFO] Changing to $AUGENBLICK_DIR"
cd "$AUGENBLICK_DIR" || exit 1

# --- LOAD MODULES ---
echo "[INFO] Loading modules..."
module load conda/25.3.1
module load cuda/12.4.1

# --- CONDA ENV ---
eval "$(/apps/conda/25.3.1/bin/conda shell.bash hook)"
echo "[INFO] Activating conda environment 'augenblick'"
conda activate augenblick

# --- CUDA ENV VARS ---
export CUDA_HOME=/apps/compilers/cuda/12.4.1
export CUDA_PATH=/apps/compilers/cuda/12.4.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# --- OPTIX ENV VARS ---
export OPTIX_ROOT=$HOME/optix/SDK
export OptiX_INSTALL_DIR=$OPTIX_ROOT
export CPLUS_INCLUDE_PATH=$OPTIX_ROOT/include:$CPLUS_INCLUDE_PATH
export LIBRARY_PATH=$OPTIX_ROOT/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$OPTIX_ROOT/lib64:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$OPTIX_ROOT:$CMAKE_PREFIX_PATH

# --- PYTHON PATHS ---
export PYTHONPATH=$PYTHONPATH:$AUGENBLICK_DIR/src:$NEUS2_DIR
export PATH=/blue/arthur.porto-biocosmos/jhennessy7.gatech/.conda/envs/augenblick/bin:$PATH

# --- NEUS2 STATUS CHECK ---
echo "[INFO] Checking NeuS2 status..."
NEED_BUILD=false

if [ -f "$NEUS2_BUILD" ]; then
    echo "[✓] Found local build: $NEUS2_BUILD"
    ln -sf "$NEUS2_BUILD" "$NEUS2_SO"
elif [ -f "$CACHED_SO" ]; then
    echo "[✓] Using cached build: $CACHED_SO"
    mkdir -p "$(dirname "$NEUS2_BUILD")"
    cp "$CACHED_SO" "$NEUS2_BUILD"
    ln -sf "$NEUS2_BUILD" "$NEUS2_SO"
else
    echo "[✗] No existing pyngp.so found"
    NEED_BUILD=true
fi

if [ "$NEED_BUILD" = false ]; then
    echo "[INFO] Testing NeuS2 import..."
    if python -c "import sys; sys.path.insert(0, '$NEUS2_DIR'); import pyngp; print('✓ NeuS2 is ready')" 2>/dev/null; then
        echo "✓ NeuS2 ready to use"
    else
        echo "[WARN] NeuS2 import failed — triggering rebuild"
        NEED_BUILD=true
    fi
fi

# --- BUILD NEUS2 IF NEEDED ---
if [ "$NEED_BUILD" = true ]; then
    echo "[BUILD] Building NeuS2 from source..."
    cd "$NEUS2_DIR" || exit 1

    echo "[BUILD] Creating dummy GL headers..."
    mkdir -p $HOME/dummy_gl/GL
    cat > $HOME/dummy_gl/GL/gl.h << 'EOF'
#ifndef __gl_h_
#define __gl_h_
typedef unsigned int GLenum;
typedef unsigned int GLuint;
#endif
EOF

    echo "[BUILD] Running CMake..."
    rm -rf instant-ngp/build
    mkdir -p instant-ngp/build
    cd instant-ngp/build

    cmake ../.. \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DNGP_BUILD_WITH_GUI=OFF \
        -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES=80 \
        -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME \
        -DCMAKE_CUDA_FLAGS="-I$HOME/dummy_gl" \
        -DCMAKE_CXX_FLAGS="-I$HOME/dummy_gl"

    echo "[BUILD] Compiling pyngp..."
    make pyngp -j8

    echo "[BUILD] Linking shared object..."
    cd "$NEUS2_DIR"
    ln -sf instant-ngp/build/pyngp.cpython-310-x86_64-linux-gnu.so pyngp.so

    echo "[BUILD] Saving build to cache..."
    mkdir -p "$AUGENBLICK_DIR/build_artifacts"
    cp "$NEUS2_BUILD" "$CACHED_SO"

    echo "[TEST] Testing import again..."
    if python -c "import sys; sys.path.insert(0, '$NEUS2_DIR'); import pyngp; print('✓ NeuS2 built successfully')" 2>/dev/null; then
        echo "✓ NeuS2 built and validated"
    else
        echo "✗ NeuS2 build failed"
    fi
fi

# --- PYTHON PACKAGE CHECK ---
echo ""
echo "[INFO] Verifying PyTorch install..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')" || {
    echo "[INSTALL] Installing PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
}

# --- ALIASES ---
echo ""
echo "[INFO] Defining shell aliases..."
alias cdneus='cd $NEUS2_DIR'
alias cdpipe='cd $AUGENBLICK_DIR/src/pipeline'
alias testneus='python -c "import sys; sys.path.insert(0, \"$NEUS2_DIR\"); import pyngp; print(\"NeuS2 OK\")"'

# --- FINAL STATUS ---
echo ""
echo "=== Environment Status ==="
echo "Project dir:     $AUGENBLICK_DIR"
echo "Python:          $(which python)"
echo "CUDA:            $CUDA_HOME"
echo "GPU:             $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

echo "=== Useful Commands ==="
echo "Test NeuS2:      testneus"
echo "Go to NeuS2:     cdneus"
echo "Go to pipeline:  cdpipe"
echo ""
echo "Run reconstruction:"
echo "  python run_complete_pipeline.py ~/scratch/images_from_dropbox/noscale --output_dir ~/scratch/test_run"
echo ""

touch ~/.augenblick_gpu_setup
echo "✓ GPU environment ready!"

