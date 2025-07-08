#!/bin/bash
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=13:00:00
#SBATCH --job-name=neus2_b2uild
#SBATCH --output=neus2_buildw.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

source ~/augenblick/complete_augenblick_setup.sh

cmake . -B build -DNGP_BUILD_WITH_GUI=OFF -DNGP_BUILD_EXECUTABLE=OFF -DNGP_BUILD_WITH_PYTHON_BINDINGS=ON
cmake --build build --target pyngp

