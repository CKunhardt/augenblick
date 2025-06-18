# Augenblick: A Photogrammetry Pipeline based on VGG-T + NeuS2

A high-quality 3D reconstruction pipeline that combines VGG-T feature extraction with NeuS2 neural surface reconstruction to generate detailed meshed geometry from multi-view images.

## Overview

This pipeline leverages state-of-the-art computer vision and neural rendering techniques to create accurate 3D models from photogrammetric input. The system is designed for high-quality surface reconstruction with robust feature matching and neural implicit surface representation.

### Key Features

- **Multi-view stereo reconstruction** with neural implicit surfaces
- **Robust feature extraction** using VGG-T architecture
- **High-quality mesh generation** through NeuS2's neural surface reconstruction
- **End-to-end pipeline** from raw images to textured 3D models
- **Scalable processing** for varying numbers of input views

## Architecture

The pipeline consists of two main components working in sequence:

### 1. VGG-T Feature Extraction
VGG-T (Vision Graph Neural Network - Transformer) serves as our feature extraction backbone, providing:
- Dense feature maps from input images
- Multi-scale feature representations
- Robust feature matching across viewpoints
- Camera pose estimation refinement

### 2. NeuS2 Neural Surface Reconstruction
NeuS2 takes the VGG-T output and performs neural implicit surface reconstruction:
- Learns implicit surface representations from multi-view features
- Generates high-quality surface normals and geometry
- Produces watertight meshes with fine detail preservation
- Supports texture reconstruction and material estimation

### Pipeline Flow
```
Input Images → VGG-T Feature Extraction → Camera Poses + Features → NeuS2 Reconstruction → 3D Mesh Output
```

The VGG-T component processes input images to extract dense features and estimate camera poses, which are then fed into NeuS2 for volumetric neural surface reconstruction and final mesh generation.

## Requirements

### System Requirements
- Python 3.10
- CUDA-capable GPU (recommended: RTX 3080 or better)
- 16GB+ RAM
- 50GB+ available storage

### Dependencies
VGG-T and NeuS2 must be configured individually within their subdirectories.

See https://github.com/NVlabs/instant-ngp#building-instant-ngp-windows--linux and https://github.com/19reborn/NeuS2 for NeuS2 setup (PyTorch must be installed with CUDA)

See https://github.com/facebookresearch/vggt for VGG-T (mostly just installing python depedencies)

We recommend using an environment manager such as Anaconda, and have provided an `environment.yml` to facilitate creation under the vggt directory.

Please note that conda doesn't seem to support installing PyTorch with CUDA, so this will have to be installed through pip, i.e. `pip install torch==2.3.1+cu118`

## Installation

```bash
git clone --recursive [repository-url]
cd augenblick

conda env create -n augenblick -f environment.yml
conda actiavte augenblick

cd src/vggt
pip install -r requirements.txt

cd ../NeuS2
cmake . -B build
# If above doesn't work, try: cmake . -B build -T v143,version=14.36
cmake --build build --config RelWithDebInfo -j 
pip install -r requirements.txt

cd ../data
# Install datasets here, i.e. https://roboimagedata.compute.dtu.dk/?page_id=36

```

## Project Structure

```
# Project structure will be documented here
photogrammetry-pipeline/
├── src/
│   ├── vggt/           # VGG-T implementation
│   ├── neus2/          # NeuS2 implementation
│   ├── pipeline/       # Main pipeline orchestration
│   └── utils/          # Utility functions
├── config/            # Configuration files
├── data/              # Sample data and datasets
├── output/           # Generated outputs
├── tests/             # Unit tests
└── docs/              # Additional documentation
```

## Usage

### Quick Start
```bash
cd pipeline
python run_pipeline.py --input_dir ../data/path/to/images --output_dir ../output
```
## Build Instructions

### Testing
```bash
# Testing instructions and test suite information
```

## Input Requirements

### Image Specifications
- **Format**: JPG, PNG, or TIFF
- **Resolution**: Minimum 1024x768, recommended 2048x1536 or higher
- **Overlap**: 60-80% overlap between adjacent views recommended
- **Lighting**: Consistent lighting conditions across all views
- **Focus**: Sharp focus with minimal motion blur

### Camera Setup Recommendations
- Use consistent camera settings (ISO, aperture, white balance)
- Capture from multiple viewpoints with good baseline separation
- Include sufficient texture detail for feature matching
- Avoid reflective or transparent surfaces when possible

## Output Formats

The pipeline generates several outputs:
- **3D Mesh**: `.ply`, `.obj`, or `.glb` files
- **Texture Maps**: High-resolution texture maps
- **Camera Parameters**: Estimated camera poses and intrinsics
- **Feature Maps**: Intermediate VGG-T feature representations
- **Quality Metrics**: Reconstruction accuracy and completeness scores

## Performance Optimization

### GPU Memory Management

### Processing Time Estimates

## Troubleshooting

## Examples

### Sample Datasets
- [Links to example datasets]
- [Expected outputs for validation]

### Benchmark Results
- [Performance benchmarks on standard datasets]
- [Quality metrics and comparisons]

## Acknowledgements & Citations

If you use this pipeline in your research, please cite:

```bibtex
@software{photogrammetry_pipeline_2025,
  title={Photogrammetry Pipeline: VGG-T + NeuS2},
  author={[Clinton Kunhardt and James Hennessey and Charles Clark and Caleb Wheeler and Xin Lin and Bree Wang and Arthur Porto]},
  year={2025},
  url={[repository-url]}
}
```

### Related Work Citations
```bibtex
@misc{wang2023neus2fastlearningneural,
      title={NeuS2: Fast Learning of Neural Implicit Surfaces for Multi-view Reconstruction}, 
      author={Yiming Wang and Qin Han and Marc Habermann and Kostas Daniilidis and Christian Theobalt and Lingjie Liu},
      year={2023},
      eprint={2212.05231},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2212.05231}, 
}
@misc{wang2025vggtvisualgeometrygrounded,
      title={VGGT: Visual Geometry Grounded Transformer}, 
      author={Jianyuan Wang and Minghao Chen and Nikita Karaev and Andrea Vedaldi and Christian Rupprecht and David Novotny},
      year={2025},
      eprint={2503.11651},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.11651}, 
}
```

## Support

For questions, issues, or feature requests:
- **Issues**: [Link to issue tracker]
- **Discussions**: [Link to discussions forum]
- **Email**: ckunhardt3@gatech.edu

---

*Last updated: 2025-06-*