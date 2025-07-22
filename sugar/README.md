# SuGaR 3D Reconstruction Benchmark

This repository contains tools for benchmarking and comparing sparse and dense 3D reconstructions using the SuGaR (Surface-Guided Gaussian Splatting) framework.

## Overview

The benchmark compares sparse point clouds with dense reconstructions (either point clouds or meshes) across four key metrics:
1. **Point Size Comparison** - Total number of points/vertices
2. **Point Density by View** - Points per cubic unit for each camera view
3. **Surface Coverage by View** - Visibility ratio from rendered images
4. **Mesh Complexity/Point Density Ratio** - Mesh face counts or density ratios by view

## Key Files

### Main Scripts

- **`benchmark.py`** - Main benchmarking script that compares sparse and dense reconstructions
- **`view_sugar_results.py`** - Script for visualizing and rendering SuGaR results
- **`train_full_pipeline.py`** - Complete training pipeline for SuGaR models

### Data Directories

- **`data/`** - Input datasets and sparse reconstructions
  - `sparse/0/points.ply` - Sparse point cloud reconstructions (this can be generated using VGGT or Colmap)
- **`output/`** - Generated outputs
  - `vanilla_gs/` - Gaussian Splatting checkpoints
  - `refined_ply/` - Dense point cloud reconstructions
  - `refined_mesh/` - Mesh reconstructions (.obj files)
  - `metrics/` - Benchmark results and plots
- **`renders/`** - Rendered images from dense reconstructions
- **`sparse_renders/`** - Rendered images from sparse reconstructions


## Installation

**Reference the original repo: https://github.com/Anttwo/SuGaR**

## Usage

### Running the Benchmark

The main benchmark script compares sparse and dense reconstructions using rendered images:

```bash
python benchmark.py \
    --source_path ./data/YOUR_SCENE/noscale/ \
    --gs_checkpoint_path ./output/vanilla_gs/YOUR_SCENE/ \
    --sparse_ply_path ./data/YOUR_SCENE/noscale/sparse/0/points.ply \
    --dense_ply_path ./output/refined_ply/YOUR_SCENE/REFINED_MODEL.ply \
    --sparse_renders_dir ./sparse_renders/YOUR_SCENE/ \
    --dense_renders_dir ./renders/YOUR_SCENE/ \
    --output_dir ./output/metrics/plots
```

### Example with Your Data

Based on your existing structure:

```bash
python benchmark.py \
    --source_path ./data/UF_mammals_36342_skull/noscale/ \
    --gs_checkpoint_path ./output/vanilla_gs/noscale/ \
    --sparse_ply_path ./data/UF_mammals_36342_skull/noscale/sparse/0/points.ply \
    --dense_ply_path ./output/refined_ply/noscale/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim200000_normalconsistency01_gaussperface6.ply \
    --sparse_renders_dir ./sparse_renders/noscale/ \
    --dense_renders_dir ./renders/noscale/ \
    --output_dir ./output/metrics/plots
```

### Training a New Model

To train a complete SuGaR model from scratch:
## Run this script if you use the VGGT output as input for Sugar
```bash
python image_undistorter.py 

```

## Then start the train pipeline
```bash
python train_full_pipeline.py \
    --s ./data/YOUR_SCENE/ \
    -r dn_consistency \
    --refinement_time short \
    --low_poly True
```

### Viewing Results

To visualize and render SuGaR results:

```bash
python view_sugar_results.py \
    --scene_name YOUR_SCENE \
    --gs_checkpoint_path ./output/vanilla_gs/YOUR_SCENE/ \
    --refined_sugar_folder ./output/refined/YOUR_SCENE \
    --refined_iteration 2000 \
    --render_mesh \
    --output_dir ./output/YOUR_SCENE
    --camera_index 1
```

## Output

The benchmark generates:

1. **Plots** (`reconstruction_metrics.png`):
   - Point size comparison
   - Point density by view
   - Surface coverage by view
   - Mesh complexity or point density ratios

2. **JSON Results** (`benchmark_results.json`):
   - Raw numerical data for all metrics

## Understanding the Metrics

### Point Density
- Calculated as `number_of_visible_points / bounding_box_volume`
- Varies by view based on camera position and visible points
- Higher density indicates more detailed reconstruction in that view

### Surface Coverage
- Calculated from rendered images as `non_black_pixels / total_pixels`
- Measures how much of the image area is covered by the reconstruction
- Ratio compares sparse vs dense coverage

### Mesh Complexity
- For meshes: Shows vertex, face, and edge counts
- Uses Poisson surface reconstruction (enhanced by SuGaR)
- Face density varies by view based on mesh visibility


## File Formats

- **Input point clouds**: PLY format with XYZ coordinates
- **Input meshes**: OBJ format with vertices and faces
- **Rendered images**: PNG/JPG format
- **Camera data**: COLMAP format in Gaussian Splatting checkpoints

## Notes

- The benchmark uses SuGaR's camera system for view-based analysis
- Mesh extraction uses enhanced Poisson reconstruction
- Point density calculations use cubic volume measurements
- Surface coverage is computed directly from rendered images
- All metrics are calculated per camera view for detailed analysis

## Citation

If you use this benchmark in your research, please cite the original SuGaR paper:

```bibtex
@article{sugar2023,
    title={SuGaR: Surface-Guided Representation for Efficient 3D Gaussian Splatting},
    author={...},
    journal={...},
    year={2023}
}
```
