#!/usr/bin/env python3
"""
COLMAP → Neuralangelo pipeline using pycolmap Python API
No external binaries needed!
"""

import os
import sys
import json
import shutil
import numpy as np
from pathlib import Path
import argparse
import logging
import pycolmap
from typing import Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PyColmapNeuralangelo:
    def __init__(self, image_dir: str, output_dir: str, gpu_index: int = 0):
        """
        Initialize pipeline using pycolmap
        
        Args:
            image_dir: Directory containing input images
            output_dir: Root directory for all outputs
            gpu_index: GPU to use (default: 0)
        """
        self.image_dir = Path(image_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.gpu_index = gpu_index
        
        # Setup directories
        self.colmap_dir = self.output_dir / "colmap"
        self.database_path = self.colmap_dir / "database.db"
        self.sparse_dir = self.colmap_dir / "sparse/0"
        self.neuralangelo_dir = self.output_dir / "neuralangelo_data"
        
        # Create directories
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        self.neuralangelo_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Pipeline initialized:")
        logger.info(f"  Images: {self.image_dir}")
        logger.info(f"  Output: {self.output_dir}")
    
    def step1_feature_extraction(self):
        """Extract features using pycolmap"""
        logger.info("=== Extracting features ===")
        
        # Setup extraction options
        sift_options = pycolmap.SiftExtractionOptions()
        sift_options.use_gpu = True
        sift_options.gpu_index = str(self.gpu_index)  # Back to string
        sift_options.max_image_size = 3200  # Adjust based on your images
        sift_options.max_num_features = 8192
        
        # Force CPU thread count to 1 when using GPU
        sift_options.num_threads = 1
        
        # Check if GPU extraction is actually being used
        logger.info(f"SIFT options: use_gpu={sift_options.use_gpu}, gpu_index={sift_options.gpu_index}")
        logger.info(f"GPU index being used: {self.gpu_index}")
        
        # Setup reader options
        reader_options = pycolmap.ImageReaderOptions()
        reader_options.camera_model = "SIMPLE_PINHOLE"
        # Check if single_camera attribute exists (API might have changed)
        if hasattr(reader_options, 'single_camera'):
            reader_options.single_camera = True
        elif hasattr(reader_options, 'single_camera_per_folder'):
            reader_options.single_camera_per_folder = False
        # If neither exists, we'll just use the default
        
        # Extract features
        pycolmap.extract_features(
            database_path=str(self.database_path),
            image_path=str(self.image_dir),
            reader_options=reader_options,
            sift_options=sift_options
        )
        
        logger.info("✓ Feature extraction complete")
    
    def step2_feature_matching(self):
        """Match features between images"""
        logger.info("=== Matching features ===")
        
        # Setup matching options
        matching_options = pycolmap.SiftMatchingOptions()
        matching_options.use_gpu = True
        matching_options.gpu_index = str(self.gpu_index)  # Back to string
        matching_options.max_error = 4.0
        matching_options.max_num_matches = 32768
        
        # Exhaustive matching for small datasets
        pycolmap.match_exhaustive(
            database_path=str(self.database_path),
            sift_options=matching_options
        )
        
        logger.info("✓ Feature matching complete")
    
    def step3_sparse_reconstruction(self):
        """Run sparse reconstruction"""
        logger.info("=== Running sparse reconstruction ===")
        
        # Setup mapper options
        mapper_options = pycolmap.IncrementalMapperOptions()
        mapper_options.num_threads = 16
        mapper_options.init_min_tri_angle = 4.0
        mapper_options.multiple_models = False
        mapper_options.extract_colors = False
        
        # Run reconstruction
        maps = pycolmap.incremental_mapping(
            database_path=str(self.database_path),
            image_path=str(self.image_dir),
            output_path=str(self.colmap_dir / "sparse"),
            options=mapper_options
        )
        
        if not maps:
            raise ValueError("Reconstruction failed - no valid models")
        
        logger.info(f"✓ Reconstruction complete with {len(maps)} model(s)")
        
        # The reconstruction is already saved, but let's verify
        if not (self.sparse_dir / "cameras.bin").exists():
            raise ValueError(f"Reconstruction files not found in {self.sparse_dir}")
    
    def step4_colmap_to_neuralangelo(self):
        """Convert COLMAP output to Neuralangelo format"""
        logger.info("=== Converting COLMAP to Neuralangelo format ===")
        
        # Read the reconstruction
        reconstruction = pycolmap.Reconstruction(str(self.sparse_dir))
        
        # Get camera parameters (assuming single camera)
        if len(reconstruction.cameras) == 0:
            raise ValueError("No cameras found in reconstruction")
        
        camera_id = list(reconstruction.cameras.keys())[0]
        camera = reconstruction.cameras[camera_id]
        
        # Create transforms.json
        transforms = {
            "camera_model": "OPENCV",
            "frames": []
        }
        
        # Set intrinsics based on camera model
        if camera.model_name == "SIMPLE_PINHOLE":
            transforms["fl_x"] = camera.params[0]
            transforms["fl_y"] = camera.params[0]
            transforms["cx"] = camera.params[1]
            transforms["cy"] = camera.params[2]
        elif camera.model_name == "PINHOLE":
            transforms["fl_x"] = camera.params[0]
            transforms["fl_y"] = camera.params[1]
            transforms["cx"] = camera.params[2]
            transforms["cy"] = camera.params[3]
        else:
            raise ValueError(f"Unsupported camera model: {camera.model_name}")
        
        transforms["w"] = camera.width
        transforms["h"] = camera.height
        
        # Process each image
        images_out = self.neuralangelo_dir / "images"
        images_out.mkdir(exist_ok=True)
        
        for image_id, image in reconstruction.images.items():
            # Get camera-to-world transformation
            R = image.rotation_matrix()
            t = image.tvec
            
            # Create 4x4 transformation matrix
            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = t
            
            # COLMAP uses a different coordinate system than Neuralangelo
            # Apply coordinate transformation
            coord_change = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
            c2w = c2w @ coord_change
            
            # Copy image
            src_image = self.image_dir / image.name
            dst_image = images_out / image.name
            if src_image.exists():
                shutil.copy2(src_image, dst_image)
            
            # Add frame to transforms
            frame = {
                "file_path": f"images/{image.name}",
                "transform_matrix": c2w.tolist()
            }
            transforms["frames"].append(frame)
        
        # Save transforms
        transforms_path = self.neuralangelo_dir / "transforms.json"
        with open(transforms_path, 'w') as f:
            json.dump(transforms, f, indent=2)
        
        logger.info(f"✓ Converted {len(transforms['frames'])} frames to Neuralangelo format")
        logger.info(f"  Output: {self.neuralangelo_dir}")
    
    def step5_train_neuralangelo(self, config_path: str = None, max_steps: int = 50000):
        """Train Neuralangelo model"""
        logger.info("=== Training Neuralangelo ===")
        
        # Use default config if not provided
        if config_path is None:
            config_path = os.path.expanduser("~/src/neuralangelo/projects/neuralangelo/configs/custom/object.yaml")
        
        workspace = self.output_dir / "neuralangelo_logs"
        workspace.mkdir(exist_ok=True)
        
        # Build training command
        import subprocess
        cmd = [
            "torchrun", "--standalone", "--nproc_per_node=1",
            os.path.expanduser("~/src/neuralangelo/train.py"),
            "--config", config_path,
            "--data.source", str(self.neuralangelo_dir),
            "--data.root", str(self.neuralangelo_dir),
            "--trainer.output_dir", str(workspace),
            "--trainer.max_steps", str(max_steps),
            "--trainer.val_check_interval", "5000",
            "--trainer.save_interval", "10000"
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stdout:
                logger.debug(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            if e.stdout:
                logger.error(f"STDOUT: {e.stdout}")
            if e.stderr:
                logger.error(f"STDERR: {e.stderr}")
            raise
        
        logger.info("✓ Training complete")
        return workspace
    
    def step6_extract_mesh(self, checkpoint_path: str = None, resolution: int = 2048):
        """Extract mesh from trained model"""
        logger.info("=== Extracting mesh ===")
        
        workspace = self.output_dir / "neuralangelo_logs"
        
        # Find latest checkpoint if not provided
        if checkpoint_path is None:
            checkpoints = list(workspace.glob("**/*.pth"))
            if not checkpoints:
                raise ValueError(f"No checkpoints found in {workspace}")
            checkpoint_path = max(checkpoints, key=os.path.getmtime)
        
        mesh_path = self.output_dir / "final_mesh.ply"
        
        # Build extraction command
        import subprocess
        cmd = [
            "torchrun", "--standalone", "--nproc_per_node=1",
            os.path.expanduser("~/src/neuralangelo/projects/neuralangelo/scripts/extract_mesh.py"),
            "--config", str(workspace / "config.yaml"),
            "--checkpoint", str(checkpoint_path),
            "--output_file", str(mesh_path),
            "--resolution", str(resolution),
            "--block_res", "128"
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stdout:
                logger.debug(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"Mesh extraction failed: {e}")
            if e.stdout:
                logger.error(f"STDOUT: {e.stdout}")
            if e.stderr:
                logger.error(f"STDERR: {e.stderr}")
            raise
        
        if mesh_path.exists():
            logger.info(f"✓ Mesh saved to: {mesh_path}")
            return mesh_path
        else:
            raise ValueError("Mesh extraction failed - output file not found")
    
    def run_full_pipeline(self, max_steps: int = 50000, mesh_resolution: int = 2048):
        """Run the complete pipeline"""
        logger.info("Starting PyColmap → Neuralangelo pipeline")
        
        try:
            # COLMAP reconstruction
            self.step1_feature_extraction()
            self.step2_feature_matching()
            self.step3_sparse_reconstruction()
            
            # Convert to Neuralangelo format
            self.step4_colmap_to_neuralangelo()
            
            # Train Neuralangelo
            workspace = self.step5_train_neuralangelo(max_steps=max_steps)
            
            # Extract mesh
            mesh_path = self.step6_extract_mesh(resolution=mesh_resolution)
            
            logger.info("✓ Pipeline completed successfully!")
            logger.info(f"Final mesh: {mesh_path}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="PyColmap to Neuralangelo pipeline")
    parser.add_argument("image_dir", help="Directory containing input images")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--max_steps", type=int, default=50000, help="Training steps")
    parser.add_argument("--mesh_resolution", type=int, default=2048, help="Mesh extraction resolution")
    parser.add_argument("--skip_colmap", action="store_true", help="Skip COLMAP if already done")
    parser.add_argument("--skip_training", action="store_true", help="Skip training if checkpoint exists")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PyColmapNeuralangelo(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        gpu_index=args.gpu
    )
    
    # Run pipeline
    if args.skip_colmap:
        logger.info("Skipping COLMAP reconstruction")
        pipeline.step4_colmap_to_neuralangelo()
        
        if not args.skip_training:
            pipeline.step5_train_neuralangelo(max_steps=args.max_steps)
        
        pipeline.step6_extract_mesh(resolution=args.mesh_resolution)
    else:
        pipeline.run_full_pipeline(
            max_steps=args.max_steps,
            mesh_resolution=args.mesh_resolution
        )


if __name__ == "__main__":
    main()
