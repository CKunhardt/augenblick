#!/usr/bin/env python3
"""
Robust COLMAP → Neuralangelo pipeline for 3D reconstruction
Handles the complete workflow from images to mesh
"""

import os
import sys
import subprocess
import json
import shutil
import numpy as np
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class COLMAPNeuralangelo:
    def __init__(self, image_dir: str, output_dir: str, gpu_index: int = 0):
        """
        Initialize pipeline
        
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
        self.sparse_dir = self.colmap_dir / "sparse"
        self.database_path = self.colmap_dir / "database.db"
        self.neuralangelo_dir = self.output_dir / "neuralangelo_data"
        
        # Create directories
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        self.neuralangelo_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Pipeline initialized:")
        logger.info(f"  Images: {self.image_dir}")
        logger.info(f"  Output: {self.output_dir}")
    
    def run_command(self, cmd: List[str], description: str = None) -> subprocess.CompletedProcess:
        """Run command with logging"""
        if description:
            logger.info(f"=== {description} ===")
        
        logger.debug(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stdout:
                logger.debug(result.stdout)
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            if e.stdout:
                logger.error(f"STDOUT: {e.stdout}")
            if e.stderr:
                logger.error(f"STDERR: {e.stderr}")
            raise
    
    def step1_feature_extraction(self):
        """Extract features using COLMAP"""
        cmd = [
            "colmap", "feature_extractor",
            "--database_path", str(self.database_path),
            "--image_path", str(self.image_dir),
            "--ImageReader.camera_model", "SIMPLE_PINHOLE",
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", "1",
            "--SiftExtraction.gpu_index", str(self.gpu_index)
        ]
        
        self.run_command(cmd, "Extracting features")
    
    def step2_feature_matching(self):
        """Match features between images"""
        cmd = [
            "colmap", "exhaustive_matcher",
            "--database_path", str(self.database_path),
            "--SiftMatching.use_gpu", "1",
            "--SiftMatching.gpu_index", str(self.gpu_index)
        ]
        
        self.run_command(cmd, "Matching features")
    
    def step3_sparse_reconstruction(self):
        """Run sparse reconstruction"""
        cmd = [
            "colmap", "mapper",
            "--database_path", str(self.database_path),
            "--image_path", str(self.image_dir),
            "--output_path", str(self.sparse_dir),
            "--Mapper.num_threads", "16",
            "--Mapper.init_min_tri_angle", "4",
            "--Mapper.multiple_models", "0",
            "--Mapper.extract_colors", "0"
        ]
        
        self.run_command(cmd, "Sparse reconstruction")
    
    def step4_colmap_to_neuralangelo(self):
        """Convert COLMAP output to Neuralangelo format"""
        logger.info("=== Converting COLMAP to Neuralangelo format ===")
        
        # Find the reconstruction (usually in '0' subdirectory)
        recon_dir = self.sparse_dir / "0"
        if not recon_dir.exists():
            # Try to find any numbered directory
            numbered_dirs = [d for d in self.sparse_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            if numbered_dirs:
                recon_dir = numbered_dirs[0]
            else:
                raise ValueError(f"No reconstruction found in {self.sparse_dir}")
        
        # Export to text format for easier parsing
        text_dir = self.colmap_dir / "text"
        text_dir.mkdir(exist_ok=True)
        
        cmd = [
            "colmap", "model_converter",
            "--input_path", str(recon_dir),
            "--output_path", str(text_dir),
            "--output_type", "TXT"
        ]
        
        self.run_command(cmd, "Exporting COLMAP model to text")
        
        # Parse COLMAP output
        cameras = self._parse_colmap_cameras(text_dir / "cameras.txt")
        images = self._parse_colmap_images(text_dir / "images.txt")
        
        # Create transforms.json for Neuralangelo
        transforms = self._create_transforms_json(cameras, images)
        
        # Save transforms
        transforms_path = self.neuralangelo_dir / "transforms.json"
        with open(transforms_path, 'w') as f:
            json.dump(transforms, f, indent=2)
        
        # Copy images to Neuralangelo directory
        images_out = self.neuralangelo_dir / "images"
        images_out.mkdir(exist_ok=True)
        
        for img_info in images.values():
            src = self.image_dir / img_info['name']
            dst = images_out / img_info['name']
            if src.exists():
                shutil.copy2(src, dst)
        
        logger.info(f"✓ Converted to Neuralangelo format at {self.neuralangelo_dir}")
    
    def _parse_colmap_cameras(self, cameras_file: Path) -> Dict:
        """Parse COLMAP cameras.txt"""
        cameras = {}
        
        with open(cameras_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
                parts = line.split()
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                
                if model == "SIMPLE_PINHOLE":
                    f = float(parts[4])
                    cx = float(parts[5])
                    cy = float(parts[6])
                    
                    cameras[camera_id] = {
                        'model': model,
                        'width': width,
                        'height': height,
                        'f': f,
                        'cx': cx,
                        'cy': cy
                    }
                elif model == "PINHOLE":
                    fx = float(parts[4])
                    fy = float(parts[5])
                    cx = float(parts[6])
                    cy = float(parts[7])
                    
                    cameras[camera_id] = {
                        'model': model,
                        'width': width,
                        'height': height,
                        'fx': fx,
                        'fy': fy,
                        'cx': cx,
                        'cy': cy
                    }
        
        return cameras
    
    def _parse_colmap_images(self, images_file: Path) -> Dict:
        """Parse COLMAP images.txt"""
        images = {}
        
        with open(images_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('#') or not line:
                i += 1
                continue
            
            parts = line.split()
            if len(parts) >= 10:
                image_id = int(parts[0])
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                camera_id = int(parts[8])
                name = parts[9]
                
                images[image_id] = {
                    'name': name,
                    'camera_id': camera_id,
                    'qvec': [qw, qx, qy, qz],
                    'tvec': [tx, ty, tz]
                }
                
                # Skip the points2D line
                i += 2
            else:
                i += 1
        
        return images
    
    def _create_transforms_json(self, cameras: Dict, images: Dict) -> Dict:
        """Create transforms.json for Neuralangelo"""
        # Get the first (and usually only) camera
        camera_id = list(cameras.keys())[0]
        camera = cameras[camera_id]
        
        transforms = {
            "camera_model": "OPENCV",
            "frames": []
        }
        
        # Set camera intrinsics
        if camera['model'] == "SIMPLE_PINHOLE":
            transforms["fl_x"] = camera['f']
            transforms["fl_y"] = camera['f']
        else:  # PINHOLE
            transforms["fl_x"] = camera['fx']
            transforms["fl_y"] = camera['fy']
        
        transforms["cx"] = camera['cx']
        transforms["cy"] = camera['cy']
        transforms["w"] = camera['width']
        transforms["h"] = camera['height']
        
        # Convert each image
        for img_id, img_info in images.items():
            # Convert quaternion and translation to 4x4 matrix
            c2w = self._qvec_tvec_to_matrix(img_info['qvec'], img_info['tvec'])
            
            # Neuralangelo expects camera-to-world matrix
            frame = {
                "file_path": f"images/{img_info['name']}",
                "transform_matrix": c2w.tolist()
            }
            
            transforms["frames"].append(frame)
        
        return transforms
    
    def _qvec_tvec_to_matrix(self, qvec: List[float], tvec: List[float]) -> np.ndarray:
        """Convert COLMAP quaternion and translation to 4x4 matrix"""
        qw, qx, qy, qz = qvec
        
        # Quaternion to rotation matrix
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])
        
        # Create 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec
        
        # COLMAP uses a different coordinate system than Neuralangelo
        # You might need to adjust this transformation
        # This is a common coordinate system conversion:
        coord_change = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        
        return T @ coord_change
    
    def step5_train_neuralangelo(self, config_path: str = None, max_steps: int = 50000):
        """Train Neuralangelo model"""
        logger.info("=== Training Neuralangelo ===")
        
        # Use default config if not provided
        if config_path is None:
            config_path = os.path.expanduser("~/src/neuralangelo/projects/neuralangelo/configs/custom/object.yaml")
        
        workspace = self.output_dir / "neuralangelo_logs"
        workspace.mkdir(exist_ok=True)
        
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
        
        self.run_command(cmd, f"Training Neuralangelo for {max_steps} steps")
        
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
        
        cmd = [
            "torchrun", "--standalone", "--nproc_per_node=1",
            os.path.expanduser("~/src/neuralangelo/projects/neuralangelo/scripts/extract_mesh.py"),
            "--config", str(workspace / "config.yaml"),
            "--checkpoint", str(checkpoint_path),
            "--output_file", str(mesh_path),
            "--resolution", str(resolution),
            "--block_res", "128"
        ]
        
        self.run_command(cmd, "Extracting mesh")
        
        if mesh_path.exists():
            logger.info(f"✓ Mesh saved to: {mesh_path}")
            return mesh_path
        else:
            raise ValueError("Mesh extraction failed")
    
    def run_full_pipeline(self, max_steps: int = 50000, mesh_resolution: int = 2048):
        """Run the complete pipeline"""
        logger.info("Starting COLMAP → Neuralangelo pipeline")
        
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
    parser = argparse.ArgumentParser(description="COLMAP to Neuralangelo pipeline")
    parser.add_argument("image_dir", help="Directory containing input images")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--max_steps", type=int, default=50000, help="Training steps")
    parser.add_argument("--mesh_resolution", type=int, default=2048, help="Mesh extraction resolution")
    parser.add_argument("--skip_colmap", action="store_true", help="Skip COLMAP if already done")
    parser.add_argument("--skip_training", action="store_true", help="Skip training if checkpoint exists")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = COLMAPNeuralangelo(
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
