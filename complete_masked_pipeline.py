#!/usr/bin/env python3
"""
Complete pipeline for masked dataset: organize, COLMAP, convert, and train Neuralangelo
Run once to process everything from raw data to 3D mesh
"""

import os
import sys
import json
import shutil
import subprocess
import numpy as np
from pathlib import Path
import argparse
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MaskedReconstructionPipeline:
    def __init__(self, input_dir, output_dir, gpu_index=0):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.gpu_index = gpu_index
        
        # Setup all paths
        self.organized_dir = self.output_dir / "organized"
        self.images_dir = self.organized_dir / "images"
        self.masks_dir = self.organized_dir / "masks"
        self.colmap_dir = self.output_dir / "colmap"
        self.neuralangelo_dir = self.output_dir / "neuralangelo"
        self.logs_dir = self.output_dir / "logs"
        
        # Create base directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if we need to use module system for COLMAP
        self.use_module_colmap = False
        try:
            subprocess.run(["which", "colmap"], check=True, capture_output=True)
        except:
            self.use_module_colmap = True
            logger.info("COLMAP not in PATH, will use module system")
        
        logger.info("="*60)
        logger.info("Masked 3D Reconstruction Pipeline")
        logger.info("="*60)
        logger.info(f"Input: {self.input_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"GPU: {self.gpu_index}")
    
    def step1_organize_dataset(self):
        """Separate images and masks"""
        logger.info("\n=== Step 1: Organizing Dataset ===")
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all files
        all_files = list(self.input_dir.glob("*"))
        
        # Separate images and masks
        image_files = []
        mask_files = []
        
        for f in all_files:
            if f.is_file():
                if ".mask.png" in f.name:
                    mask_files.append(f)
                elif f.suffix.upper() in ['.JPG', '.JPEG', '.PNG'] and '.mask' not in f.name:
                    image_files.append(f)
        
        logger.info(f"Found {len(image_files)} images and {len(mask_files)} masks")
        
        # Copy files
        image_count = 0
        mask_count = 0
        
        for img_file in image_files:
            # Copy image
            dst_image = self.images_dir / img_file.name
            if not dst_image.exists():
                shutil.copy2(img_file, dst_image)
            image_count += 1
            
            # Find corresponding mask
            base_name = img_file.stem
            mask_patterns = [
                f"{base_name}.jpg.mask.png",
                f"{base_name}.JPG.mask.png"
            ]
            
            for mask_file in mask_files:
                if mask_file.name in mask_patterns:
                    dst_mask = self.masks_dir / f"{base_name}.png"
                    if not dst_mask.exists():
                        shutil.copy2(mask_file, dst_mask)
                    mask_count += 1
                    break
        
        logger.info(f"Organized: {image_count} images, {mask_count} masks")
        return image_count > 0
    
    def step2_run_colmap(self):
        """Run COLMAP reconstruction with GPU"""
        logger.info("\n=== Step 2: Running COLMAP ===")
        
        # Setup paths
        self.colmap_dir.mkdir(parents=True, exist_ok=True)
        database_path = self.colmap_dir / "database.db"
        sparse_dir = self.colmap_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)
        
        # Check if already done
        if (sparse_dir / "0" / "cameras.bin").exists():
            logger.info("COLMAP reconstruction already exists, skipping...")
            return True
        
        # Always use module system for COLMAP
        logger.info("Using COLMAP module system...")
        return self._run_colmap_with_module()
    
    def _run_colmap_with_module(self):
        """Run COLMAP using module system"""
        script_content = f"""#!/bin/bash
module load colmap/3.11

# Feature extraction (CPU-based to avoid GPU issues in container)
colmap feature_extractor \\
    --database_path {self.colmap_dir}/database.db \\
    --image_path {self.images_dir} \\
    --ImageReader.camera_model SIMPLE_PINHOLE \\
    --ImageReader.single_camera 1 \\
    --SiftExtraction.use_gpu 0 \\
    --SiftExtraction.num_threads 8 \\
    --SiftExtraction.max_image_size 3200 \\
    --SiftExtraction.max_num_features 8192

# Feature matching (CPU-based)
colmap exhaustive_matcher \\
    --database_path {self.colmap_dir}/database.db \\
    --SiftMatching.use_gpu 0 \\
    --SiftMatching.num_threads 8

# Reconstruction
colmap mapper \\
    --database_path {self.colmap_dir}/database.db \\
    --image_path {self.images_dir} \\
    --output_path {self.colmap_dir}/sparse \\
    --Mapper.num_threads 16

# Create text directory
mkdir -p {self.colmap_dir}/text

# Convert to text
colmap model_converter \\
    --input_path {self.colmap_dir}/sparse/0 \\
    --output_path {self.colmap_dir}/text \\
    --output_type TXT
"""
        
        script_path = self.output_dir / "run_colmap.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        
        try:
            logger.info("Running COLMAP with CPU (avoiding container GPU issues)...")
            logger.info("This will take longer but is more reliable on HPC systems")
            subprocess.run(["bash", str(script_path)], check=True)
            
            # Verify the output exists
            if (self.colmap_dir / "sparse" / "0" / "cameras.bin").exists():
                logger.info("COLMAP reconstruction successful!")
                return True
            else:
                logger.error("COLMAP output not found")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"COLMAP with module failed: {e}")
            return False
    
    def step3_convert_to_neuralangelo(self):
        """Convert COLMAP output to Neuralangelo format"""
        logger.info("\n=== Step 3: Converting to Neuralangelo Format ===")
        
        # Setup directories
        self.neuralangelo_dir.mkdir(parents=True, exist_ok=True)
        ngp_images = self.neuralangelo_dir / "images"
        ngp_masks = self.neuralangelo_dir / "masks"
        ngp_images.mkdir(exist_ok=True)
        ngp_masks.mkdir(exist_ok=True)
        
        # Find COLMAP text files
        text_dir = self.colmap_dir / "text"
        if not text_dir.exists():
            text_dir = self.colmap_dir / "sparse" / "0"
        
        if not (text_dir / "cameras.txt").exists():
            logger.error("COLMAP text files not found!")
            return False
        
        # Parse COLMAP files
        cameras = self._parse_colmap_cameras(text_dir / "cameras.txt")
        images = self._parse_colmap_images(text_dir / "images.txt")
        
        # Get camera info
        camera_id = list(cameras.keys())[0]
        camera = cameras[camera_id]
        
        # Create transforms.json
        transforms = {
            "camera_model": "OPENCV",
            "frames": []
        }
        
        # Set intrinsics
        if camera['model'] == "SIMPLE_PINHOLE":
            transforms["fl_x"] = camera['f']
            transforms["fl_y"] = camera['f']
        else:
            transforms["fl_x"] = camera['fx']
            transforms["fl_y"] = camera['fy']
        
        transforms["cx"] = camera['cx']
        transforms["cy"] = camera['cy']
        transforms["w"] = camera['width']
        transforms["h"] = camera['height']
        
        # Process each image
        for img_id, img_info in sorted(images.items()):
            # Get transformation matrix
            R = self._qvec_to_rotation_matrix(img_info['qvec'])
            t = np.array(img_info['tvec'])
            
            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = t
            
            # Coordinate system conversion
            coord_change = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])
            c2w = c2w @ coord_change
            
            # Copy files
            src_image = self.images_dir / img_info['name']
            if src_image.exists():
                dst_image = ngp_images / img_info['name']
                if not dst_image.exists():
                    shutil.copy2(src_image, dst_image)
                
                # Copy mask
                base_name = src_image.stem
                src_mask = self.masks_dir / f"{base_name}.png"
                if src_mask.exists():
                    dst_mask = ngp_masks / f"{base_name}.png"
                    if not dst_mask.exists():
                        shutil.copy2(src_mask, dst_mask)
                
                # Add frame
                frame = {
                    "file_path": f"images/{img_info['name']}",
                    "transform_matrix": c2w.tolist(),
                    "mask_path": f"masks/{base_name}.png"
                }
                transforms["frames"].append(frame)
        
        # Save transforms
        with open(self.neuralangelo_dir / "transforms.json", 'w') as f:
            json.dump(transforms, f, indent=2)
        
        # Create config
        self._create_neuralangelo_config()
        
        logger.info(f"Converted {len(transforms['frames'])} frames")
        return True
    
    def step4_train_neuralangelo(self, max_steps=50000):
        """Train Neuralangelo model"""
        logger.info(f"\n=== Step 4: Training Neuralangelo ({max_steps} steps) ===")
        
        self.logs_dir.mkdir(exist_ok=True)
        
        # Check if already trained
        checkpoint_dir = self.logs_dir / "checkpoints"
        if checkpoint_dir.exists() and list(checkpoint_dir.glob("*.pth")):
            logger.info("Found existing checkpoints, skipping training...")
            return True
        
        cmd = [
            "torchrun", "--standalone", "--nproc_per_node=1",
            os.path.expanduser("~/src/neuralangelo/train.py"),
            "--config", str(self.neuralangelo_dir / "config.yaml"),
            "--data.source", str(self.neuralangelo_dir),
            "--data.root", str(self.neuralangelo_dir),
            "--trainer.output_dir", str(self.logs_dir),
            "--trainer.max_steps", str(max_steps),
            "--trainer.val_check_interval", "5000",
            "--trainer.save_interval", "10000"
        ]
        
        try:
            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            logger.info("Training complete!")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def step5_extract_mesh(self, resolution=2048):
        """Extract final mesh"""
        logger.info(f"\n=== Step 5: Extracting Mesh (resolution={resolution}) ===")
        
        # Find checkpoint
        checkpoint_dir = self.logs_dir / "checkpoints"
        if not checkpoint_dir.exists():
            logger.error("No checkpoints found!")
            return False
        
        checkpoints = sorted(checkpoint_dir.glob("*.pth"), key=os.path.getmtime)
        if not checkpoints:
            logger.error("No checkpoint files found!")
            return False
        
        latest_checkpoint = checkpoints[-1]
        mesh_path = self.output_dir / "final_mesh.ply"
        
        cmd = [
            "torchrun", "--standalone", "--nproc_per_node=1",
            os.path.expanduser("~/src/neuralangelo/projects/neuralangelo/scripts/extract_mesh.py"),
            "--config", str(self.logs_dir / "config.yaml"),
            "--checkpoint", str(latest_checkpoint),
            "--output_file", str(mesh_path),
            "--resolution", str(resolution),
            "--block_res", "128"
        ]
        
        try:
            logger.info(f"Using checkpoint: {latest_checkpoint.name}")
            subprocess.run(cmd, check=True)
            
            if mesh_path.exists():
                logger.info(f"✓ Mesh saved to: {mesh_path}")
                file_size = mesh_path.stat().st_size / (1024 * 1024)
                logger.info(f"  File size: {file_size:.1f} MB")
                return True
            else:
                logger.error("Mesh file not created!")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Mesh extraction failed: {e}")
            return False
    
    def _parse_colmap_cameras(self, cameras_file):
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
                    cameras[camera_id] = {
                        'model': model, 'width': width, 'height': height,
                        'f': float(parts[4]), 'cx': float(parts[5]), 'cy': float(parts[6])
                    }
                elif model == "PINHOLE":
                    cameras[camera_id] = {
                        'model': model, 'width': width, 'height': height,
                        'fx': float(parts[4]), 'fy': float(parts[5]),
                        'cx': float(parts[6]), 'cy': float(parts[7])
                    }
        return cameras
    
    def _parse_colmap_images(self, images_file):
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
                images[image_id] = {
                    'name': parts[9],
                    'camera_id': int(parts[8]),
                    'qvec': [float(parts[j]) for j in range(1, 5)],
                    'tvec': [float(parts[j]) for j in range(5, 8)]
                }
                i += 2  # Skip points2D line
            else:
                i += 1
        
        return images
    
    def _qvec_to_rotation_matrix(self, qvec):
        """Convert quaternion to rotation matrix"""
        qw, qx, qy, qz = qvec
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])
        return R
    
    def _create_neuralangelo_config(self):
        """Create Neuralangelo configuration"""
        config = """name: masked_reconstruction
model:
    object_radius: 1.0
    sdf:
        encoding:
            hashgrid:
                dict_size: 22
        mlp:
            hidden_dim: 64
            num_layers: 2
    render:
        rand_rays: 1024
        use_mask: true
        mask_threshold: 0.5

data:
    type: projects.neuralangelo.data
    root: ./
    
training:
    batch_size: 2
    num_workers: 4
    optim:
        lr: 5e-4
    scheduler:
        type: cosine
        warm_up_end: 5000
"""
        
        with open(self.neuralangelo_dir / "config.yaml", 'w') as f:
            f.write(config)
    
    def run_full_pipeline(self, max_steps=50000, mesh_resolution=2048):
        """Run the complete pipeline"""
        start_time = time.time()
        
        try:
            # Step 1: Organize dataset
            if not self.step1_organize_dataset():
                raise RuntimeError("Failed to organize dataset")
            
            # Step 2: Run COLMAP
            if not self.step2_run_colmap():
                raise RuntimeError("COLMAP reconstruction failed")
            
            # Step 3: Convert to Neuralangelo format
            if not self.step3_convert_to_neuralangelo():
                raise RuntimeError("Failed to convert to Neuralangelo format")
            
            # Step 4: Train Neuralangelo
            if not self.step4_train_neuralangelo(max_steps):
                raise RuntimeError("Training failed")
            
            # Step 5: Extract mesh
            if not self.step5_extract_mesh(mesh_resolution):
                raise RuntimeError("Mesh extraction failed")
            
            elapsed_time = time.time() - start_time
            logger.info("\n" + "="*60)
            logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            logger.info(f"Total time: {elapsed_time/3600:.1f} hours")
            logger.info(f"Output directory: {self.output_dir}")
            logger.info(f"Final mesh: {self.output_dir}/final_mesh.ply")
            
        except Exception as e:
            logger.error(f"\n✗ Pipeline failed: {e}")
            elapsed_time = time.time() - start_time
            logger.error(f"Failed after {elapsed_time/60:.1f} minutes")
            raise


def main():
    parser = argparse.ArgumentParser(description="Complete masked 3D reconstruction pipeline")
    parser.add_argument("input_dir", help="Directory with images and masks")
    parser.add_argument("output_dir", help="Output directory for all results")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--max_steps", type=int, default=50000, help="Training steps")
    parser.add_argument("--mesh_resolution", type=int, default=2048, help="Mesh resolution")
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = MaskedReconstructionPipeline(
        args.input_dir,
        args.output_dir,
        args.gpu
    )
    
    pipeline.run_full_pipeline(
        max_steps=args.max_steps,
        mesh_resolution=args.mesh_resolution
    )


if __name__ == "__main__":
    main()
