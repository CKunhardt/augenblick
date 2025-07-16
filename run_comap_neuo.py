#!/usr/bin/env python3
"""
End-to-end RealityCapture-style pipeline:
  images → VGGT pkl → COLMAP → transforms.json → Neuralangelo → mesh

Designed for a single NVIDIA L4 (24 GB).
"""

import os
import sys
import subprocess
import glob
import json
import shutil
from pathlib import Path
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VGGTNeuralangelo_Pipeline:
    def __init__(self, data_root, vggt_pkl_path, image_dir=None):
        """
        Initialize pipeline with paths
        
        Args:
            data_root: Root directory for all outputs
            vggt_pkl_path: Path to VGGT predictions pickle file
            image_dir: Directory containing input images (optional, will try to find)
        """
        self.data_root = Path(data_root).expanduser()
        self.vggt_pkl = Path(vggt_pkl_path).expanduser()
        
        # Find image directory if not provided
        if image_dir is None:
            self.img_dir = self._find_image_dir()
        else:
            self.img_dir = Path(image_dir).expanduser()
            
        # Setup working directories
        self.scene_id = self.vggt_pkl.stem  # e.g., vggt_predictions_1752432185
        self.work_dir = self.data_root / "test_run"
        self.ngp_dir = self.work_dir / "neuralangelo_preprocessed" / self.scene_id
        self.log_dir = self.work_dir / "neuralangelo_output_l4" / self.scene_id
        self.mesh_out = self.work_dir / f"{self.scene_id}_mesh.ply"
        
        # Create directories
        self.ngp_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Pipeline initialized:")
        logger.info(f"  Scene ID: {self.scene_id}")
        logger.info(f"  Images: {self.img_dir}")
        logger.info(f"  Output: {self.work_dir}")
    
    def _find_image_dir(self):
        """Try to find image directory based on VGGT output location"""
        possible_dirs = [
            self.data_root / "images_from_dropbox" / "noscale",
            self.data_root / "images_from_dropbox",
            self.vggt_pkl.parent.parent / "images",
        ]
        
        for d in possible_dirs:
            if d.exists() and any(d.glob("*.jpg")) or any(d.glob("*.JPG")):
                logger.info(f"Found image directory: {d}")
                return d
                
        raise ValueError("Could not find image directory. Please specify with --image_dir")
    
    def run_command(self, cmd, description=None):
        """Run a shell command and handle errors"""
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
    
    def step1_vggt_to_colmap(self):
        """Convert VGGT pickle to COLMAP format"""
        colmap_dir = self.ngp_dir / "colmap"
        
        # Check if already done
        if (colmap_dir / "cameras.bin").exists():
            logger.info("COLMAP files already exist, skipping conversion")
            return
        
        cmd = [
            sys.executable,
            os.path.expanduser("~/src/vggt/tools/pkl_to_colmap.py"),
            "--predictions", str(self.vggt_pkl),
            "--out_dir", str(colmap_dir),
            "--image_dir", str(self.img_dir)
        ]
        
        self.run_command(cmd, "Converting VGGT pickle to COLMAP format")
    
    def step2_colmap_to_ngp(self):
        """Convert COLMAP to transforms.json (Instant-NGP format)"""
        transforms_path = self.ngp_dir / "transforms.json"
        
        if transforms_path.exists():
            logger.info("transforms.json already exists, skipping conversion")
            return
            
        cmd = [
            sys.executable,
            os.path.expanduser("~/src/vggt/tools/colmap_to_ngp.py"),
            "--colmap", str(self.ngp_dir / "colmap"),
            "--images", str(self.img_dir),
            "--out_dir", str(self.ngp_dir)
        ]
        
        self.run_command(cmd, "Writing transforms.json & processing frames")
    
    def step3_train_neuralangelo(self, max_iter=50000, base_res=256, level=1):
        """Train Neuralangelo model"""
        # Check if training already completed
        checkpoints = list((self.log_dir / "checkpoints").glob("*.pth"))
        if checkpoints:
            logger.info(f"Found {len(checkpoints)} existing checkpoints, skipping training")
            return
            
        cmd = [
            "torchrun", "--standalone", "--nproc_per_node=1",
            os.path.expanduser("~/src/neuralangelo/train.py"),
            "--config", os.path.expanduser("~/src/neuralangelo/projects/neuralangelo/configs/custom/object.yaml"),
            "--data", str(self.ngp_dir),
            "--workspace", str(self.log_dir),
            "--base_res", str(base_res),
            "--level", str(level),
            "--max_iter", str(max_iter)
        ]
        
        self.run_command(cmd, f"Training Neuralangelo ({max_iter} iterations)")
    
    def step4_extract_mesh(self, resolution=2048, block_res=128, textured=True):
        """Extract final mesh from trained model"""
        # Find latest checkpoint
        checkpoint_dir = self.log_dir / "checkpoints"
        checkpoints = sorted(checkpoint_dir.glob("*.pth"), key=os.path.getmtime)
        
        if not checkpoints:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")
            
        latest_ckpt = checkpoints[-1]
        logger.info(f"Using checkpoint: {latest_ckpt.name}")
        
        cmd = [
            "torchrun", "--standalone", "--nproc_per_node=1",
            os.path.expanduser("~/src/neuralangelo/extract_mesh.py"),
            "--config", str(self.log_dir / "config.yaml"),
            "--checkpoint", str(latest_ckpt),
            "--output_file", str(self.mesh_out),
            "--resolution", str(resolution),
            "--block_res", str(block_res)
        ]
        
        if textured:
            cmd.append("--textured")
            
        self.run_command(cmd, "Extracting mesh")
        
        if self.mesh_out.exists():
            logger.info(f"✓ Mesh saved to: {self.mesh_out}")
            # Print mesh statistics
            self._print_mesh_stats()
        else:
            logger.error("Mesh extraction failed - output file not found")
    
    def _print_mesh_stats(self):
        """Print basic mesh statistics if trimesh is available"""
        try:
            import trimesh
            mesh = trimesh.load(self.mesh_out)
            logger.info(f"Mesh statistics:")
            logger.info(f"  Vertices: {len(mesh.vertices):,}")
            logger.info(f"  Faces: {len(mesh.faces):,}")
            logger.info(f"  Watertight: {mesh.is_watertight}")
            logger.info(f"  File size: {self.mesh_out.stat().st_size / 1024 / 1024:.1f} MB")
        except ImportError:
            pass
    
    def run_full_pipeline(self, **kwargs):
        """Run the complete pipeline end-to-end"""
        logger.info("Starting VGGT → Neuralangelo pipeline")
        
        try:
            # Step 1: VGGT to COLMAP
            self.step1_vggt_to_colmap()
            
            # Step 2: COLMAP to transforms.json
            self.step2_colmap_to_ngp()
            
            # Step 3: Train Neuralangelo
            train_kwargs = {k: v for k, v in kwargs.items() 
                           if k in ['max_iter', 'base_res', 'level']}
            self.step3_train_neuralangelo(**train_kwargs)
            
            # Step 4: Extract mesh
            mesh_kwargs = {k: v for k, v in kwargs.items() 
                          if k in ['resolution', 'block_res', 'textured']}
            self.step4_extract_mesh(**mesh_kwargs)
            
            logger.info("✓ Pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="VGGT to Neuralangelo pipeline")
    parser.add_argument("--vggt_pkl", required=True, help="Path to VGGT predictions pickle")
    parser.add_argument("--data_root", default="~/scratch", help="Root directory for outputs")
    parser.add_argument("--image_dir", help="Directory containing input images")
    
    # Training parameters
    parser.add_argument("--max_iter", type=int, default=50000, help="Training iterations")
    parser.add_argument("--base_res", type=int, default=256, help="Base resolution")
    parser.add_argument("--level", type=int, default=1, help="Coarse-to-fine level")
    
    # Mesh extraction parameters
    parser.add_argument("--resolution", type=int, default=2048, help="Mesh extraction resolution")
    parser.add_argument("--block_res", type=int, default=128, help="Block resolution for extraction")
    parser.add_argument("--textured", action="store_true", help="Extract textured mesh")
    
    # Pipeline control
    parser.add_argument("--skip_training", action="store_true", help="Skip training if checkpoint exists")
    parser.add_argument("--mesh_only", action="store_true", help="Only extract mesh from existing checkpoint")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = VGGTNeuralangelo_Pipeline(
        data_root=args.data_root,
        vggt_pkl_path=args.vggt_pkl,
        image_dir=args.image_dir
    )
    
    # Run pipeline
    if args.mesh_only:
        pipeline.step4_extract_mesh(
            resolution=args.resolution,
            block_res=args.block_res,
            textured=args.textured
        )
    else:
        pipeline.run_full_pipeline(
            max_iter=args.max_iter,
            base_res=args.base_res,
            level=args.level,
            resolution=args.resolution,
            block_res=args.block_res,
            textured=args.textured
        )


if __name__ == "__main__":
    main()

