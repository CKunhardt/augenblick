#!/usr/bin/env python3
"""
Integrated VGGT → Neuralangelo Pipeline
Customized for your Georgia Tech HPC setup with Augenblick and Neuralangelo
"""

import os
import sys
import torch
import numpy as np
import json
import time
import shutil
from pathlib import Path
import cv2
import argparse
from typing import Dict, List, Tuple, Optional
import subprocess

# Setup paths based on your directory structure
HOME_DIR = Path.home()
AUGENBLICK_DIR = HOME_DIR / "augenblick"
NEURALANGELO_DIR = HOME_DIR / "neuralangelo_setup" / "neuralangelo"
SCRATCH_DIR = HOME_DIR / "scratch"

# Add Augenblick paths
sys.path.insert(0, str(AUGENBLICK_DIR / "src"))
sys.path.insert(0, str(AUGENBLICK_DIR / "src" / "vggt"))

print(f"Home directory: {HOME_DIR}")
print(f"Augenblick: {AUGENBLICK_DIR}")
print(f"Neuralangelo: {NEURALANGELO_DIR}")
print(f"Scratch: {SCRATCH_DIR}")

class IntegratedVGGTPipeline:
    def __init__(self, vggt_checkpoint: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Find VGGT checkpoint
        if vggt_checkpoint is None:
            vggt_checkpoint = self.find_vggt_checkpoint()
        
        # Load VGGT model
        self.load_vggt(vggt_checkpoint)
        
    def find_vggt_checkpoint(self) -> str:
        """Search for VGGT checkpoint in common locations"""
        possible_paths = [
            AUGENBLICK_DIR / "checkpoints" / "vggt_large.pth",
            AUGENBLICK_DIR / "checkpoints" / "vggt.pth",
            AUGENBLICK_DIR / "models" / "vggt_large.pth",
            AUGENBLICK_DIR / "src" / "vggt" / "vggt_large.pth",
            AUGENBLICK_DIR / "src" / "vggt" / "checkpoints" / "vggt_large.pth",
            HOME_DIR / "vggt_large.pth",
        ]
        
        for path in possible_paths:
            if path.exists():
                print(f"Found VGGT checkpoint: {path}")
                return str(path)
        
        print("Warning: No VGGT checkpoint found. Using placeholder.")
        print("Please download the checkpoint or specify with --vggt_checkpoint")
        return "vggt_large.pth"
    
    def load_vggt(self, checkpoint_path: str):
        """Load VGGT model from Augenblick"""
        print("\nLoading VGGT model...")
        
        try:
            # Try different import strategies based on Augenblick structure
            try:
                from vggt.models import VGGT
                print("Imported VGGT from vggt.models")
            except ImportError:
                try:
                    from models import VGGT
                    print("Imported VGGT from models")
                except ImportError:
                    # Check if there's a model file directly
                    vggt_model_file = AUGENBLICK_DIR / "src" / "vggt" / "model.py"
                    if vggt_model_file.exists():
                        import importlib.util
                        spec = importlib.util.spec_from_file_location("vggt_model", vggt_model_file)
                        vggt_model = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(vggt_model)
                        VGGT = vggt_model.VGGT
                        print("Imported VGGT from model.py")
                    else:
                        raise ImportError("Could not find VGGT model implementation")
            
            # Initialize model
            self.model = VGGT().to(self.device)
            
            # Load checkpoint if exists
            if Path(checkpoint_path).exists():
                print(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                else:
                    self.model.load_state_dict(checkpoint)
                
                print("✓ Checkpoint loaded successfully")
            else:
                print(f"Warning: Checkpoint not found at {checkpoint_path}")
            
            self.model.eval()
            
        except Exception as e:
            print(f"Error loading VGGT: {e}")
            print("\nTroubleshooting:")
            print("1. Check if VGGT files exist in:", AUGENBLICK_DIR / "src" / "vggt")
            print("2. Ensure all dependencies are installed")
            print("3. Check the model architecture matches the checkpoint")
            raise
    
    def prepare_images(self, input_dir: Path, output_dir: Path) -> Tuple[List[torch.Tensor], List[str], Path]:
        """Prepare images for VGGT processing"""
        print(f"\nPreparing images from {input_dir}")
        
        # Create clean image directory
        clean_dir = output_dir / "images_clean"
        clean_dir.mkdir(exist_ok=True, parents=True)
        
        # Find all images (handle your naming convention)
        image_files = []
        for pattern in ['*.JPG', '*.jpg', '*.png', '*.PNG']:
            image_files.extend(input_dir.glob(pattern))
        
        # Filter out mask files
        image_files = [f for f in image_files if '.mask.' not in f.name]
        image_files = sorted(image_files)
        
        print(f"Found {len(image_files)} images")
        
        images = []
        names = []
        
        for idx, img_path in enumerate(image_files):
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not load {img_path}")
                continue
            
            # Clean filename for Neuralangelo
            clean_name = f"image_{idx:04d}.jpg"
            clean_path = clean_dir / clean_name
            
            # Save clean copy
            cv2.imwrite(str(clean_path), img)
            
            # Convert for VGGT
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize for VGGT (max 518px)
            h, w = img_rgb.shape[:2]
            if max(h, w) > 518:
                scale = 518 / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Convert to tensor
            img_tensor = torch.from_numpy(img_rgb).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)
            
            # Normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std
            
            images.append(img_tensor)
            names.append(clean_name)
        
        if not images:
            raise ValueError("No valid images found")
        
        # Stack into batch
        images_batch = torch.stack(images).to(self.device)
        
        return images_batch, names, clean_dir
    
    def run_vggt_inference(self, images: torch.Tensor) -> Dict:
        """Run VGGT inference"""
        print("\nRunning VGGT inference...")
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model(images)
        
        inference_time = time.time() - start_time
        print(f"✓ VGGT inference completed in {inference_time:.3f} seconds!")
        
        # Convert outputs to numpy
        result = {}
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                if torch.is_tensor(value):
                    result[key] = value.cpu().numpy()
                else:
                    result[key] = value
        else:
            # Handle different output formats
            result = {'output': outputs.cpu().numpy() if torch.is_tensor(outputs) else outputs}
        
        return result
    
    def create_neuralangelo_project(self, vggt_outputs: Dict, image_names: List[str], 
                                  clean_dir: Path, output_dir: Path):
        """Create Neuralangelo project structure with VGGT initialization"""
        print("\nCreating Neuralangelo project...")
        
        # Create directory structure
        na_dir = output_dir / "neuralangelo_project"
        na_dir.mkdir(exist_ok=True, parents=True)
        
        (na_dir / "images").mkdir(exist_ok=True)
        (na_dir / "depth_vggt").mkdir(exist_ok=True)
        
        # Copy images
        for name in image_names:
            src = clean_dir / name
            dst = na_dir / "images" / name
            shutil.copy2(src, dst)
        
        # Save VGGT outputs
        if 'depth_maps' in vggt_outputs:
            for i, depth in enumerate(vggt_outputs['depth_maps']):
                np.save(na_dir / "depth_vggt" / f"{i:04d}.npy", depth)
        
        # Create transforms.json
        transforms = self.create_transforms_json(vggt_outputs, image_names, na_dir)
        
        # Create Neuralangelo config
        self.create_neuralangelo_config(na_dir)
        
        # Create run script
        self.create_run_script(na_dir)
        
        return na_dir
    
    def create_transforms_json(self, vggt_outputs: Dict, image_names: List[str], 
                             output_dir: Path) -> Dict:
        """Create transforms.json for Neuralangelo"""
        transforms = {
            "camera_model": "PINHOLE",
            "frames": []
        }
        
        # Process each frame
        for i, name in enumerate(image_names):
            frame = {
                "file_path": f"images/{name}",
                "transform_matrix": np.eye(4).tolist()  # Identity matrix as placeholder
            }
            
            # Add VGGT outputs if available
            if 'cameras' in vggt_outputs and i < len(vggt_outputs['cameras']):
                # Convert VGGT camera to matrix
                camera = vggt_outputs['cameras'][i]
                if len(camera) >= 7:  # Has position and orientation
                    matrix = self.camera_to_matrix(camera)
                    frame["transform_matrix"] = matrix.tolist()
            
            if 'depth_maps' in vggt_outputs:
                frame["vggt_depth_path"] = f"depth_vggt/{i:04d}.npy"
            
            transforms["frames"].append(frame)
        
        # Add camera intrinsics (you may need to adjust these)
        transforms["fl_x"] = 800.0
        transforms["fl_y"] = 800.0
        transforms["cx"] = 400.0
        transforms["cy"] = 300.0
        transforms["w"] = 800
        transforms["h"] = 600
        
        # Save transforms
        with open(output_dir / "transforms.json", 'w') as f:
            json.dump(transforms, f, indent=2)
        
        return transforms
    
    def camera_to_matrix(self, camera: np.ndarray) -> np.ndarray:
        """Convert camera parameters to 4x4 matrix"""
        if len(camera) >= 7:
            # Assume: position (3) + quaternion (4)
            pos = camera[:3]
            quat = camera[3:7]
            
            # Quaternion to rotation matrix
            x, y, z, w = quat
            R = np.array([
                [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
                [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
                [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
            ])
            
            # Build matrix
            matrix = np.eye(4)
            matrix[:3, :3] = R
            matrix[:3, 3] = pos
            
            return matrix
        else:
            return np.eye(4)
    
    def create_neuralangelo_config(self, output_dir: Path):
        """Create Neuralangelo configuration file"""
        config = """
name: vggt_initialized
type: projects.neuralangelo.trainer

model:
    type: projects.neuralangelo.model
    object_radius: 1.0
    
    sdf:
        type: models.neuralangelo.sdf
        n_layers: 2
        n_layers_bg: 2
        dim_hidden: 256
        
    render:
        type: models.neuralangelo.render
        num_samples: 128
        num_samples_fine: 0
        
data:
    type: projects.neuralangelo.data
    root_dir: .
    
train:
    batch_size: 1
    num_workers: 0
    
    optim:
        type: Adam
        lr: 5e-4
        
    max_iter: 100000
    
    loss_weight:
        rgb: 1.0
        eikonal: 0.1
        
    print_interval: 100
    save_interval: 10000
"""
        
        with open(output_dir / "config.yaml", 'w') as f:
            f.write(config)
    
    def create_run_script(self, output_dir: Path):
        """Create script to run Neuralangelo"""
        script = f"""#!/bin/bash
#SBATCH --job-name=neuralangelo_vggt
#SBATCH --output=neuralangelo_%j.out
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Load modules (adjust as needed for your HPC)
module load cuda/11.7
module load singularity

# Paths
NEURALANGELO_SIF="{HOME_DIR}/neuralangelo_setup/neuralangelo.sif"
PROJECT_DIR="{output_dir}"

# Run Neuralangelo with Singularity
cd $PROJECT_DIR

# First run COLMAP if needed
echo "Running COLMAP..."
singularity exec --nv \\
    {HOME_DIR}/neuralangelo_setup/colmap.sif \\
    colmap automatic_reconstructor \\
    --workspace_path . \\
    --image_path images

# Then run Neuralangelo
echo "Running Neuralangelo..."
singularity exec --nv \\
    $NEURALANGELO_SIF \\
    python {NEURALANGELO_DIR}/projects/neuralangelo/train.py \\
    --config config.yaml \\
    --logdir .

echo "Done!"
"""
        
        script_path = output_dir / "run_neuralangelo.sh"
        with open(script_path, 'w') as f:
            f.write(script)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        print(f"Created run script: {script_path}")
    
    def process(self, input_dir: str, output_name: str = None):
        """Main processing pipeline"""
        input_dir = Path(input_dir)
        
        # Create output directory
        if output_name is None:
            output_name = f"vggt_na_{input_dir.name}_{time.strftime('%Y%m%d_%H%M%S')}"
        
        output_dir = SCRATCH_DIR / output_name
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print("\n" + "="*60)
        print("Integrated VGGT → Neuralangelo Pipeline")
        print("="*60)
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        
        # Step 1: Prepare images
        images, names, clean_dir = self.prepare_images(input_dir, output_dir)
        
        # Step 2: Run VGGT
        vggt_outputs = self.run_vggt_inference(images)
        
        # Step 3: Create Neuralangelo project
        na_dir = self.create_neuralangelo_project(vggt_outputs, names, clean_dir, output_dir)
        
        # Step 4: Save VGGT outputs for inspection
        vggt_output_file = output_dir / "vggt_outputs.npz"
        np.savez(vggt_output_file, **vggt_outputs)
        print(f"\nVGGT outputs saved to: {vggt_output_file}")
        
        print("\n" + "="*60)
        print("✓ Pipeline complete!")
        print("="*60)
        print(f"\nNeuralangelo project created at: {na_dir}")
        print("\nTo run Neuralangelo:")
        print(f"1. Submit job: sbatch {na_dir}/run_neuralangelo.sh")
        print(f"2. Or run interactively: cd {na_dir} && bash run_neuralangelo.sh")
        
        return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Integrated VGGT → Neuralangelo pipeline for Georgia Tech HPC"
    )
    parser.add_argument("input_dir", help="Directory containing input images")
    parser.add_argument("--output_name", help="Name for output directory (default: auto-generated)")
    parser.add_argument("--vggt_checkpoint", help="Path to VGGT checkpoint")
    
    # Quick test options
    parser.add_argument("--test", action="store_true", 
                       help="Run on test images (3 images from neuralangelo_data)")
    
    args = parser.parse_args()
    
    # Handle test mode
    if args.test:
        test_dir = SCRATCH_DIR / "images_from_dropbox" / "neuralangelo_data" / "images"
        if test_dir.exists():
            args.input_dir = str(test_dir)
            args.output_name = "test_vggt_na"
            print(f"Test mode: Using images from {test_dir}")
        else:
            print(f"Test directory not found: {test_dir}")
            return
    
    # Initialize pipeline
    pipeline = IntegratedVGGTPipeline(args.vggt_checkpoint)
    
    # Run processing
    pipeline.process(args.input_dir, args.output_name)


if __name__ == "__main__":
    main()
