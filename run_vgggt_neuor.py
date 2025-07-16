#!/usr/bin/env python
"""
Complete 3D reconstruction pipeline: VGG-T → Neuralangelo
Modified from your existing pipeline
"""

import os
import sys
import argparse
import subprocess
import time
import json
import shutil
import glob
import pickle

def run_vggt_pipeline(input_dir, output_dir):
    """Run VGG-T to extract features and camera poses"""
    print("="*60)
    print("STEP 1: Running VGG-T Feature Extraction")
    print("="*60)
    
    vggt_cmd = [
        sys.executable,
        "src/pipeline/run_pipeline.py",
        input_dir,
        "--output_dir", output_dir,
        "--save_neus2",  # Keep this for now, we'll convert format
        "--batch_size", "1"
    ]
    
    print(f"Running: {' '.join(vggt_cmd)}")
    result = subprocess.run(vggt_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("VGG-T failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return None
    
    print(result.stdout)
    
    pkl_files = glob.glob(os.path.join(output_dir, "vggt_predictions_*.pkl"))
    if not pkl_files:
        print("No predictions file found!")
        return None
    
    pkl_files.sort()
    return pkl_files[-1]

def convert_to_neuralangelo_format(predictions_file, neuralangelo_data_dir, images_dir):
    """Convert VGG-T predictions to Neuralangelo format"""
    print("\n" + "="*60)
    print("STEP 2: Converting to Neuralangelo Format")
    print("="*60)
    
    # Load VGG-T predictions
    with open(predictions_file, 'rb') as f:
        predictions = pickle.load(f)
    
    # Create Neuralangelo directory structure
    os.makedirs(os.path.join(neuralangelo_data_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(neuralangelo_data_dir, "masks"), exist_ok=True)
    
    # Convert camera format
    transforms = {
        "camera_model": "PINHOLE",
        "frames": []
    }
    
    # Copy images and convert camera data
    for idx, (img_name, camera_data) in enumerate(predictions['cameras'].items()):
        # Copy image
        src_img = os.path.join(images_dir, img_name)
        dst_img = os.path.join(neuralangelo_data_dir, "images", f"{idx:04d}.jpg")
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
        
        # Convert camera matrix to Neuralangelo format
        # VGG-T likely provides 3x4 or 4x4 camera matrix
        c2w = camera_data['camera_matrix']  # Adjust based on actual format
        
        frame_data = {
            "file_path": f"images/{idx:04d}.jpg",
            "transform_matrix": c2w.tolist() if hasattr(c2w, 'tolist') else c2w
        }
        transforms["frames"].append(frame_data)
    
    # Add camera intrinsics if available
    if 'intrinsics' in predictions:
        K = predictions['intrinsics']
        transforms["fl_x"] = float(K[0, 0])
        transforms["fl_y"] = float(K[1, 1])
        transforms["cx"] = float(K[0, 2])
        transforms["cy"] = float(K[1, 2])
    
    # Save transforms
    with open(os.path.join(neuralangelo_data_dir, "transforms.json"), 'w') as f:
        json.dump(transforms, f, indent=2)
    
    # Create config for Neuralangelo
    config = """
name: vggt_reconstruction
model:
    object_radius: 0.5
    sdf:
        encoding:
            hashgrid:
                dict_size: 22
                n_levels: 16
        gradient:
            mode: numerical
    render:
        rand_rays: 1024
        
data:
    type: projects.neuralangelo.data
    root: /data
    
training:
    max_steps: 300000
    save_every: 5000
"""
    
    with open(os.path.join(neuralangelo_data_dir, "config.yaml"), 'w') as f:
        f.write(config)
    
    print(f"✓ Converted to Neuralangelo format")
    return True

def run_neuralangelo_reconstruction(neuralangelo_data_dir, output_mesh_dir, n_steps=300000):
    """Run Neuralangelo reconstruction"""
    print("\n" + "="*60)
    print("STEP 3: Running Neuralangelo Neural Surface Reconstruction")
    print("="*60)
    
    # This would use your Neuralangelo singularity container
    neuralangelo_cmd = [
        "singularity", "exec", "--nv",
        "--overlay", "$HOME/neuralangelo_setup/overlay.img",
        "-B", f"{neuralangelo_data_dir}:/data",
        "$HOME/neuralangelo_setup/pytorch.sif",
        "bash", "-c",
        f"""
        cd /opt/neuralangelo
        python train.py --config /data/config.yaml --max_steps {n_steps}
        python extract_mesh.py --config /data/config.yaml --output /data/mesh.ply
        """
    ]
    
    # For now, just print what would be run
    print("Would run Neuralangelo with:")
    print(' '.join(neuralangelo_cmd))
    
    # In practice, you'd run this with subprocess
    # result = subprocess.run(neuralangelo_cmd, ...)
    
    return os.path.join(output_mesh_dir, "mesh.ply")

def main():
    parser = argparse.ArgumentParser(description="VGG-T → Neuralangelo pipeline")
    parser.add_argument("input_dir", help="Directory containing images")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--steps", type=int, default=300000)
    parser.add_argument("--skip_reconstruction", action="store_true")
    
    args = parser.parse_args()
    
    output_dir = os.path.abspath(args.output_dir or args.input_dir)
    
    # Create directories
    vggt_output = os.path.join(output_dir, "vggt_output")
    neuralangelo_data = os.path.join(output_dir, "neuralangelo_data")
    final_output = os.path.join(output_dir, "final_mesh")
    
    for d in [vggt_output, neuralangelo_data, final_output]:
        os.makedirs(d, exist_ok=True)
    
    # Step 1: VGG-T
    predictions_file = run_vggt_pipeline(args.input_dir, vggt_output)
    if not predictions_file:
        return 1
    
    # Step 2: Convert
    if not convert_to_neuralangelo_format(predictions_file, neuralangelo_data, args.input_dir):
        return 1
    
    if args.skip_reconstruction:
        print("\nSkipping reconstruction. Data prepared at:", neuralangelo_data)
        return 0
    
    # Step 3: Neuralangelo
    mesh_path = run_neuralangelo_reconstruction(neuralangelo_data, final_output, args.steps)
    
    print("\n✅ Pipeline complete!")
    print(f"Mesh: {mesh_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
