#!/usr/bin/env python
"""
Complete 3D reconstruction pipeline: VGG-T → NeuS2
Runs both feature extraction and neural surface reconstruction
"""

import os
import sys
import argparse
import subprocess
import time
import json
import shutil

# Add necessary paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../NeuS2'))

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
        "--save_for_neus2",
        "--skip_glb"  # We'll generate mesh with NeuS2 instead
    ]
    
    print(f"Running: {' '.join(vggt_cmd)}")
    result = subprocess.run(vggt_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("VGG-T failed!")
        print(result.stderr)
        return None
    
    print(result.stdout)
    
    # Find the predictions pickle file
    import glob
    pkl_files = glob.glob(os.path.join(output_dir, "vggt_predictions_*.pkl"))
    if not pkl_files:
        print("No predictions file found!")
        return None
    
    return pkl_files[0]

def convert_to_neus2_format(predictions_file, neus2_data_dir, images_dir):
    """Convert VGG-T predictions to NeuS2 format"""
    print("\n" + "="*60)
    print("STEP 2: Converting to NeuS2 Format")
    print("="*60)
    
    converter_cmd = [
        sys.executable,
        "src/pipeline/vggt_to_neus2_converter.py",
        predictions_file,
        neus2_data_dir,
        "--images_dir", images_dir
    ]
    
    print(f"Running: {' '.join(converter_cmd)}")
    result = subprocess.run(converter_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Conversion failed!")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True

def run_neus2_reconstruction(neus2_data_dir, output_mesh_dir):
    """Run NeuS2 reconstruction"""
    print("\n" + "="*60)
    print("STEP 3: Running NeuS2 Neural Surface Reconstruction")
    print("="*60)
    
    # Change to NeuS2 directory
    original_dir = os.getcwd()
    neus2_dir = os.path.join(original_dir, "src/NeuS2")
    os.chdir(neus2_dir)
    
    try:
        # Import NeuS2
        import pyngp as ngp
        import numpy as np
        
        # Create config for custom data
        config = {
            "dataset": "nerf",
            "data_dir": os.path.abspath(neus2_data_dir),
            "mode": "train",
            "near": 0.1,
            "far": 10.0,
            "n_steps": 20000,  # Adjust based on quality needs
            "save_mesh": True,
            "mesh_resolution": 512,
            "mesh_threshold": 0.0,
        }
        
        # Initialize testbed
        testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
        testbed.load_training_data(config["data_dir"])
        
        # Train
        print("Training neural surface...")
        for i in range(config["n_steps"]):
            testbed.frame()
            if i % 1000 == 0:
                print(f"Step {i}/{config['n_steps']}")
        
        # Export mesh
        mesh_path = os.path.join(output_mesh_dir, "reconstruction.ply")
        print(f"Exporting mesh to {mesh_path}")
        testbed.save_mesh(mesh_path, resolution=config["mesh_resolution"])
        
        print("✓ NeuS2 reconstruction complete!")
        return mesh_path
        
    except Exception as e:
        print(f"NeuS2 failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        os.chdir(original_dir)

def main():
    parser = argparse.ArgumentParser(description="Complete 3D reconstruction pipeline")
    parser.add_argument("input_dir", help="Directory containing images")
    parser.add_argument("--output_dir", default="./reconstruction_output", 
                       help="Output directory for all results")
    parser.add_argument("--neus2_steps", type=int, default=20000,
                       help="Number of NeuS2 training steps")
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = os.path.abspath(args.output_dir)
    vggt_output = os.path.join(output_dir, "vggt_output")
    neus2_data = os.path.join(output_dir, "neus2_data")
    final_output = os.path.join(output_dir, "final_mesh")
    
    os.makedirs(vggt_output, exist_ok=True)
    os.makedirs(neus2_data, exist_ok=True)
    os.makedirs(final_output, exist_ok=True)
    
    # Check if input has images subdirectory
    if os.path.exists(os.path.join(args.input_dir, "images")):
        input_dir = args.input_dir
        images_dir = os.path.join(args.input_dir, "images")
    else:
        # Assume input_dir IS the images directory
        # Create temporary structure for VGG-T
        temp_dir = os.path.join(output_dir, "temp_input")
        os.makedirs(temp_dir, exist_ok=True)
        images_dir = os.path.join(temp_dir, "images")
        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)
        shutil.copytree(args.input_dir, images_dir)
        input_dir = temp_dir
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Run pipeline
    start_time = time.time()
    
    # Step 1: VGG-T
    predictions_file = run_vggt_pipeline(input_dir, vggt_output)
    if not predictions_file:
        print("Pipeline failed at VGG-T stage")
        return 1
    
    # Step 2: Convert to NeuS2 format
    if not convert_to_neus2_format(predictions_file, neus2_data, images_dir):
        print("Pipeline failed at conversion stage")
        return 1
    
    # Step 3: NeuS2 reconstruction
    mesh_path = run_neus2_reconstruction(neus2_data, final_output)
    if not mesh_path:
        print("Pipeline failed at NeuS2 stage")
        return 1
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE!")
    print("="*60)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Final mesh: {mesh_path}")
    print(f"All outputs in: {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
