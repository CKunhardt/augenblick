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
import glob

# Add necessary paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../NeuS2'))

def run_vggt_pipeline(input_dir, output_dir):
    """Run VGG-T to extract features and camera poses using the simple pipeline"""
    print("="*60)
    print("STEP 1: Running VGG-T Feature Extraction")
    print("="*60)
    
    vggt_cmd = [
        sys.executable,
        "src/pipeline/run_pipeline.py",  # Use the new simple pipeline
        input_dir,
        "--output_dir", output_dir,
        "--save_neus2",
        "--batch_size", "1"  # Process one at a time for stability
    ]
    
    print(f"Running: {' '.join(vggt_cmd)}")
    result = subprocess.run(vggt_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("VGG-T failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return None
    
    print(result.stdout)
    
    # Find the predictions pickle file
    pkl_files = glob.glob(os.path.join(output_dir, "vggt_predictions_*.pkl"))
    if not pkl_files:
        print("No predictions file found!")
        return None
    
    # Return the most recent one
    pkl_files.sort()
    return pkl_files[-1]

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
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    
    print(result.stdout)
    return True

def run_neus2_reconstruction(neus2_data_dir, output_mesh_dir, n_steps=20000):
    """Run NeuS2 reconstruction"""
    print("\n" + "="*60)
    print("STEP 3: Running NeuS2 Neural Surface Reconstruction")
    print("="*60)
    
    # Change to NeuS2 directory
    original_dir = os.getcwd()
    neus2_dir = os.path.join(original_dir, "src/NeuS2")
    
    # Check if NeuS2 directory exists
    if not os.path.exists(neus2_dir):
        print(f"NeuS2 directory not found at: {neus2_dir}")
        print("Please ensure NeuS2 is installed in src/NeuS2")
        return None
    
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
            "n_steps": n_steps,
            "save_mesh": True,
            "mesh_resolution": 512,
            "mesh_threshold": 0.0,
        }
        
        # Initialize testbed
        print("Initializing NeuS2...")
        testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
        testbed.load_training_data(config["data_dir"])
        
        # Train
        print(f"Training neural surface ({n_steps} steps)...")
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
        
    except ImportError as e:
        print(f"Failed to import NeuS2: {e}")
        print("Make sure NeuS2 is properly installed with:")
        print("  cd src/NeuS2 && pip install -e .")
        return None
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
    parser.add_argument("--output_dir", default=None, 
                       help="Output directory for all results (default: input_dir)")
    parser.add_argument("--neus2_steps", type=int, default=20000,
                       help="Number of NeuS2 training steps")
    parser.add_argument("--skip_neus2", action="store_true",
                       help="Skip NeuS2 reconstruction (only run VGG-T)")
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.abspath(args.input_dir)
    
    # Create output directories
    vggt_output = os.path.join(output_dir, "vggt_output")
    neus2_data = os.path.join(output_dir, "neus2_data")
    final_output = os.path.join(output_dir, "final_mesh")
    
    os.makedirs(vggt_output, exist_ok=True)
    os.makedirs(neus2_data, exist_ok=True)
    os.makedirs(final_output, exist_ok=True)
    
    # Check if input has images subdirectory
    input_dir = args.input_dir
    if os.path.exists(os.path.join(input_dir, "images")):
        images_dir = os.path.join(input_dir, "images")
    else:
        # Assume input_dir IS the images directory
        images_dir = input_dir
        # For compatibility with some tools that expect images/ subdirectory
        if not os.path.basename(input_dir) == "images":
            # Create a temporary structure
            temp_dir = os.path.join(output_dir, "temp_input")
            temp_images_dir = os.path.join(temp_dir, "images")
            if os.path.exists(temp_images_dir):
                shutil.rmtree(temp_images_dir)
            os.makedirs(temp_dir, exist_ok=True)
            # Create symlink instead of copying to save space
            os.symlink(os.path.abspath(input_dir), temp_images_dir)
            input_dir = temp_dir
            images_dir = temp_images_dir
    
    print(f"Input directory: {input_dir}")
    print(f"Images directory: {images_dir}")
    print(f"Output directory: {output_dir}")
    
    # Count images
    image_count = len(glob.glob(os.path.join(images_dir, "*.[jJ][pP][gG]")) + 
                     glob.glob(os.path.join(images_dir, "*.[jJ][pP][eE][gG]")) +
                     glob.glob(os.path.join(images_dir, "*.[pP][nN][gG]")))
    print(f"Found {image_count} images")
    
    if image_count == 0:
        print("ERROR: No images found!")
        return 1
    
    # Run pipeline
    start_time = time.time()
    
    # Step 1: VGG-T
    predictions_file = run_vggt_pipeline(input_dir, vggt_output)
    if not predictions_file:
        print("Pipeline failed at VGG-T stage")
        return 1
    
    vggt_time = time.time() - start_time
    print(f"\nVGG-T completed in {vggt_time:.2f} seconds")
    
    if args.skip_neus2:
        print("\nSkipping NeuS2 reconstruction as requested")
        print(f"VGG-T predictions saved to: {predictions_file}")
        return 0
    
    # Step 2: Convert to NeuS2 format
    convert_start = time.time()
    if not convert_to_neus2_format(predictions_file, neus2_data, images_dir):
        print("Pipeline failed at conversion stage")
        return 1
    
    convert_time = time.time() - convert_start
    print(f"\nConversion completed in {convert_time:.2f} seconds")
    
    # Step 3: NeuS2 reconstruction
    neus2_start = time.time()
    mesh_path = run_neus2_reconstruction(neus2_data, final_output, args.neus2_steps)
    if not mesh_path:
        print("Pipeline failed at NeuS2 stage")
        print("Note: You can still use the VGG-T predictions at:", predictions_file)
        return 1
    
    neus2_time = time.time() - neus2_start
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE!")
    print("="*60)
    print(f"Timing breakdown:")
    print(f"  VGG-T extraction: {vggt_time:.2f} seconds")
    print(f"  Format conversion: {convert_time:.2f} seconds")
    print(f"  NeuS2 reconstruction: {neus2_time:.2f} seconds")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"\nOutputs:")
    print(f"  VGG-T predictions: {predictions_file}")
    print(f"  NeuS2 data: {neus2_data}")
    print(f"  Final mesh: {mesh_path}")
    print(f"  All outputs in: {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
