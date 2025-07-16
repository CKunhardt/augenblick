#!/usr/bin/env python
"""
Complete 3D reconstruction pipeline: VGG-T → Neuralangelo
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
import pickle
import numpy as np

# Add necessary paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../neuralangelo'))

def run_vggt_pipeline(input_dir, output_dir):
    """Run VGG-T to extract features and camera poses using the simple pipeline"""
    print("="*60)
    print("STEP 1: Running VGG-T Feature Extraction")
    print("="*60)
    
    # First, count the images to show progress
    images_dir = os.path.join(input_dir, "images") if os.path.exists(os.path.join(input_dir, "images")) else input_dir
    image_files = glob.glob(os.path.join(images_dir, "*.[jJ][pP][gG]")) + \
                  glob.glob(os.path.join(images_dir, "*.[jJ][pP][eE][gG]")) + \
                  glob.glob(os.path.join(images_dir, "*.[pP][nN][gG]"))
    total_images = len(image_files)
    print(f"Processing {total_images} images...")
    
    vggt_cmd = [
        sys.executable,
        "src/pipeline/run_pipeline.py",  # Use the new simple pipeline
        input_dir,
        "--output_dir", output_dir,
        "--save_neus2",  # Keep this flag as it saves camera params
        "--batch_size", "1",  # Process one at a time for stability
        "--verbose"  # Add verbose flag if supported
    ]
    
    print(f"Running: {' '.join(vggt_cmd)}")
    
    # Run with real-time output
    process = subprocess.Popen(vggt_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    
    # Track progress
    processed = 0
    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        if line:
            # Look for processing indicators
            if "Processing" in line or "Image" in line or "Frame" in line:
                processed += 1
                print(f"[{processed}/{total_images}] {line}")
            elif "Error" in line or "Warning" in line:
                print(f"⚠️  {line}")
            else:
                print(f"    {line}")
    
    process.wait()
    
    if process.returncode != 0:
        print("VGG-T failed!")
        stderr = process.stderr.read()
        if stderr:
            print("STDERR:", stderr)
        return None
    
    print(f"\n✓ Completed processing {total_images} images")
    
    # Find the predictions pickle file
    pkl_files = glob.glob(os.path.join(output_dir, "vggt_predictions_*.pkl"))
    if not pkl_files:
        print("No predictions file found!")
        return None
    
    # Return the most recent one
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
    
    # Create Neuralangelo data structure
    os.makedirs(neuralangelo_data_dir, exist_ok=True)
    
    # Create images directories
    images_output_dir = os.path.join(neuralangelo_data_dir, "images")
    if os.path.exists(images_output_dir):
        if os.path.islink(images_output_dir):
            os.unlink(images_output_dir)
        else:
            shutil.rmtree(images_output_dir)
    shutil.copytree(images_dir, images_output_dir)
    
    # Get data from predictions
    extrinsics = predictions['extrinsic']  # (N, 3, 4) world-to-camera
    intrinsics = predictions['intrinsic']  # (N, 3, 3)
    num_frames = extrinsics.shape[0]
    
    # Get image dimensions from predictions
    if 'images' in predictions:
        h, w = predictions['images'].shape[1:3]
    else:
        # Default dimensions
        h, w = 350, 518
    
    # Get image files (exclude mask files)
    image_files = []
    for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]:
        files = glob.glob(os.path.join(images_dir, ext))
        # Filter out mask files
        files = [f for f in files if not f.endswith('.mask.png')]
        image_files.extend(files)
    image_files = sorted(image_files)
    
    frames = []
    for i in range(min(num_frames, len(image_files))):
        # Convert 3x4 world-to-camera to 4x4
        w2c = np.eye(4)
        w2c[:3, :] = extrinsics[i]
        
        # Invert to get camera-to-world (what Neuralangelo expects)
        c2w = np.linalg.inv(w2c)
        
        # Get intrinsics for this frame
        K = intrinsics[i]
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        
        # Get image filename
        img_name = os.path.basename(image_files[i])
        
        # Include intrinsics in each frame
        frame = {
            "file_path": f"./images/{img_name}",
            "transform_matrix": c2w.tolist(),
            "fl_x": fx,
            "fl_y": fy,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h
        }
        frames.append(frame)
    
    # Compute scale and offset from world points (optional but helpful)
    if 'world_points' in predictions:
        points = predictions['world_points'].reshape(-1, 3)
        # Remove any invalid points
        valid_mask = ~np.any(np.isnan(points) | np.isinf(points), axis=1)
        points = points[valid_mask]
        
        if len(points) > 0:
            # Compute bounding box
            pts_min = points.min(axis=0)
            pts_max = points.max(axis=0)
            center = (pts_min + pts_max) / 2
            scale = np.max(pts_max - pts_min)
            
            print(f"  - Scene bounds: min={pts_min}, max={pts_max}")
            print(f"  - Scene center: {center}")
            print(f"  - Scene scale: {scale}")
        else:
            # Default values
            center = [0, 0, 0]
            scale = 1.0
    else:
        center = [0, 0, 0]
        scale = 1.0
    
    # Create transforms.json
    transforms = {
        "frames": frames,
        "scale": float(scale),
        "offset": center.tolist() if isinstance(center, np.ndarray) else center
    }
    
    # Add camera angle for compatibility
    if num_frames > 0:
        w = predictions['images'].shape[2]
        transforms["camera_angle_x"] = float(2 * np.arctan(w / (2 * fx)))
    
    # Save transforms.json
    transforms_path = os.path.join(neuralangelo_data_dir, "transforms.json")
    with open(transforms_path, 'w') as f:
        json.dump(transforms, f, indent=2)
    
    print(f"Created Neuralangelo data directory: {neuralangelo_data_dir}")
    print(f"  - Images: {len(os.listdir(images_output_dir))} files")
    print(f"  - Transforms: {transforms_path}")
    print(f"  - Processed {len(frames)} frames")
    print(f"  - First camera: fx={frames[0]['fl_x']:.2f}, fy={frames[0]['fl_y']:.2f}")
    
    return True

def run_neuralangelo_preprocessing(data_dir, output_dir):
    """Run Neuralangelo preprocessing (COLMAP if needed)"""
    print("\n" + "="*60)
    print("STEP 3: Checking Neuralangelo Data Format")
    print("="*60)
    
    # Since we already have poses from VGG-T in transforms.json format,
    # we might not need additional preprocessing
    # Just copy the data to the preprocessing output directory
    
    print("VGG-T already provided camera poses, skipping COLMAP preprocessing")
    
    # Copy data to output directory
    import shutil
    if os.path.abspath(data_dir) != os.path.abspath(output_dir):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        shutil.copytree(data_dir, output_dir)
    
    print(f"Data prepared at: {output_dir}")
    return True

def run_neuralangelo_reconstruction(data_dir, config_path, output_dir, n_steps=20000):
    """Run Neuralangelo reconstruction"""
    print("\n" + "="*60)
    print("STEP 4: Running Neuralangelo Neural Surface Reconstruction")
    print("="*60)
    
    # Change to Neuralangelo directory
    original_dir = os.getcwd()
    neuralangelo_dir = os.path.join(original_dir, "src/neuralangelo")
    
    if not os.path.exists(neuralangelo_dir):
        print(f"Neuralangelo directory not found at: {neuralangelo_dir}")
        return None
    
    os.chdir(neuralangelo_dir)
    
    try:
        # IMPORTANT: Add the neuralangelo directory to Python path
        sys.path.insert(0, neuralangelo_dir)
        
        # Check what training script exists
        possible_scripts = ["train.py", "scripts/train.py", "main.py", "run.py"]
        train_script = None
        for script in possible_scripts:
            if os.path.exists(script):
                train_script = script
                break
        
        if not train_script:
            print("Could not find Neuralangelo training script!")
            print("Available files:")
            for f in os.listdir("."):
                print(f"  - {f}")
            return None
        
        print(f"Found training script: {train_script}")
        
        # Prepare config file if not provided
        if config_path is None:
            # Create a custom config based on template
            template_path = "projects/neuralangelo/configs/custom/template.yaml"
            if os.path.exists(template_path):
                print(f"Creating custom config from template: {template_path}")
                
                # Read template
                with open(template_path, 'r') as f:
                    config_content = f.read()
                
                # Get number of images
                num_images = len(glob.glob(os.path.join(data_dir, "images", "*")))
                
                # Modify config with our data path and settings
                config_content = config_content.replace(
                    "root:  # The root path of the dataset.",
                    f"root: {os.path.abspath(data_dir)}"
                )
                config_content = config_content.replace(
                    "num_images:  # The number of training images.",
                    f"num_images: {num_images}"
                )
                
                # Adjust max iterations based on our request
                parent_line = "*parent*: projects/neuralangelo/configs/base.yaml"
                config_content = config_content.replace(
                    parent_line,
                    f"{parent_line}\n\nmax_iter: {n_steps}"
                )
                
                # Use smaller image size for faster training (optional)
                config_content = config_content.replace(
                    "image_size: [1200,1600]",
                    "image_size: [350,518]"  # Match VGG-T output size
                )
                
                # Save custom config
                config_path = os.path.join(output_dir, "custom_config.yaml")
                os.makedirs(output_dir, exist_ok=True)
                with open(config_path, 'w') as f:
                    f.write(config_content)
                
                print(f"Created custom config: {config_path}")
                print(f"  - Data root: {data_dir}")
                print(f"  - Number of images: {num_images}")
                print(f"  - Max iterations: {n_steps}")
            else:
                print(f"Template config not found at: {template_path}")
                # Try other configs
                config_candidates = [
                    "projects/neuralangelo/configs/base.yaml",
                    "neuralangelo.yaml"
                ]
                
                for cfg in config_candidates:
                    if os.path.exists(cfg):
                        config_path = cfg
                        print(f"Using config: {config_path}")
                        break
                
                if config_path is None:
                    print("No config file found!")
                    return None
        
        # Run training - Neuralangelo expects minimal command line args
        train_cmd = [
            sys.executable,
            train_script,
            "--config", config_path,
            "--logdir", os.path.abspath(output_dir)
        ]
        
        # Optionally add single GPU flag if not using distributed training
        train_cmd.extend(["--single_gpu"])
        
        print(f"Running: {' '.join(train_cmd)}")
        print(f"Training for {n_steps} iterations...")
        print(f"Config file: {config_path}")
        print(f"Log directory: {output_dir}")
        
        # Run with real-time output
        process = subprocess.Popen(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        # Capture all output
        full_output = []
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line:
                full_output.append(line)
                # Print important lines
                if any(keyword in line.lower() for keyword in ["error", "iteration", "step", "loss", "saving", "traceback", "exception"]):
                    print(line)
        
        process.wait()
        
        if process.returncode != 0:
            print("\n" + "="*60)
            print("Training failed with full error output:")
            print("="*60)
            for line in full_output[-50:]:  # Print last 50 lines
                print(line)
            print("="*60)
            
            # Also try to run with Python's traceback
            print("\nTrying to get more detailed error...")
            try:
                result = subprocess.run(train_cmd, capture_output=True, text=True, timeout=30)
                if result.stderr:
                    print("STDERR:")
                    print(result.stderr)
                if result.stdout:
                    print("STDOUT:")
                    print(result.stdout)
            except subprocess.TimeoutExpired:
                print("(Timed out while getting detailed error)")
            
            return None
        
        # Look for output mesh
        mesh_candidates = [
            os.path.join(output_dir, "mesh.ply"),
            os.path.join(output_dir, "meshes", "mesh.ply"),
            os.path.join(output_dir, "mesh", "mesh.ply"),
            os.path.join(output_dir, "reconstruction.ply")
        ]
        
        mesh_path = None
        for candidate in mesh_candidates:
            if os.path.exists(candidate):
                mesh_path = candidate
                break
        
        # If no mesh found, try to extract it
        if not mesh_path:
            print("No mesh found, attempting extraction...")
            # Look for checkpoint
            ckpt_path = os.path.join(output_dir, "checkpoint.pth")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(output_dir, "checkpoints", "latest.pth")
            
            if os.path.exists(ckpt_path):
                mesh_output_dir = os.path.join(output_dir, "meshes")
                os.makedirs(mesh_output_dir, exist_ok=True)
                mesh_path = os.path.join(mesh_output_dir, "reconstruction.ply")
                
                extract_cmd = [
                    sys.executable,
                    "scripts/extract_mesh.py",
                    "--config", config_path,
                    "--checkpoint", ckpt_path,
                    "--output", mesh_path,
                    "--resolution", "512"
                ]
                
                print(f"Extracting mesh: {' '.join(extract_cmd)}")
                result = subprocess.run(extract_cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print("Mesh extraction failed:", result.stderr)
                    mesh_path = None
        
        if mesh_path and os.path.exists(mesh_path):
            print(f"✓ Neuralangelo reconstruction complete!")
            return mesh_path
        else:
            print("Warning: Training completed but no mesh file found")
            print(f"Check output directory: {output_dir}")
            return None
        
    except Exception as e:
        print(f"Neuralangelo failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        os.chdir(original_dir)

def main():
    parser = argparse.ArgumentParser(description="Complete 3D reconstruction pipeline: VGG-T → Neuralangelo")
    parser.add_argument("input_dir", help="Directory containing images")
    parser.add_argument("--output_dir", default=None, 
                       help="Output directory for all results (default: input_dir)")
    parser.add_argument("--neuralangelo_steps", type=int, default=20000,
                       help="Number of Neuralangelo training steps")
    parser.add_argument("--skip_neuralangelo", action="store_true",
                       help="Skip Neuralangelo reconstruction (only run VGG-T)")
    parser.add_argument("--config", default=None,
                       help="Path to custom Neuralangelo config file")
    parser.add_argument("--force-rerun", action="store_true",
                       help="Force re-run of all steps even if outputs exist")
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.abspath(args.input_dir)
    
    # Create output directories
    vggt_output = os.path.join(output_dir, "vggt_output")
    neuralangelo_data = os.path.join(output_dir, "neuralangelo_data")
    neuralangelo_preprocess = os.path.join(output_dir, "neuralangelo_preprocessed")
    final_output = os.path.join(output_dir, "neuralangelo_output")
    
    os.makedirs(vggt_output, exist_ok=True)
    os.makedirs(neuralangelo_data, exist_ok=True)
    os.makedirs(neuralangelo_preprocess, exist_ok=True)
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
                if os.path.islink(temp_images_dir):
                    os.unlink(temp_images_dir)
                else:
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
    # Check if predictions already exist
    existing_pkl = glob.glob(os.path.join(vggt_output, "vggt_predictions_*.pkl"))
    if existing_pkl and not args.force_rerun:
        print(f"\n✓ Found existing VGG-T predictions: {existing_pkl[0]}")
        print("Skipping VGG-T extraction (use --force-rerun to reprocess)")
        predictions_file = existing_pkl[0]
        vggt_time = 0
    else:
        predictions_file = run_vggt_pipeline(input_dir, vggt_output)
        if not predictions_file:
            print("Pipeline failed at VGG-T stage")
            return 1
        
        vggt_time = time.time() - start_time
        print(f"\nVGG-T completed in {vggt_time:.2f} seconds")
    
    if args.skip_neuralangelo:
        print("\nSkipping Neuralangelo reconstruction as requested")
        print(f"VGG-T predictions saved to: {predictions_file}")
        return 0
    
    # Step 2: Convert to Neuralangelo format
    convert_start = time.time()
    if not convert_to_neuralangelo_format(predictions_file, neuralangelo_data, images_dir):
        print("Pipeline failed at conversion stage")
        return 1
    
    convert_time = time.time() - convert_start
    print(f"\nConversion completed in {convert_time:.2f} seconds")
    
    # Step 3: Neuralangelo preprocessing
    preprocess_start = time.time()
    if not run_neuralangelo_preprocessing(neuralangelo_data, neuralangelo_preprocess):
        print("Pipeline failed at preprocessing stage")
        return 1
    
    preprocess_time = time.time() - preprocess_start
    print(f"\nPreprocessing completed in {preprocess_time:.2f} seconds")
    
    # Step 4: Neuralangelo reconstruction
    neuralangelo_start = time.time()
    mesh_path = run_neuralangelo_reconstruction(
        neuralangelo_preprocess, 
        args.config,
        final_output, 
        args.neuralangelo_steps
    )
    
    if not mesh_path:
        print("Pipeline failed at Neuralangelo stage")
        print("Note: You can still use the VGG-T predictions at:", predictions_file)
        return 1
    
    neuralangelo_time = time.time() - neuralangelo_start
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE!")
    print("="*60)
    print(f"Timing breakdown:")
    print(f"  VGG-T extraction: {vggt_time:.2f} seconds")
    print(f"  Format conversion: {convert_time:.2f} seconds")
    print(f"  Preprocessing: {preprocess_time:.2f} seconds")
    print(f"  Neuralangelo reconstruction: {neuralangelo_time:.2f} seconds")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"\nOutputs:")
    print(f"  VGG-T predictions: {predictions_file}")
    print(f"  Neuralangelo data: {neuralangelo_data}")
    print(f"  Final mesh: {mesh_path}")
    print(f"  All outputs in: {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
