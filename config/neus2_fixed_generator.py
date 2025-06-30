#!/usr/bin/env python3
"""
Fixed NeuS2 Transform Generator
Based on analysis of the NeuS2 run.py code and training requirements
"""

import numpy as np
import json
import argparse
import math
from pathlib import Path
from PIL import Image
import os
import glob

def get_image_dimensions(image_dir):
    """Get image dimensions from the first image in the directory"""
    if not os.path.exists(image_dir):
        return None
        
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif', '*.bmp']
    image_files = []
    
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    image_files.sort()
    
    if not image_files:
        return None
    
    try:
        with Image.open(image_files[0]) as img:
            return img.size
    except Exception as e:
        print(f"Warning: Could not read image {image_files[0]}: {e}")
        return None

def create_camera_intrinsics(focal_length_mm=50, sensor_width_mm=35.9, sensor_height_mm=24.0, 
                           image_width=1600, image_height=1200):
    """Calculate camera intrinsics for Canon EOS RP"""
    fx = (focal_length_mm / sensor_width_mm) * image_width
    fy = (focal_length_mm / sensor_height_mm) * image_height
    
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    camera_angle_x = 2 * math.atan(sensor_width_mm / (2 * focal_length_mm))
    camera_angle_y = 2 * math.atan(sensor_height_mm / (2 * focal_length_mm))
    
    return {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'camera_angle_x': camera_angle_x,
        'camera_angle_y': camera_angle_y,
        'k1': 0.0,
        'k2': 0.0,
        'k3': 0.0,
        'k4': 0.0,
        'p1': 0.0,
        'p2': 0.0
    }

def create_rotation_matrix_y(angle_degrees):
    """Create rotation matrix for Y-axis rotation (turntable)"""
    theta = math.radians(angle_degrees)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    
    return np.array([
        [cos_theta, 0, sin_theta, 0],
        [0, 1, 0, 0],
        [-sin_theta, 0, cos_theta, 0],
        [0, 0, 0, 1]
    ])

def create_camera_positions(distance_from_center=2.0, camera_spacing=0.5):
    """Create camera positions for line array setup"""
    camera_configs = []
    
    # Camera positions in line array (stacked vertically)
    base_positions = [
        np.array([distance_from_center, -camera_spacing, 0, 1]),  # Camera 1 (bottom)
        np.array([distance_from_center, 0, 0, 1]),                # Camera 2 (middle)  
        np.array([distance_from_center, camera_spacing, 0, 1])     # Camera 3 (top)
    ]
    
    for i, pos in enumerate(base_positions):
        camera_pos = pos[:3]
        look_at = np.array([0, 0, 0])
        up = np.array([0, 0, 1])
        
        # Calculate camera orientation
        forward = look_at - camera_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        camera_up = np.cross(right, forward)
        camera_up = camera_up / np.linalg.norm(camera_up)
        
        # Create 4x4 transform matrix (camera-to-world)
        transform = np.eye(4)
        transform[0, :3] = right
        transform[1, :3] = camera_up  
        transform[2, :3] = -forward
        transform[:3, 3] = camera_pos
        
        camera_configs.append({
            'position': camera_pos,
            'transform': transform,
            'camera_id': i
        })
    
    return camera_configs

def apply_neus2_coordinate_transform(transform):
    """
    Apply coordinate system transform specifically for NeuS2
    Based on the reference transforms.json format
    """
    # NeuS2 seems to use a different coordinate convention
    # Looking at the reference, we need to match that exactly
    
    # Apply the coordinate system conversion
    flip_matrix = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0], 
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    
    return np.dot(transform, flip_matrix)

def setup_mask_directory(image_dir, mask_dir):
    """
    Set up mask directory structure that NeuS2 expects
    """
    if not mask_dir or not os.path.exists(mask_dir):
        return None
    
    # NeuS2 expects masks in the same directory structure as images
    # Let's verify the mask setup
    image_files = []
    mask_files = []
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif', '*.bmp']
    
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
        mask_files.extend(glob.glob(os.path.join(mask_dir, ext)))
        mask_files.extend(glob.glob(os.path.join(mask_dir, ext.upper())))
    
    image_files.sort()
    mask_files.sort()
    
    print(f"Found {len(image_files)} images and {len(mask_files)} masks")
    
    return len(mask_files) > 0

def generate_neus2_transforms(image_dir, mask_dir=None, output_path="transforms.json",
                             num_rotation_steps=45, rotation_step_degrees=8, 
                             image_width=None, image_height=None,
                             distance_from_center=2.0, camera_spacing=0.5,
                             aabb_scale=4, scale=0.5):
    """
    Generate transforms.json specifically for NeuS2
    """
    
    # Auto-detect image dimensions
    if image_width is None or image_height is None:
        detected_dims = get_image_dimensions(image_dir)
        if detected_dims:
            image_width, image_height = detected_dims
            print(f"Auto-detected image dimensions: {image_width} x {image_height}")
        else:
            image_width = image_width or 1600
            image_height = image_height or 1200
            print(f"Using default dimensions: {image_width} x {image_height}")
    
    # Check mask setup
    has_masks = setup_mask_directory(image_dir, mask_dir)
    
    # Calculate camera intrinsics
    intrinsics = create_camera_intrinsics(image_width=image_width, image_height=image_height)
    
    # Get camera positions
    camera_configs = create_camera_positions(distance_from_center, camera_spacing)
    
    # Create the transforms JSON structure for NeuS2
    transforms_data = {
        "w": int(image_width),
        "h": int(image_height),
        "aabb_scale": aabb_scale,
        "scale": scale,
        "offset": [0.5, 0.5, 0.5],
        "frames": []
    }
    
    # Add camera parameters (NeuS2 format)
    transforms_data.update({
        "camera_angle_x": intrinsics['camera_angle_x'],
        "camera_angle_y": intrinsics['camera_angle_y'],
        "fl_x": intrinsics['fx'],
        "fl_y": intrinsics['fy'],
        "k1": intrinsics['k1'],
        "k2": intrinsics['k2'],
        "k3": intrinsics['k3'], 
        "k4": intrinsics['k4'],
        "p1": intrinsics['p1'],
        "p2": intrinsics['p2'],
        "cx": intrinsics['cx'],
        "cy": intrinsics['cy'],
        "is_fisheye": False
    })
    
    frame_idx = 0
    
    # Generate frames
    for rotation_step in range(num_rotation_steps):
        rotation_angle = rotation_step * rotation_step_degrees
        rotation_matrix = create_rotation_matrix_y(rotation_angle)
        
        for cam_idx, camera_config in enumerate(camera_configs):
            # Apply turntable rotation
            world_transform = np.dot(rotation_matrix, camera_config['transform'])
            
            # Apply NeuS2 coordinate system transformation
            final_transform = apply_neus2_coordinate_transform(world_transform)
            
            # Create frame entry (simplified for NeuS2)
            frame = {
                "file_path": f"images/{frame_idx:06d}.png",
                "transform_matrix": final_transform.tolist()
            }
            
            # Add per-frame intrinsics (NeuS2 supports this)
            intrinsic_matrix = [
                [intrinsics['fx'], 0.0, intrinsics['cx'], 0.0],
                [0.0, intrinsics['fy'], intrinsics['cy'], 0.0], 
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
            frame["intrinsic_matrix"] = intrinsic_matrix
            
            transforms_data["frames"].append(frame)
            frame_idx += 1
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(transforms_data, f, indent=2)
    
    print(f"\n=== NeuS2 Transform Generation Complete ===")
    print(f"Generated transforms.json with {len(transforms_data['frames'])} frames")
    print(f"Image dimensions: {image_width} x {image_height}")
    print(f"AABB scale: {aabb_scale}")
    print(f"Masks detected: {'Yes' if has_masks else 'No'}")
    print(f"Saved to: {output_path}")
    
    return transforms_data

def create_neus2_training_script(data_dir, output_name="neus2_test"):
    """
    Create a training script for NeuS2
    """
    script_content = f"""#!/bin/bash

# NeuS2 Training Script
# Generated for data directory: {data_dir}

echo "Starting NeuS2 training..."

# Basic training command
python run.py \\
    --name {output_name} \\
    --scene {data_dir} \\
    --mode nerf \\
    --n_steps 5000 \\
    --save_mesh \\
    --marching_cubes_res 256

echo "Training complete! Check output/{output_name}/ for results"

# Optional: Run evaluation
# python run.py \\
#     --name {output_name} \\
#     --scene {data_dir} \\
#     --test \\
#     --load_snapshot output/{output_name}/checkpoints/5000.msgpack \\
#     --save_mesh \\
#     --marching_cubes_res 512
"""
    
    with open("train_neus2.sh", "w") as f:
        f.write(script_content)
    
    os.chmod("train_neus2.sh", 0o755)
    print("Created training script: train_neus2.sh")

def main():
    parser = argparse.ArgumentParser(description="Generate NeuS2-specific transforms.json")
    
    parser.add_argument("--image_dir", type=str, required=True,
                       help="Path to directory containing input images")
    parser.add_argument("--mask_dir", type=str, default=None,
                       help="Path to directory containing mask images")
    parser.add_argument("--output", type=str, default="transforms.json",
                       help="Output transforms.json path")
    
    parser.add_argument("--width", type=int, default=None,
                       help="Image width (auto-detected if not specified)")
    parser.add_argument("--height", type=int, default=None, 
                       help="Image height (auto-detected if not specified)")
    
    parser.add_argument("--rotation_steps", type=int, default=45, 
                       help="Number of rotation positions")
    parser.add_argument("--rotation_degrees", type=float, default=8.0,
                       help="Degrees per rotation step")
    
    parser.add_argument("--distance", type=float, default=2.0,
                       help="Camera distance from center in meters")
    parser.add_argument("--spacing", type=float, default=0.5,
                       help="Camera spacing in meters")
    
    parser.add_argument("--aabb_scale", type=int, default=4,
                       help="Scene bounding box scale")
    parser.add_argument("--scale", type=float, default=0.5,
                       help="Global scene scale")
    
    parser.add_argument("--create_training_script", action="store_true",
                       help="Create a training script for NeuS2")
    
    args = parser.parse_args()
    
    # Generate transforms
    generate_neus2_transforms(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_path=args.output,
        num_rotation_steps=args.rotation_steps,
        rotation_step_degrees=args.rotation_degrees,
        image_width=args.width,
        image_height=args.height,
        distance_from_center=args.distance,
        camera_spacing=args.spacing,
        aabb_scale=args.aabb_scale,
        scale=args.scale
    )
    
    # Create training script if requested
    if args.create_training_script:
        data_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "."
        create_neus2_training_script(data_dir)
    
    print(f"\n=== Next Steps ===")
    print(f"1. Place your images in: {args.image_dir}")
    if args.mask_dir:
        print(f"2. Place your masks in: {args.mask_dir}")
    print(f"3. Run NeuS2 training:")
    print(f"   python run.py --name test --scene . --n_steps 5000 --save_mesh")

if __name__ == "__main__":
    main()