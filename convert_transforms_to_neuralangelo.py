#!/usr/bin/env python3
"""
Convert transforms.json from VGGT format to Neuralangelo format
"""

import json
import numpy as np
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_transforms(input_path: Path, output_path: Path):
    """Convert transforms.json to Neuralangelo format"""
    
    # Load transforms
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Get first frame to extract camera parameters
    first_frame = data['frames'][0]
    if first_frame['intrinsics'] is None:
        logger.error("No intrinsics found in transforms.json!")
        return False
    
    # Extract global intrinsics (assuming same camera for all frames)
    intrinsics = first_frame['intrinsics']
    
    # Calculate scene bounds from camera positions
    camera_positions = []
    for frame in data['frames']:
        if frame['transform_matrix'] is not None:
            transform = np.array(frame['transform_matrix'])
            # Camera position is the last column of the transform matrix
            pos = transform[:3, 3]
            camera_positions.append(pos)
    
    if camera_positions:
        camera_positions = np.array(camera_positions)
        scene_center = np.mean(camera_positions, axis=0)
        scene_radius = np.max(np.linalg.norm(camera_positions - scene_center, axis=1)) * 1.5
    else:
        scene_center = [0.0, 0.0, 0.0]
        scene_radius = 2.0
    
    # Convert to Neuralangelo format
    neuralangelo_format = {
        "fl_x": intrinsics['fx'],
        "fl_y": intrinsics['fy'],
        "cx": intrinsics['cx'],
        "cy": intrinsics['cy'],
        "sk_x": 0.0,  # Skew, usually 0
        "sk_y": 0.0,
        "k1": 0.0,    # Distortion coefficients
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "camera_model": "OPENCV",
        "w": int(intrinsics.get('width', 6240)),   # Image width
        "h": int(intrinsics.get('height', 4160)),  # Image height
        "aabb_scale": 4,  # Scene scale (increased for better coverage)
        "scale": 1.0,
        "offset": [0.0, 0.0, 0.0],
        "sphere_center": scene_center.tolist() if isinstance(scene_center, np.ndarray) else scene_center,
        "sphere_radius": float(scene_radius),
        "near": 0.01,
        "far": 100.0,
        "frames": []
    }
    
    # Process each frame
    for idx, frame in enumerate(data['frames']):
        if frame['transform_matrix'] is None:
            logger.warning(f"Frame {idx} missing transform matrix!")
            continue
        
        # Use transform matrix as-is (VGGT should have correct coordinate system)
        transform = frame['transform_matrix']
        
        frame_data = {
            "file_path": frame['file_path'],
            "transform_matrix": transform,
            "sharpness": 30.0  # Default sharpness value
        }
        
        neuralangelo_format['frames'].append(frame_data)
    
    # Save in Neuralangelo format
    with open(output_path, 'w') as f:
        json.dump(neuralangelo_format, f, indent=2)
    
    logger.info(f"Converted {len(neuralangelo_format['frames'])} frames")
    logger.info(f"Camera parameters: fx={neuralangelo_format['fl_x']:.1f}, fy={neuralangelo_format['fl_y']:.1f}")
    logger.info(f"Image dimensions: {neuralangelo_format['w']}x{neuralangelo_format['h']}")
    logger.info(f"Scene sphere: center={neuralangelo_format['sphere_center']}, radius={neuralangelo_format['sphere_radius']:.2f}")
    logger.info(f"Saved to: {output_path}")
    
    return True


def verify_neuralangelo_format(transforms_path: Path):
    """Verify the converted format has all required fields"""
    with open(transforms_path, 'r') as f:
        data = json.load(f)
    
    required_fields = ['fl_x', 'fl_y', 'cx', 'cy', 'w', 'h', 'frames']
    missing = [field for field in required_fields if field not in data]
    
    if missing:
        logger.error(f"Missing required fields: {missing}")
        return False
    
    logger.info("✓ All required fields present")
    logger.info(f"✓ {len(data['frames'])} frames with transforms")
    
    # Check if all frames have transform matrices
    null_count = sum(1 for frame in data['frames'] 
                     if frame.get('transform_matrix') is None)
    if null_count > 0:
        logger.warning(f"⚠ {null_count} frames missing transform matrices")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert transforms to Neuralangelo format")
    parser.add_argument("input", type=Path, help="Input transforms.json")
    parser.add_argument("--output", type=Path, help="Output path (default: transforms_neuralangelo.json)")
    parser.add_argument("--verify-only", action="store_true", help="Only verify format")
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_neuralangelo_format(args.input)
        return
    
    output_path = args.output or args.input.parent / "transforms_neuralangelo.json"
    
    success = convert_transforms(args.input, output_path)
    
    if success:
        verify_neuralangelo_format(output_path)


if __name__ == "__main__":
    main()
