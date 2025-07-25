#!/usr/bin/env python3
"""
Quick fix for camera orientations in transforms.json

This script applies simple transformations to fix camera orientations.
"""

import json
import numpy as np
import argparse

def apply_camera_fixes(transforms_file, output_file):
    """Apply various camera orientation fixes."""
    
    print(f"Loading transforms from: {transforms_file}")
    with open(transforms_file, 'r') as f:
        data = json.load(f)
    
    frames = data['frames']
    print(f"Found {len(frames)} camera frames")
    
    print("\nTrying different camera orientation fixes...")
    
    # Fix 1: Flip Z-axis (camera forward direction)
    print("Fix 1: Flipping camera Z-axis (forward direction)")
    for frame in frames:
        transform = np.array(frame['transform_matrix'])
        transform[:3, 2] *= -1  # Flip Z-axis
        frame['transform_matrix'] = transform.tolist()
    
    # Save fixed version
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"   Applied Z-axis flip fix")
    print(f"   Saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Quick fix for camera orientations")
    parser.add_argument("input_file", help="Input transforms.json file")
    parser.add_argument("--output", default="transforms_fixed_simple.json", help="Output file")
    
    args = parser.parse_args()
    
    apply_camera_fixes(args.input_file, args.output)

if __name__ == "__main__":
    main()