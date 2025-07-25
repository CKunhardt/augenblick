#!/usr/bin/env python3
"""
Script to add all missing parameters to transforms.json for Neuralangelo compatibility
"""

import json
import sys
import shutil
from pathlib import Path
import numpy as np

def compute_sphere_params(data):
    """Compute sphere normalization parameters from camera positions"""
    
    # Extract camera positions from transform matrices
    camera_positions = []
    for frame in data.get('frames', []):
        transform = np.array(frame['transform_matrix'])
        # Camera position is the translation component (last column, first 3 rows)
        camera_pos = transform[:3, 3]
        camera_positions.append(camera_pos)
    
    if not camera_positions:
        print("Warning: No frames found, using default sphere parameters")
        return [0.0, 0.0, 0.0], 1.0
    
    camera_positions = np.array(camera_positions)
    
    # Compute sphere center as mean of camera positions
    sphere_center = camera_positions.mean(axis=0).tolist()
    
    # Compute sphere radius as max distance from center to any camera
    distances = np.linalg.norm(camera_positions - sphere_center, axis=1)
    sphere_radius = float(distances.max() * 1.1)  # Add 10% margin
    
    return sphere_center, sphere_radius

def add_neuralangelo_params(json_file):
    """Add all missing parameters required by Neuralangelo"""
    
    # Create backup
    backup_file = json_file.with_suffix('.json.backup')
    if not backup_file.exists():  # Don't overwrite existing backup
        shutil.copy2(json_file, backup_file)
        print(f"Created backup: {backup_file}")
    
    # Load the JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Add skew parameters if missing
    if 'sk_x' not in data:
        data['sk_x'] = 0.0
        print("Added sk_x: 0.0")
    
    if 'sk_y' not in data:
        data['sk_y'] = 0.0
        print("Added sk_y: 0.0")
    
    # Compute and add sphere parameters if missing
    if 'sphere_center' not in data or 'sphere_radius' not in data:
        sphere_center, sphere_radius = compute_sphere_params(data)
        
        if 'sphere_center' not in data:
            data['sphere_center'] = sphere_center
            print(f"Added sphere_center: {sphere_center}")
        
        if 'sphere_radius' not in data:
            data['sphere_radius'] = sphere_radius
            print(f"Added sphere_radius: {sphere_radius}")
    
    # Add other potentially missing parameters with sensible defaults
    params_with_defaults = {
        'near': 0.01,  # Near clipping plane
        'far': 100.0,   # Far clipping plane
        'integer_depth_scale': 1.0,  # Depth scale factor
        'background': 1.0,  # Background color (1.0 = white, 0.0 = black)
    }
    
    for param, default_value in params_with_defaults.items():
        if param not in data:
            data[param] = default_value
            print(f"Added {param}: {default_value}")
    
    # Save the modified JSON
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSuccessfully updated {json_file}")
    
    # Verify the changes
    required_params = ['sk_x', 'sk_y', 'sphere_center', 'sphere_radius']
    with open(json_file, 'r') as f:
        verify_data = json.load(f)
        
    missing = [p for p in required_params if p not in verify_data]
    if not missing:
        print("✓ Verification passed - all required parameters present")
        return True
    else:
        print(f"✗ Verification failed - missing parameters: {missing}")
        return False

def check_current_params(json_file):
    """Check which parameters are currently present"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print("\nCurrent parameters in transforms.json:")
    expected_params = ['fl_x', 'fl_y', 'cx', 'cy', 'sk_x', 'sk_y', 
                      'sphere_center', 'sphere_radius', 'near', 'far',
                      'k1', 'k2', 'p1', 'p2', 'w', 'h']
    
    for param in expected_params:
        if param in data:
            print(f"  ✓ {param}: {data[param]}")
        else:
            print(f"  ✗ {param}: MISSING")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_transforms_neuralangelo.py <path_to_transforms.json>")
        sys.exit(1)
    
    json_file = Path(sys.argv[1])
    
    if not json_file.exists():
        print(f"Error: File '{json_file}' not found")
        sys.exit(1)
    
    if json_file.suffix != '.json':
        print(f"Error: File '{json_file}' is not a JSON file")
        sys.exit(1)
    
    # First check what we have
    check_current_params(json_file)
    
    # Then add missing parameters
    print("\nAdding missing parameters...")
    success = add_neuralangelo_params(json_file)
    
    # Show final state
    if success:
        print("\nFinal state:")
        check_current_params(json_file)
    
    sys.exit(0 if success else 1)
