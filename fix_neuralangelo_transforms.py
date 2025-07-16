#!/usr/bin/env python
"""
Fix transforms.json to match Neuralangelo's expected format
"""

import json
import sys
import numpy as np

def fix_neuralangelo_format(json_path):
    """Convert transforms.json to Neuralangelo's expected format"""
    
    # Load the JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get camera parameters from first frame (assuming all frames have same intrinsics)
    if data['frames'] and 'fl_x' in data['frames'][0]:
        first_frame = data['frames'][0]
        
        # Add camera parameters at top level
        data['fl_x'] = first_frame['fl_x']
        data['fl_y'] = first_frame['fl_y']
        data['cx'] = first_frame['cx']
        data['cy'] = first_frame['cy']
        data['w'] = first_frame.get('w', 518)
        data['h'] = first_frame.get('h', 350)
        data['sk_x'] = 0.0  # Skew parameter (usually 0)
        data['sk_y'] = 0.0
        
        # Calculate camera angle if not present
        if 'camera_angle_x' not in data:
            data['camera_angle_x'] = 2 * np.arctan(data['w'] / (2 * data['fl_x']))
        
        # Remove per-frame intrinsics (keep only in top level)
        for frame in data['frames']:
            # Keep only file_path and transform_matrix in frames
            keys_to_keep = ['file_path', 'transform_matrix']
            keys_to_remove = [k for k in frame.keys() if k not in keys_to_keep]
            for k in keys_to_remove:
                frame.pop(k, None)
    
    # Add sphere_center if not present (required by Neuralangelo)
    if 'sphere_center' not in data:
        data['sphere_center'] = [0.0, 0.0, 0.0]
    
    # Add sphere_radius if not present
    if 'sphere_radius' not in data:
        data['sphere_radius'] = 1.0
    
    # Add other parameters Neuralangelo might expect
    if 'scale' not in data:
        data['scale'] = 1.0
    if 'offset' not in data:
        data['offset'] = [0.0, 0.0, 0.0]
    
    # Save backup
    import shutil
    backup_path = json_path + '.neuralangelo_backup'
    shutil.copy(json_path, backup_path)
    print(f"Created backup: {backup_path}")
    
    # Save fixed version
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Fixed transforms.json for Neuralangelo format")
    print(f"Camera parameters:")
    print(f"  fl_x: {data.get('fl_x', 'N/A')}")
    print(f"  fl_y: {data.get('fl_y', 'N/A')}")
    print(f"  cx: {data.get('cx', 'N/A')}")
    print(f"  cy: {data.get('cy', 'N/A')}")
    print(f"  w: {data.get('w', 'N/A')}")
    print(f"  h: {data.get('h', 'N/A')}")
    print(f"  sphere_center: {data.get('sphere_center', 'N/A')}")
    print(f"  sphere_radius: {data.get('sphere_radius', 'N/A')}")
    print(f"  Number of frames: {len(data['frames'])}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        json_path = "/home/jhennessy7.gatech/scratch/test_run/neuralangelo_data/transforms.json"
    
    fix_neuralangelo_format(json_path)
