#!/usr/bin/env python3
"""
Convert transforms.json to use dictionary format intrinsics
"""

import json
import sys
from pathlib import Path
import shutil

def convert_to_dict_intrinsics(input_path, output_path=None):
    """Convert transforms.json to have per-frame intrinsics in dictionary format"""
    
    # Load the transforms
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Extract global intrinsics
    if 'fl_x' not in data:
        print("Error: No global intrinsics found in transforms.json")
        return False
    
    # Build intrinsics dictionary
    intrinsics_dict = {
        'fx': data['fl_x'],
        'fy': data['fl_y'],
        'cx': data['cx'],
        'cy': data['cy']
    }
    
    print(f"Found global intrinsics:")
    print(f"  Focal length: fx={data['fl_x']:.1f}, fy={data['fl_y']:.1f}")
    print(f"  Principal point: cx={data['cx']:.1f}, cy={data['cy']:.1f}")
    print(f"  Image size: {data['w']}x{data['h']}")
    
    # Convert matrix intrinsics to dictionary format in each frame
    frames_updated = 0
    for frame in data['frames']:
        if 'intrinsics' in frame:
            # Check if it's already in matrix format
            if isinstance(frame['intrinsics'], list):
                # Convert from matrix to dictionary
                frame['intrinsics'] = {
                    'fx': frame['intrinsics'][0][0],
                    'fy': frame['intrinsics'][1][1],
                    'cx': frame['intrinsics'][0][2],
                    'cy': frame['intrinsics'][1][2]
                }
                frames_updated += 1
            elif not isinstance(frame['intrinsics'], dict):
                # If not dict or list, replace with our dict
                frame['intrinsics'] = intrinsics_dict
                frames_updated += 1
        else:
            # Add intrinsics if missing
            frame['intrinsics'] = intrinsics_dict
            frames_updated += 1
    
    print(f"\nUpdated intrinsics format for {frames_updated} frames")
    
    # Save the updated transforms
    if output_path is None:
        # Backup original
        backup_path = input_path.replace('.json', '_matrix_backup.json')
        shutil.copy(input_path, backup_path)
        print(f"Backed up original to: {backup_path}")
        output_path = input_path
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Updated transforms saved to: {output_path}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_dict_intrinsics.py <path_to_transforms.json> [output_path]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if convert_to_dict_intrinsics(input_path, output_path):
        print("\nSuccess! Intrinsics are now in dictionary format")
    else:
        print("\nFailed to convert transforms")
        sys.exit(1)
