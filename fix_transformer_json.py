#!/usr/bin/env python
"""
Fix transforms.json to remove mask files and ensure proper format
"""

import json
import sys

def fix_transforms(json_path):
    """Remove mask files from transforms.json"""
    
    # Load the JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Filter out mask files
    original_count = len(data['frames'])
    data['frames'] = [
        frame for frame in data['frames'] 
        if not frame['file_path'].endswith('.mask.png')
    ]
    
    # Add width and height to each frame if missing
    for frame in data['frames']:
        if 'w' not in frame:
            frame['w'] = 518  # Based on VGG-T output
        if 'h' not in frame:
            frame['h'] = 350  # Based on VGG-T output
    
    new_count = len(data['frames'])
    print(f"Filtered {original_count - new_count} mask files")
    print(f"Remaining frames: {new_count}")
    
    # Save backup
    import shutil
    backup_path = json_path + '.backup'
    shutil.copy(json_path, backup_path)
    print(f"Created backup: {backup_path}")
    
    # Save fixed version
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Fixed transforms.json saved to: {json_path}")
    
    # Print first frame as example
    if data['frames']:
        print("\nFirst frame example:")
        print(f"  File: {data['frames'][0]['file_path']}")
        print(f"  fl_x: {data['frames'][0]['fl_x']}")
        print(f"  fl_y: {data['frames'][0]['fl_y']}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        json_path = "/home/jhennessy7.gatech/scratch/test_run/neuralangelo_data/transforms.json"
    
    fix_transforms(json_path)
