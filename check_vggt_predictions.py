#!/usr/bin/env python
"""
Check the structure of VGG-T predictions file
"""

import pickle
import sys
import numpy as np

def check_predictions(pkl_file):
    """Examine the structure of VGG-T predictions"""
    print(f"Loading predictions from: {pkl_file}")
    
    with open(pkl_file, 'rb') as f:
        predictions = pickle.load(f)
    
    print("\n=== Predictions Structure ===")
    print(f"Type: {type(predictions)}")
    
    if isinstance(predictions, dict):
        print(f"Keys: {list(predictions.keys())}")
        
        for key, value in predictions.items():
            print(f"\n[{key}]:")
            if isinstance(value, (list, tuple)):
                print(f"  Type: {type(value)}, Length: {len(value)}")
                if len(value) > 0:
                    print(f"  First item type: {type(value[0])}")
                    if isinstance(value[0], np.ndarray):
                        print(f"  First item shape: {value[0].shape}")
            elif isinstance(value, np.ndarray):
                print(f"  Type: numpy array, Shape: {value.shape}")
            elif isinstance(value, dict):
                print(f"  Type: dict, Keys: {list(value.keys())[:5]}...")
                if len(value) > 0:
                    first_key = next(iter(value.keys()))
                    print(f"  First item key: {first_key}")
                    first_val = value[first_key]
                    if isinstance(first_val, dict):
                        print(f"  First item value keys: {list(first_val.keys())}")
            else:
                print(f"  Type: {type(value)}")
    
    # Check for camera parameters
    print("\n=== Looking for Camera Parameters ===")
    
    # Common places to find camera params
    if "pose_enc" in predictions:
        print(f"Found pose_enc: shape {predictions['pose_enc'].shape}")
    
    if "intrinsic" in predictions:
        print(f"Found intrinsic: shape {predictions['intrinsic'].shape}")
    
    if "extrinsic" in predictions:
        print(f"Found extrinsic: shape {predictions['extrinsic'].shape}")
    
    if "images" in predictions:
        print(f"Found images: shape {predictions['images'].shape}")
    
    # Check if data is batched
    if "depth" in predictions:
        depth_shape = predictions["depth"].shape
        print(f"\nDepth shape: {depth_shape}")
        print(f"Likely batch size: {depth_shape[0]}")
        print(f"Likely sequence length: {depth_shape[1] if len(depth_shape) > 4 else 'N/A'}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        pkl_file = sys.argv[1]
    else:
        # Default to the file from the error
        pkl_file = "/home/jhennessy7.gatech/scratch/test_run/vggt_output/vggt_predictions_1752432185.pkl"
    
    check_predictions(pkl_file)
