import numpy as np
import json
import os

def create_turntable_poses(n_views, radius=3.0, height=0.0, look_at=[0,0,0]):
    """Create camera poses for turntable capture"""
    poses = []
    
    for i in range(n_views):
        # Angle for this view
        angle = 2 * np.pi * i / n_views
        
        # Camera position on circle
        cam_pos = np.array([
            radius * np.cos(angle),
            height,
            radius * np.sin(angle)
        ])
        
        # Look-at matrix (camera looking at origin)
        forward = np.array(look_at) - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        # Assume up is Y axis
        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Build camera-to-world matrix
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward  # OpenGL convention
        c2w[:3, 3] = cam_pos
        
        poses.append(c2w)
    
    return poses

# Load existing transforms
with open('/home/jhennessy7.gatech/scratch/test_run/neuralangelo_preprocessed/transforms.json', 'r') as f:
    data = json.load(f)

# Count unique cameras
n_frames = len(data['frames'])
print(f"Total frames: {n_frames}")

# Assuming even distribution across 3 cameras
n_views_per_camera = n_frames // 3
print(f"Views per camera: {n_views_per_camera}")

# Create proper poses
all_poses = []
for cam_idx in range(3):
    # Each camera at different height/radius
    height = -0.5 + cam_idx * 0.5  # Different heights
    radius = 2.0 + cam_idx * 0.3   # Different distances
    
    poses = create_turntable_poses(n_views_per_camera, radius=radius, height=height)
    all_poses.extend(poses)

# Update transforms
for i, frame in enumerate(data['frames'][:len(all_poses)]):
    frame['transform_matrix'] = all_poses[i].tolist()

# Save fixed transforms
output_path = '/home/jhennessy7.gatech/scratch/test_run/neuralangelo_preprocessed/transforms_fixed.json'
with open(output_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Saved fixed transforms to {output_path}")

# Also save for data directory
