#!/usr/bin/env python3
"""
VGG-T GLB to HeatSDF Data Loader
Loads and prepares VGG-T output for HeatSDF training
"""

import numpy as np
import trimesh
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple
import json

class VGGTDataLoader:
    """Loads VGG-T GLB outputs and prepares them for HeatSDF"""
    
    def __init__(self, glb_path: str, downsample_factor: Optional[float] = None):
        """
        Args:
            glb_path: Path to VGG-T output GLB file
            downsample_factor: If provided, randomly downsample points (e.g., 0.1 keeps 10%)
        """
        self.glb_path = Path(glb_path)
        self.downsample_factor = downsample_factor
        
        # Load and process data
        self.data = self._load_glb()
        
    def _load_glb(self) -> Dict[str, np.ndarray]:
        """Load GLB and extract point cloud data"""
        scene = trimesh.load(self.glb_path, force='scene')
        
        # Extract point cloud (VGG-T stores it as geometry_0)
        if isinstance(scene, trimesh.Scene) and 'geometry_0' in scene.geometry:
            geom = scene.geometry['geometry_0']
            if isinstance(geom, trimesh.PointCloud):
                positions = geom.vertices
                colors = geom.colors if hasattr(geom, 'colors') else None
            else:
                raise ValueError(f"geometry_0 is not a PointCloud but {type(geom)}")
        elif isinstance(scene, trimesh.PointCloud):
            positions = scene.vertices
            colors = scene.colors if hasattr(scene, 'colors') else None
        else:
            raise ValueError("Could not find point cloud in GLB file")
        
        # Normalize colors if needed
        if colors is not None and colors.max() > 1.0:
            colors = colors.astype(np.float32) / 255.0
        
        # Downsample if requested
        if self.downsample_factor is not None and 0 < self.downsample_factor < 1:
            n_points = len(positions)
            n_keep = int(n_points * self.downsample_factor)
            indices = np.random.choice(n_points, n_keep, replace=False)
            positions = positions[indices]
            if colors is not None:
                colors = colors[indices]
            print(f"Downsampled from {n_points:,} to {n_keep:,} points")
        
        data = {'positions': positions}
        if colors is not None:
            # Keep only RGB, discard alpha if present
            data['colors'] = colors[:, :3] if colors.shape[1] > 3 else colors
        
        return data
    
    def get_normalized_positions(self) -> np.ndarray:
        """Get positions normalized to [-1, 1] cube as required by HeatSDF"""
        positions = self.data['positions']
        
        # Compute bounds
        bbox_min = positions.min(axis=0)
        bbox_max = positions.max(axis=0)
        center = (bbox_min + bbox_max) / 2
        scale = (bbox_max - bbox_min).max()
        
        # Normalize to [-1, 1] with some padding
        normalized = 2.0 * (positions - center) / (scale * 1.1)
        
        # Store normalization parameters for later use
        self.normalization = {
            'center': center,
            'scale': scale * 1.1 / 2.0  # Factor to unnormalize
        }
        
        return normalized
    
    def get_surface_points(self, n_points: int = 10000) -> np.ndarray:
        """
        Get a subset of surface points for training
        
        Args:
            n_points: Number of points to sample
        
        Returns:
            Array of shape (n_points, 3) with normalized positions
        """
        normalized = self.get_normalized_positions()
        
        if len(normalized) > n_points:
            indices = np.random.choice(len(normalized), n_points, replace=False)
            return normalized[indices]
        else:
            return normalized
    
    def get_point_weights(self, positions: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
        """
        Compute adaptive weights for non-uniform point clouds (Eq. 11-12 from paper)
        
        Args:
            positions: Point positions (already normalized)
            epsilon: Radius for local density estimation
        
        Returns:
            Weights for each point
        """
        from scipy.spatial import KDTree
        
        # Build KD tree
        tree = KDTree(positions)
        
        # For each point, count neighbors within epsilon
        weights = []
        for i, pos in enumerate(positions):
            # Find neighbors within epsilon
            neighbors = tree.query_ball_point(pos, epsilon)
            # Weight is inverse of local density
            weight = 1.0 / max(len(neighbors), 1)
            weights.append(weight)
        
        weights = np.array(weights)
        # Normalize weights to sum to 1
        weights = weights / weights.sum()
        
        return weights
    
    def get_training_batch(self, batch_size: int = 10000) -> Dict[str, torch.Tensor]:
        """
        Get a batch of data for HeatSDF training
        
        Returns dict with:
            - surface_points: Points on the surface
            - surface_weights: Adaptive weights for each point
            - colors (optional): Colors for each point
        """
        # Sample surface points
        all_positions = self.get_normalized_positions()
        
        if len(all_positions) > batch_size:
            indices = np.random.choice(len(all_positions), batch_size, replace=False)
            positions = all_positions[indices]
            colors = self.data.get('colors', None)
            if colors is not None:
                colors = colors[indices]
        else:
            positions = all_positions
            colors = self.data.get('colors', None)
        
        # Compute adaptive weights
        weights = self.get_point_weights(positions)
        
        # Convert to torch tensors
        batch = {
            'surface_points': torch.FloatTensor(positions),
            'surface_weights': torch.FloatTensor(weights)
        }
        
        if colors is not None:
            batch['colors'] = torch.FloatTensor(colors)
        
        return batch
    
    def save_processed_data(self, output_dir: str = "./processed"):
        """Save processed data for later use"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Get normalized positions
        normalized = self.get_normalized_positions()
        
        # Save data
        save_dict = {
            'positions': normalized,
            'normalization': self.normalization
        }
        
        if 'colors' in self.data:
            save_dict['colors'] = self.data['colors']
        
        # Save as NPZ
        output_path = output_dir / f"{self.glb_path.stem}_processed.npz"
        np.savez(output_path, **save_dict)
        
        # Save normalization params as JSON for easy reading
        norm_path = output_dir / f"{self.glb_path.stem}_normalization.json"
        with open(norm_path, 'w') as f:
            json.dump({
                'center': self.normalization['center'].tolist(),
                'scale': float(self.normalization['scale'])
            }, f, indent=2)
        
        print(f"Saved processed data to {output_path}")
        print(f"Saved normalization params to {norm_path}")
        
        return output_path


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python vgg_t_loader.py <path_to_glb> [downsample_factor]")
        sys.exit(1)
    
    glb_path = sys.argv[1]
    downsample = float(sys.argv[2]) if len(sys.argv) > 2 else None
    
    # Load data
    loader = VGGTDataLoader(glb_path, downsample_factor=downsample)
    
    # Print info
    print(f"\nLoaded VGG-T data:")
    print(f"  Original points: {len(loader.data['positions']):,}")
    print(f"  Has colors: {'colors' in loader.data}")
    
    # Get normalized positions
    normalized = loader.get_normalized_positions()
    print(f"\nNormalized to [-1, 1]:")
    print(f"  Min: {normalized.min(axis=0)}")
    print(f"  Max: {normalized.max(axis=0)}")
    
    # Get a training batch
    batch = loader.get_training_batch(batch_size=10000)
    print(f"\nTraining batch:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")
    
    # Save processed data
    loader.save_processed_data()