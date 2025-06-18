# Large-Scale VGGT Pipeline for 200+ Images
# Author: Clinton T. Kunhardt

import os
import cv2
import torch
import numpy as np
import sys
import glob
import time
import open3d as o3d
from typing import Dict, List, Optional, Tuple
import gc
from scipy.spatial.distance import cdist
import json
import pickle

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"

class LargeScaleVGGTPipeline:
    """
    VGGT pipeline optimized for processing 200+ images by using:
    1. Smart image selection/clustering
    2. Overlapping batches with alignment
    3. Progressive reconstruction
    """
    def __init__(self, max_batch_size: int = 16, max_resolution: int = 320, overlap_ratio: float = 0.3):
        self.max_batch_size = max_batch_size
        self.max_resolution = max_resolution
        self.overlap_ratio = overlap_ratio  # How much overlap between batches
        self.device = device
        self.model = None
        
    def load_model(self):
        """Load VGGT model with memory optimization"""
        print("Loading VGGT model...")
        start_time = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        
        state_dict = torch.hub.load_state_dict_from_url(_URL, map_location='cpu')
        self.model.load_state_dict(state_dict)
        
        self.model.eval()
        self.model = self.model.to(self.device)
        
        end_time = time.time()
        print(f"Model loaded in {end_time - start_time:.2f} seconds")
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU memory after model loading: {memory_allocated:.2f} GB")
    
    def select_key_images(self, image_paths: List[str], target_count: int = 50) -> List[str]:
        """
        Select key images from a large set using visual similarity and spatial distribution
        """
        if len(image_paths) <= target_count:
            return image_paths
            
        print(f"Selecting {target_count} key images from {len(image_paths)} total images...")
        
        # Method 1: Evenly spaced selection (simple but effective)
        indices = np.linspace(0, len(image_paths) - 1, target_count, dtype=int)
        selected_paths = [image_paths[i] for i in indices]
        
        # TODO: Could implement more sophisticated selection based on:
        # - Image similarity (avoid near-duplicates)
        # - Scene coverage (ensure good spatial distribution)
        # - Image quality metrics (blur, exposure, etc.)
        
        print(f"Selected images: {[os.path.basename(p) for p in selected_paths[:5]]}...")
        return selected_paths
    
    def create_overlapping_batches(self, image_paths: List[str]) -> List[List[str]]:
        """
        Create overlapping batches to maintain spatial coherence
        """
        batches = []
        overlap_size = int(self.max_batch_size * self.overlap_ratio)
        step_size = self.max_batch_size - overlap_size
        
        for i in range(0, len(image_paths), step_size):
            batch_end = min(i + self.max_batch_size, len(image_paths))
            batch = image_paths[i:batch_end]
            
            if len(batch) >= 3:  # Need minimum 3 images for reconstruction
                batches.append(batch)
            
            if batch_end >= len(image_paths):
                break
        
        print(f"Created {len(batches)} overlapping batches with {overlap_size} image overlap")
        return batches
    
    def process_batch(self, image_paths: List[str], batch_idx: int) -> Dict:
        """Process a single batch of images"""
        print(f"\n--- Processing Batch {batch_idx + 1}: {len(image_paths)} images ---")
        
        # Preprocess images
        processed_paths = self.preprocess_image_paths(image_paths)
        
        try:
            # Load and preprocess
            images = load_and_preprocess_images(processed_paths).to(self.device)
            print(f"Loaded batch shape: {images.shape}")
            
            # Run inference
            with torch.no_grad():
                dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = self.model(images)
            
            # Process predictions
            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic
            
            # Convert to numpy
            for key in predictions.keys():
                if isinstance(predictions[key], torch.Tensor):
                    predictions[key] = predictions[key].cpu().numpy().squeeze(0)
            
            # Generate world points
            depth_map = predictions["depth"]
            world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
            predictions["world_points_from_depth"] = world_points
            
            # Store image paths for reference
            predictions["image_paths"] = image_paths
            predictions["batch_idx"] = batch_idx
            
            return predictions
            
        finally:
            # Cleanup
            for path in processed_paths:
                if path.startswith("temp_resized_") and os.path.exists(path):
                    os.remove(path)
            torch.cuda.empty_cache()
            gc.collect()
    
    def preprocess_image_paths(self, image_paths: List[str]) -> List[str]:
        """Resize images if needed and return processed paths"""
        processed_paths = []
        
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            h, w = img.shape[:2]
            max_dim = max(h, w)
            
            if max_dim > self.max_resolution:
                # Resize image
                scale = self.max_resolution / max_dim
                new_w, new_h = int(w * scale), int(h * scale)
                img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Save resized image temporarily
                base_name = os.path.basename(img_path)
                temp_path = f"temp_resized_{base_name}"
                cv2.imwrite(temp_path, img_resized)
                processed_paths.append(temp_path)
            else:
                processed_paths.append(img_path)
        
        return processed_paths
    
    def merge_batch_predictions(self, batch_predictions: List[Dict]) -> Dict:
        """
        Merge predictions from multiple overlapping batches
        """
        print(f"\nMerging {len(batch_predictions)} batch predictions...")
        
        # For now, simple concatenation - could implement more sophisticated alignment
        merged = {}
        
        for key in batch_predictions[0].keys():
            if key in ['image_paths', 'batch_idx']:
                # Flatten lists
                merged[key] = []
                for batch in batch_predictions:
                    if isinstance(batch[key], list):
                        merged[key].extend(batch[key])
                    else:
                        merged[key].append(batch[key])
            elif isinstance(batch_predictions[0][key], np.ndarray):
                # Concatenate arrays
                arrays = [batch[key] for batch in batch_predictions]
                merged[key] = np.concatenate(arrays, axis=0)
            else:
                # Take first value for other types
                merged[key] = batch_predictions[0][key]
        
        print(f"Merged predictions shape: {merged['world_points_from_depth'].shape}")
        return merged
    
    def process_large_dataset(self, target_dir: str, strategy: str = 'select_key') -> Dict:
        """
        Process large image datasets using different strategies
        
        Args:
            target_dir: Directory containing images
            strategy: 'select_key', 'overlapping_batches', or 'progressive'
        """
        print(f"Processing large dataset with strategy: {strategy}")
        
        # Find all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(target_dir, "images", ext)))
            image_paths.extend(glob.glob(os.path.join(target_dir, "images", ext.upper())))
        
        image_paths = sorted(image_paths)
        print(f"Found {len(image_paths)} total images")
        
        if len(image_paths) == 0:
            raise ValueError("No images found")
        
        # Load model
        if self.model is None:
            self.load_model()
        
        if strategy == 'select_key':
            return self._process_key_selection(image_paths)
        elif strategy == 'overlapping_batches':
            return self._process_overlapping_batches(image_paths)
        elif strategy == 'progressive':
            return self._process_progressive(image_paths)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _process_key_selection(self, image_paths: List[str]) -> Dict:
        """Process by selecting key representative images"""
        key_images = self.select_key_images(image_paths, target_count=self.max_batch_size)
        return self.process_batch(key_images, 0)
    
    def _process_overlapping_batches(self, image_paths: List[str]) -> Dict:
        """Process using overlapping batches"""
        batches = self.create_overlapping_batches(image_paths)
        batch_predictions = []
        
        for i, batch_paths in enumerate(batches):
            try:
                pred = self.process_batch(batch_paths, i)
                batch_predictions.append(pred)
            except Exception as e:
                print(f"Failed to process batch {i}: {e}")
                continue
        
        if not batch_predictions:
            raise RuntimeError("All batches failed to process")
        
        return self.merge_batch_predictions(batch_predictions)
    
    def _process_progressive(self, image_paths: List[str]) -> Dict:
        """Process using progressive refinement (coarse to fine)"""
        # Start with key images
        key_images = self.select_key_images(image_paths, target_count=self.max_batch_size // 2)
        base_prediction = self.process_batch(key_images, 0)
        
        # TODO: Add additional images in subsequent passes
        # This would require more sophisticated alignment and merging
        
        return base_prediction

class ProgressiveMeshPipeline:
    """Generate meshes from large-scale predictions"""
    
    def __init__(self, voxel_size: float = 0.01):
        self.voxel_size = voxel_size
        
    def generate_progressive_mesh(self, predictions: Dict) -> o3d.geometry.TriangleMesh:
        """Generate mesh with progressive detail levels"""
        
        # Start with coarse mesh
        print("Generating coarse mesh...")
        coarse_mesh = self._generate_coarse_mesh(predictions)
        
        # Refine in high-detail areas
        print("Refining high-detail areas...")
        refined_mesh = self._refine_mesh_details(coarse_mesh, predictions)
        
        return refined_mesh
    
    def _generate_coarse_mesh(self, predictions: Dict) -> o3d.geometry.TriangleMesh:
        """Generate initial coarse mesh using downsampled points"""
        world_points = predictions["world_points_from_depth"]
        images = predictions["images"]
        
        # Handle image format
        if images.ndim == 4 and images.shape[1] == 3:
            images = np.transpose(images, (0, 2, 3, 1))
        
        # Reshape and downsample
        points = world_points.reshape(-1, 3)
        colors = images.reshape(-1, 3)
        
        # Filter invalid points
        valid_mask = ~np.any(np.isnan(points) | np.isinf(points), axis=1)
        points = points[valid_mask]
        colors = colors[valid_mask]
        
        # Aggressive downsampling for coarse mesh
        if len(points) > 100000:
            indices = np.random.choice(len(points), 100000, replace=False)
            points = points[indices]
            colors = colors[indices]
        
        # Create point cloud and mesh
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 10, max_nn=30)
        )
        
        # Poisson reconstruction with low depth for speed
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8, width=0, scale=1.1, linear_fit=False
        )
        
        return mesh
    
    def _refine_mesh_details(self, coarse_mesh: o3d.geometry.TriangleMesh, predictions: Dict) -> o3d.geometry.TriangleMesh:
        """Refine mesh in areas with high point density"""
        # For now, just return the coarse mesh
        # TODO: Implement detail refinement
        return coarse_mesh

def main():
    """Main function with multiple processing strategies"""
    target_dir = "C:\\repos\\gatech\\photogrammetry\\south-building"
    
    # Strategy options for handling 200+ images:
    strategies = {
        'select_key': "Select ~16 most representative images (fastest, good coverage)",
        'overlapping_batches': "Process in overlapping batches (slower, more complete)",
        'progressive': "Progressive refinement (experimental)"
    }
    
    print("Available strategies for large datasets:")
    for strategy, description in strategies.items():
        print(f"  {strategy}: {description}")
    
    # Choose strategy based on your needs
    chosen_strategy = 'overlapping_batches'  # Change this based on your preference
    
    # Configure pipeline for 12GB GPU
    pipeline = LargeScaleVGGTPipeline(
        max_batch_size=16,      # Safe batch size for 12GB GPU
        max_resolution=320,     # Conservative resolution
        overlap_ratio=0.3       # 30% overlap between batches
    )
    
    try:
        print(f"\nUsing strategy: {chosen_strategy}")
        predictions = pipeline.process_large_dataset(target_dir, strategy=chosen_strategy)
        
        # Save GLB
        glb_path = os.path.join(target_dir, f"predictions_{chosen_strategy}_{str(time.time())}.glb")
        print(f"Saving GLB to {glb_path}")
        scene = predictions_to_glb(predictions, conf_thres=50.0, target_dir=target_dir)
        scene.export(glb_path)
        
        # Generate progressive mesh
        mesh_pipeline = ProgressiveMeshPipeline(voxel_size=0.008)
        mesh = mesh_pipeline.generate_progressive_mesh(predictions)
        
        # Save mesh
        mesh_path = os.path.join(target_dir, f"progressive_mesh_{chosen_strategy}.ply")
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"Saved mesh to {mesh_path}")
        
        # Stats
        world_points = predictions["world_points_from_depth"].reshape(-1, 3)
        valid_mask = ~np.any(np.isnan(world_points) | np.isinf(world_points), axis=1)
        valid_points = world_points[valid_mask]
        
        print(f"\n--- Results ---")
        print(f"Processed {len(predictions.get('image_paths', []))} images")
        print(f"Generated {len(valid_points)} valid 3D points")
        print(f"Final mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()