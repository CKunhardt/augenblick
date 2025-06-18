# Coherent Multi-Batch Scene Reconstruction
# Solving the spatial alignment problem for large datasets

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
from scipy.optimize import minimize
import json
import pickle

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"

class CoherentSceneReconstruction:
    """
    Reconstructs large scenes by ensuring spatial coherence across batches
    """
    
    def __init__(self, max_batch_size: int = 16, max_resolution: int = 320):
        self.max_batch_size = max_batch_size
        self.max_resolution = max_resolution
        self.device = device
        self.model = None
        self.anchor_images = None  # Key images that appear in multiple batches
        
    def load_model(self):
        """Load VGGT model"""
        if self.model is not None:
            return
            
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
        
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")

    def create_anchor_strategy_batches(self, image_paths: List[str]) -> Tuple[List[List[str]], List[str]]:
        """
        Create batches with anchor images that provide spatial continuity
        
        Strategy:
        1. Select anchor images evenly distributed across the sequence
        2. Each batch includes some anchor images + local images
        3. Anchors provide registration points between batches
        """
        
        # Select anchor images (every Nth image)
        anchor_stride = max(1, len(image_paths) // (self.max_batch_size // 2))
        anchor_images = image_paths[::anchor_stride]
        
        # Ensure we don't have too many anchors
        if len(anchor_images) > self.max_batch_size // 2:
            anchor_images = anchor_images[:self.max_batch_size // 2]
        
        print(f"Selected {len(anchor_images)} anchor images")
        
        # Create batches
        batches = []
        remaining_images = [img for img in image_paths if img not in anchor_images]
        
        # First batch: all anchors + some local images
        batch_size_local = self.max_batch_size - len(anchor_images)
        if batch_size_local > 0:
            first_batch = anchor_images + remaining_images[:batch_size_local]
            batches.append(first_batch)
            remaining_images = remaining_images[batch_size_local:]
        else:
            batches.append(anchor_images)
        
        # Subsequent batches: subset of anchors + local images
        anchors_per_batch = min(4, len(anchor_images))  # 4 anchors per batch
        
        while remaining_images:
            # Select subset of anchors for this batch
            batch_anchors = anchor_images[:anchors_per_batch]
            batch_size_local = self.max_batch_size - len(batch_anchors)
            
            local_images = remaining_images[:batch_size_local]
            batch = batch_anchors + local_images
            
            batches.append(batch)
            remaining_images = remaining_images[batch_size_local:]
        
        print(f"Created {len(batches)} batches with anchor strategy")
        return batches, anchor_images

    def create_progressive_batches(self, image_paths: List[str]) -> List[List[str]]:
        """
        Create batches for progressive reconstruction:
        1. Start with well-distributed key images
        2. Add detail images in subsequent passes
        """
        
        # Phase 1: Key images (widely distributed)
        key_stride = max(1, len(image_paths) // self.max_batch_size)
        key_images = image_paths[::key_stride][:self.max_batch_size]
        
        batches = [key_images]
        
        # Phase 2: Fill in gaps with detail images
        used_images = set(key_images)
        remaining_images = [img for img in image_paths if img not in used_images]
        
        # Group remaining images by proximity to key images
        detail_batches = []
        current_batch = []
        
        for img in remaining_images:
            current_batch.append(img)
            
            if len(current_batch) >= self.max_batch_size - 4:  # Leave room for key images
                # Add some key images for registration
                batch_with_keys = key_images[:4] + current_batch
                detail_batches.append(batch_with_keys)
                current_batch = []
        
        # Add final batch if any images remain
        if current_batch:
            batch_with_keys = key_images[:4] + current_batch
            detail_batches.append(batch_with_keys)
        
        batches.extend(detail_batches)
        
        print(f"Created progressive batches: 1 key batch + {len(detail_batches)} detail batches")
        return batches

    def process_batch_with_metadata(self, image_paths: List[str], batch_idx: int) -> Dict:
        """Process batch and return with spatial metadata"""
        
        # Preprocess images
        processed_paths = self.preprocess_image_paths(image_paths)
        
        try:
            # Load and process
            images = load_and_preprocess_images(processed_paths).to(self.device)
            
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
            
            # Add metadata
            predictions["image_paths"] = image_paths
            predictions["batch_idx"] = batch_idx
            predictions["batch_centroid"] = np.mean(world_points.reshape(-1, 3), axis=0)
            predictions["batch_scale"] = np.std(world_points.reshape(-1, 3))
            
            return predictions
            
        finally:
            # Cleanup
            for path in processed_paths:
                if path.startswith("temp_resized_") and os.path.exists(path):
                    os.remove(path)
            torch.cuda.empty_cache()
            gc.collect()

    def preprocess_image_paths(self, image_paths: List[str]) -> List[str]:
        """Resize images if needed"""
        processed_paths = []
        
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            h, w = img.shape[:2]
            max_dim = max(h, w)
            
            if max_dim > self.max_resolution:
                scale = self.max_resolution / max_dim
                new_w, new_h = int(w * scale), int(h * scale)
                img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                base_name = os.path.basename(img_path)
                temp_path = f"temp_resized_{base_name}"
                cv2.imwrite(temp_path, img_resized)
                processed_paths.append(temp_path)
            else:
                processed_paths.append(img_path)
        
        return processed_paths

    def align_batches_with_anchors(self, batch_predictions: List[Dict], anchor_images: List[str]) -> List[Dict]:
        """
        Align batches using anchor images as registration points
        """
        
        print(f"Aligning {len(batch_predictions)} batches using {len(anchor_images)} anchor images...")
        
        if len(batch_predictions) <= 1:
            return batch_predictions
        
        # Use first batch as reference
        reference_batch = batch_predictions[0]
        aligned_batches = [reference_batch]
        
        for i, batch in enumerate(batch_predictions[1:], 1):
            print(f"Aligning batch {i+1} to reference...")
            
            # Find anchor images in both batches
            ref_anchors, batch_anchors = self.find_common_anchors(reference_batch, batch, anchor_images)
            
            if len(ref_anchors) >= 3:  # Need at least 3 points for rigid alignment
                # Compute alignment transformation
                transform = self.compute_alignment_transform(ref_anchors, batch_anchors)
                
                # Apply transformation to batch
                aligned_batch = self.apply_transform_to_batch(batch, transform)
                aligned_batches.append(aligned_batch)
                
                print(f"Batch {i+1} aligned using {len(ref_anchors)} anchor points")
            else:
                print(f"Warning: Only {len(ref_anchors)} anchors found for batch {i+1}, using centroid alignment")
                # Fallback to centroid alignment
                aligned_batch = self.align_by_centroid(reference_batch, batch)
                aligned_batches.append(aligned_batch)
        
        return aligned_batches

    def find_common_anchors(self, ref_batch: Dict, batch: Dict, anchor_images: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Find 3D points corresponding to anchor images in both batches"""
        
        ref_paths = ref_batch["image_paths"]
        batch_paths = batch["image_paths"]
        
        ref_anchors = []
        batch_anchors = []
        
        for anchor_img in anchor_images:
            # Find anchor in reference batch
            if anchor_img in ref_paths:
                ref_idx = ref_paths.index(anchor_img)
                ref_points = ref_batch["world_points_from_depth"][ref_idx]
                ref_centroid = np.mean(ref_points.reshape(-1, 3), axis=0)
                
                # Find same anchor in current batch
                if anchor_img in batch_paths:
                    batch_idx = batch_paths.index(anchor_img)
                    batch_points = batch["world_points_from_depth"][batch_idx]
                    batch_centroid = np.mean(batch_points.reshape(-1, 3), axis=0)
                    
                    ref_anchors.append(ref_centroid)
                    batch_anchors.append(batch_centroid)
        
        return np.array(ref_anchors), np.array(batch_anchors)

    def compute_alignment_transform(self, ref_points: np.ndarray, batch_points: np.ndarray) -> np.ndarray:
        """Compute rigid transformation (rotation + translation) between point sets"""
        
        # Compute centroids
        ref_centroid = np.mean(ref_points, axis=0)
        batch_centroid = np.mean(batch_points, axis=0)
        
        # Center the points
        ref_centered = ref_points - ref_centroid
        batch_centered = batch_points - batch_centroid
        
        # Compute rotation using SVD
        H = batch_centered.T @ ref_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = ref_centroid - R @ batch_centroid
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t
        
        return transform

    def apply_transform_to_batch(self, batch: Dict, transform: np.ndarray) -> Dict:
        """Apply transformation to all 3D points in a batch"""
        
        aligned_batch = batch.copy()
        
        # Transform world points
        world_points = batch["world_points_from_depth"]
        original_shape = world_points.shape
        
        # Reshape to (N, 3) for transformation
        points_flat = world_points.reshape(-1, 3)
        
        # Add homogeneous coordinate
        points_homo = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
        
        # Apply transformation
        points_transformed = (transform @ points_homo.T).T
        
        # Remove homogeneous coordinate and reshape
        points_transformed = points_transformed[:, :3].reshape(original_shape)
        
        aligned_batch["world_points_from_depth"] = points_transformed
        
        # Update batch centroid
        aligned_batch["batch_centroid"] = np.mean(points_transformed.reshape(-1, 3), axis=0)
        
        return aligned_batch

    def align_by_centroid(self, ref_batch: Dict, batch: Dict) -> Dict:
        """Fallback alignment using centroids only"""
        
        ref_centroid = ref_batch["batch_centroid"]
        batch_centroid = batch["batch_centroid"]
        
        translation = ref_centroid - batch_centroid
        
        # Create translation-only transform
        transform = np.eye(4)
        transform[:3, 3] = translation
        
        return self.apply_transform_to_batch(batch, transform)

    def merge_aligned_batches(self, aligned_batches: List[Dict]) -> Dict:
        """Merge spatially aligned batches into single prediction"""
        
        print(f"Merging {len(aligned_batches)} aligned batches...")
        
        merged = {}
        
        for key in aligned_batches[0].keys():
            if key in ['image_paths', 'batch_idx']:
                # Flatten lists
                merged[key] = []
                for batch in aligned_batches:
                    if isinstance(batch[key], list):
                        merged[key].extend(batch[key])
                    else:
                        merged[key].append(batch[key])
                        
            elif key in ['batch_centroid', 'batch_scale']:
                # Skip batch-specific metadata
                continue
                
            elif isinstance(aligned_batches[0][key], np.ndarray):
                # Concatenate arrays
                arrays = [batch[key] for batch in aligned_batches]
                merged[key] = np.concatenate(arrays, axis=0)
            else:
                # Take first value
                merged[key] = aligned_batches[0][key]
        
        print(f"Merged scene: {merged['world_points_from_depth'].shape}")
        return merged

    def reconstruct_scene_progressive(self, target_dir: str) -> Dict:
        """
        Progressive reconstruction with proper spatial alignment
        """
        
        # Find images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(target_dir, "images", ext)))
            image_paths.extend(glob.glob(os.path.join(target_dir, "images", ext.upper())))
        
        image_paths = sorted(image_paths)
        print(f"Found {len(image_paths)} images")
        
        if self.model is None:
            self.load_model()
        
        # Create batches with anchor strategy
        batches, anchor_images = self.create_anchor_strategy_batches(image_paths)
        
        # Process each batch
        batch_predictions = []
        for i, batch_paths in enumerate(batches):
            try:
                print(f"\nProcessing batch {i+1}/{len(batches)}: {len(batch_paths)} images")
                pred = self.process_batch_with_metadata(batch_paths, i)
                batch_predictions.append(pred)
            except Exception as e:
                print(f"Failed to process batch {i}: {e}")
                continue
        
        if not batch_predictions:
            raise RuntimeError("All batches failed")
        
        # Align batches using anchors
        aligned_batches = self.align_batches_with_anchors(batch_predictions, anchor_images)
        
        # Merge aligned batches
        final_prediction = self.merge_aligned_batches(aligned_batches)
        
        return final_prediction

def main():
    """Test the coherent reconstruction pipeline"""
    
    target_dir = "C:\\repos\\gatech\\photogrammetry\\south-building"
    
    # Create reconstructor
    reconstructor = CoherentSceneReconstruction(
        max_batch_size=16,
        max_resolution=320
    )
    
    try:
        print("Starting coherent scene reconstruction...")
        predictions = reconstructor.reconstruct_scene_progressive(target_dir)
        
        # Save results
        glb_path = os.path.join(target_dir, f"coherent_reconstruction_{str(time.time())}.glb")
        print(f"Saving GLB to {glb_path}")
        scene = predictions_to_glb(predictions, conf_thres=50.0, target_dir=target_dir)
        scene.export(glb_path)
        
        # Visualize result
        world_points = predictions["world_points_from_depth"].reshape(-1, 3)
        valid_mask = ~np.any(np.isnan(world_points) | np.isinf(world_points), axis=1)
        valid_points = world_points[valid_mask]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_points)
        
        print(f"\n--- Results ---")
        print(f"Processed {len(predictions.get('image_paths', []))} images")
        print(f"Generated {len(valid_points)} valid 3D points")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()