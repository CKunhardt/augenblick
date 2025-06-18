# Simple VGGT Large Dataset Processing
# Just run VGGT like the demo, but handle larger datasets with simple batching if needed
# No correlation analysis, no clustering - just direct VGGT processing

import os
import torch
import numpy as np
import sys
import glob
import time
import gc
from typing import Dict, List
import math

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"

def run_vggt_on_large_dataset(target_dir: str) -> Dict:
    """
    Run VGGT on large dataset - exactly like the demo but with optional batching
    """
    print(f"Processing images from {target_dir}")
    
    # Load model
    print("Loading VGGT model...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    
    # Find images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(target_dir, "images", ext)))
        image_paths.extend(glob.glob(os.path.join(target_dir, "images", ext.upper())))
    
    image_paths = sorted(image_paths)
    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        raise ValueError("No images found")
    
    # Try to process all images at once first (like the demo)
    try:
        print("Attempting to process all images at once...")
        predictions = _process_batch(model, image_paths)
        print("✓ Successfully processed all images in single batch!")
        return predictions
        
    except torch.cuda.OutOfMemoryError:
        print("✗ GPU OOM - will try batching...")
        torch.cuda.empty_cache()
        gc.collect()
        
        # Fall back to batching
        return _process_with_batching(model, image_paths)


def _process_batch(model, image_paths: List[str]) -> Dict:
    """Process a batch of images - exactly like demo_gradio.py"""
    
    # Load and preprocess images (same as demo)
    images = load_and_preprocess_images(image_paths).to(device)
    print(f"Preprocessed images shape: {images.shape}")
    
    # Run inference (same as demo)
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    
    # Convert pose encoding (same as demo)
    print("Converting pose encoding...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    
    # Convert tensors to numpy (same as demo)
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
    
    # Generate world points from depth map (same as demo)
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points
    
    # Add image paths for reference
    predictions["image_paths"] = image_paths
    
    # Clean up
    torch.cuda.empty_cache()
    
    return predictions


def _process_with_batching(model, image_paths: List[str]) -> Dict:
    """Process images in batches and merge results"""
    
    # Start with reasonable batch size and reduce if needed
    batch_size = 64
    
    while batch_size >= 1:
        try:
            print(f"Trying batch size: {batch_size}")
            return _process_batches(model, image_paths, batch_size)
        except torch.cuda.OutOfMemoryError:
            print(f"Batch size {batch_size} too large, trying smaller...")
            batch_size = batch_size // 2
            torch.cuda.empty_cache()
            gc.collect()
    
    raise RuntimeError("Could not process even single images - GPU memory insufficient")


def _process_batches(model, image_paths: List[str], batch_size: int) -> Dict:
    """Process images in fixed-size batches"""
    
    num_batches = math.ceil(len(image_paths) / batch_size)
    print(f"Processing {len(image_paths)} images in {num_batches} batches of size {batch_size}")
    
    all_predictions = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_paths))
        batch_paths = image_paths[start_idx:end_idx]
        
        print(f"Processing batch {batch_idx + 1}/{num_batches}: images {start_idx+1}-{end_idx}")
        
        # Process batch
        batch_predictions = _process_batch(model, batch_paths)
        all_predictions.append(batch_predictions)
        
        # Clean up between batches
        torch.cuda.empty_cache()
        gc.collect()
    
    # Merge results
    print("Merging batch results...")
    return _merge_predictions(all_predictions)


def _merge_predictions(predictions_list: List[Dict]) -> Dict:
    """Merge predictions from multiple batches"""
    
    if len(predictions_list) == 1:
        return predictions_list[0]
    
    print(f"Merging {len(predictions_list)} batches...")
    merged = {}
    
    # Arrays to concatenate along first dimension
    array_keys = [
        'depth', 'depth_conf', 'world_points', 'world_points_conf', 
        'world_points_from_depth', 'extrinsic', 'intrinsic', 'pose_enc'
    ]
    
    for key in array_keys:
        if key in predictions_list[0]:
            arrays = [pred[key] for pred in predictions_list if key in pred]
            if arrays:
                merged[key] = np.concatenate(arrays, axis=0)
    
    # Handle images (might be tensor or array)
    if 'images' in predictions_list[0]:
        image_arrays = []
        for pred in predictions_list:
            if 'images' in pred:
                img_data = pred['images']
                if isinstance(img_data, torch.Tensor):
                    img_data = img_data.cpu().numpy()
                image_arrays.append(img_data)
        
        if image_arrays:
            merged['images'] = np.concatenate(image_arrays, axis=0)
    
    # Merge image paths
    merged['image_paths'] = []
    for pred in predictions_list:
        merged['image_paths'].extend(pred.get('image_paths', []))
    
    # Add metadata
    merged['num_batches'] = len(predictions_list)
    merged['batch_sizes'] = [len(pred.get('image_paths', [])) for pred in predictions_list]
    
    print(f"Merged {len(predictions_list)} batches into single prediction")
    print(f"Total images: {len(merged.get('image_paths', []))}")
    
    return merged


def main():
    """Main function - process dataset and save results"""
    target_dir = "C:\\repos\\gatech\\photogrammetry\\south-building"
    
    try:
        print("="*60)
        print("SIMPLE VGGT LARGE DATASET PROCESSING")
        print("="*60)
        
        start_time = time.time()
        predictions = run_vggt_on_large_dataset(target_dir)
        total_time = time.time() - start_time
        
        print(f"\n=== PROCESSING COMPLETE ===")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Images processed: {len(predictions.get('image_paths', []))}")
        
        if 'batch_sizes' in predictions:
            print(f"Processed in {predictions['num_batches']} batches: {predictions['batch_sizes']}")
        else:
            print("Processed in single batch")
        
        # Save GLB
        glb_path = os.path.join(target_dir, f"simple_vggt_reconstruction_{int(time.time())}.glb")
        print(f"Saving GLB to {glb_path}")
        
        scene = predictions_to_glb(predictions, conf_thres=30.0, target_dir=target_dir)
        scene.export(glb_path)
        
        # Statistics
        world_points = predictions.get("world_points_from_depth")
        if world_points is not None:
            points_flat = world_points.reshape(-1, 3)
            valid_mask = ~np.any(np.isnan(points_flat) | np.isinf(points_flat), axis=1)
            valid_points = points_flat[valid_mask]
            
            print(f"\n=== RESULTS ===")
            print(f"Generated {len(valid_points):,} valid 3D points")
            print(f"Point cloud dimensions: {np.ptp(valid_points, axis=0)}")
            print(f"Processing rate: {len(predictions['image_paths']) / total_time:.1f} images/second")
        
        print(f"\n✓ SUCCESS: Results saved to {glb_path}")
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()