#!/usr/bin/env python3
"""
Simple VGGT Pipeline - A clean implementation for processing images with VGGT
"""

import os
import sys
import glob
import torch
import numpy as np
import argparse
import time
import pickle
from pathlib import Path

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from visual_util import predictions_to_glb


def load_model():
    """Load and initialize the VGGT model"""
    print("Loading VGGT model...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    
    return model, device


def process_single_image(model, image_path, device):
    """Process a single image through VGGT"""
    print(f"Processing: {os.path.basename(image_path)}")
    
    # Load and preprocess the image
    images = load_and_preprocess_images([image_path]).to(device)
    print(f"  Image tensor shape: {images.shape}")
    
    # Run inference
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            predictions = model(images)
    
    # Debug output shapes
    print("  Model outputs:")
    for key, value in predictions.items():
        print(f"    {key}: {value.shape}")
    
    # Convert pose encoding to camera matrices
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], 
        images.shape[-2:]
    )
    
    # Move everything to CPU and convert to numpy
    predictions_np = {}
    for key, value in predictions.items():
        predictions_np[key] = value.cpu().numpy()
    
    predictions_np["extrinsic"] = extrinsic.cpu().numpy()
    predictions_np["intrinsic"] = intrinsic.cpu().numpy()
    
    # Store the original image
    predictions_np["image"] = images.cpu().numpy()
    
    return predictions_np


def process_image_batch(model, image_paths, device, batch_size=4):
    """Process multiple images in batches"""
    all_results = []
    
    num_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_paths))
        batch_paths = image_paths[start_idx:end_idx]
        
        print(f"\nBatch {batch_idx + 1}/{num_batches} (images {start_idx + 1}-{end_idx})")
        
        # Process each image in the batch individually to avoid shape issues
        for img_path in batch_paths:
            result = process_single_image(model, img_path, device)
            all_results.append(result)
        
        # Clear GPU memory after each batch
        torch.cuda.empty_cache()
    
    return all_results


def combine_predictions(results_list):
    """Combine individual predictions into batched format"""
    if not results_list:
        return None
    
    combined = {}
    
    # Get all keys from first result
    keys = results_list[0].keys()
    
    for key in keys:
        # Stack all values for this key
        values = [r[key] for r in results_list]
        combined[key] = np.concatenate(values, axis=0)
        print(f"Combined {key}: {combined[key].shape}")
    
    return combined


def compute_world_points(predictions):
    """Compute world points from depth predictions"""
    print("\nComputing world points from depth...")
    
    depth = predictions["depth"]
    extrinsic = predictions["extrinsic"]
    intrinsic = predictions["intrinsic"]
    
    print(f"Depth shape: {depth.shape}")
    print(f"Extrinsic shape: {extrinsic.shape}")
    print(f"Intrinsic shape: {intrinsic.shape}")
    
    # The model outputs have an extra singleton dimension at position 1
    # Squeeze it out for all relevant tensors
    if depth.shape[1] == 1:
        depth = depth.squeeze(1)  # Remove dim 1: (20, 1, H, W, 1) -> (20, H, W, 1)
        print(f"Squeezed depth shape: {depth.shape}")
    
    if extrinsic.shape[1] == 1:
        extrinsic = extrinsic.squeeze(1)  # Remove dim 1: (20, 1, 3, 4) -> (20, 3, 4)
        print(f"Squeezed extrinsic shape: {extrinsic.shape}")
    
    if intrinsic.shape[1] == 1:
        intrinsic = intrinsic.squeeze(1)  # Remove dim 1: (20, 1, 3, 3) -> (20, 3, 3)
        print(f"Squeezed intrinsic shape: {intrinsic.shape}")
    
    # Now handle depth to get to (S, H, W) format
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)  # Remove last dim: (20, H, W, 1) -> (20, H, W)
        print(f"Final depth shape: {depth.shape}")
    
    # Compute world points
    world_points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
    predictions["world_points_from_depth"] = world_points
    
    print(f"World points shape: {world_points.shape}")
    
    return predictions


def save_for_neus2(predictions, output_path, image_dir):
    """Save predictions in NeuS2 format"""
    print(f"\nSaving predictions for NeuS2 conversion...")
    
    # Get images - handle the extra dimension from model output
    images = predictions.get("image", predictions.get("images", None))
    if images is not None and images.shape[1] == 1:
        images = images.squeeze(1)  # Remove singleton dimension
    
    neus2_data = {
        "world_points": predictions.get("world_points", None),
        "world_points_from_depth": predictions.get("world_points_from_depth", None),
        "depth": predictions["depth"],
        "extrinsic": predictions["extrinsic"],
        "intrinsic": predictions["intrinsic"],
        "metadata": {
            "source": "VGGT",
            "timestamp": time.time(),
            "num_frames": predictions["extrinsic"].shape[0],
            "image_dir": image_dir
        }
    }
    
    # Add images if available
    if images is not None:
        # Convert from CHW to HWC format and to uint8
        if images.ndim == 4 and images.shape[1] == 3:  # (N, C, H, W)
            images = images.transpose(0, 2, 3, 1)  # -> (N, H, W, C)
        # Denormalize if needed
        if images.max() <= 1.0:
            images = (images * 255).astype(np.uint8)
        neus2_data["images"] = images
    
    # Add confidence if available
    if "world_points_conf" in predictions:
        conf = predictions["world_points_conf"]
        if conf.shape[1] == 1:
            conf = conf.squeeze(1)
        neus2_data["world_points_conf"] = conf
    elif "depth_conf" in predictions:
        conf = predictions["depth_conf"]
        if conf.shape[1] == 1:
            conf = conf.squeeze(1)
        neus2_data["depth_conf"] = conf
    
    # Clean up dimensions for all data
    for key in ["depth", "extrinsic", "intrinsic", "world_points", "world_points_from_depth"]:
        if key in neus2_data and neus2_data[key] is not None:
            data = neus2_data[key]
            # Remove singleton dimension at position 1 if present
            if isinstance(data, np.ndarray) and data.ndim > 1 and data.shape[1] == 1:
                neus2_data[key] = data.squeeze(1)
    
    with open(output_path, 'wb') as f:
        pickle.dump(neus2_data, f)
    
    print(f"Saved to: {output_path}")
    
    # Print summary
    print("\nData summary:")
    for key, value in neus2_data.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")
        elif key == "metadata":
            print(f"  {key}: {value}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Simple VGGT Pipeline")
    parser.add_argument("input_dir", help="Directory containing images")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--save_neus2", action="store_true", help="Save in NeuS2 format")
    parser.add_argument("--save_glb", action="store_true", help="Save GLB file")
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    image_dir = input_dir / "images" if (input_dir / "images").exists() else input_dir
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Find images
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    for pattern in image_patterns:
        image_paths.extend(sorted(image_dir.glob(pattern)))
    
    if not image_paths:
        print(f"No images found in {image_dir}")
        return 1
    
    print(f"Found {len(image_paths)} images")
    
    # Load model
    model, device = load_model()
    
    # Process images
    print("\nProcessing images...")
    results = process_image_batch(model, image_paths, device, args.batch_size)
    
    # Combine results
    print("\nCombining predictions...")
    predictions = combine_predictions(results)
    
    # Compute world points
    predictions = compute_world_points(predictions)
    
    # Save outputs
    if args.save_neus2:
        neus2_path = output_dir / f"vggt_predictions_{int(time.time())}.pkl"
        save_for_neus2(predictions, neus2_path, str(image_dir))
        
        print("\n" + "="*60)
        print("NeuS2 Conversion Command:")
        print(f"python vggt_to_neus2_converter.py {neus2_path} ./neus2_data")
        print("="*60)
    
    if args.save_glb:
        glb_path = output_dir / f"vggt_output_{int(time.time())}.glb"
        print(f"\nGenerating GLB file...")
        scene = predictions_to_glb(predictions, target_dir=str(input_dir))
        scene.export(str(glb_path))
        print(f"Saved GLB to: {glb_path}")
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
