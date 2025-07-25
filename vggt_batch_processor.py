#!/usr/bin/env python3
"""
VGGT Batch Processor - Process all images together for proper multi-view reconstruction
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
except ImportError:
    print("Error: VGGT not found. Please install VGGT first.")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def pose_encoding_to_matrix(pose_enc: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Convert VGGT pose encoding to extrinsic and intrinsic matrices"""
    # pose_enc shape can be [9], [N, 9], or [1, N, 9]
    if pose_enc.ndim == 3 and pose_enc.shape[0] == 1:
        pose_enc = pose_enc[0]  # Remove batch dimension: [1, N, 9] -> [N, 9]
    elif pose_enc.ndim == 1:
        pose_enc = pose_enc[np.newaxis, :]  # [9] -> [1, 9]
    
    num_poses = pose_enc.shape[0]
    extrinsics = []
    intrinsics = []
    
    for i in range(num_poses):
        pose = pose_enc[i]  # Get individual pose [9]
        
        # Extract rotation (first 6) and translation (last 3)
        rot_6d = pose[:6]
        translation = pose[6:9]
        
        # Convert 6D rotation to 3x3 matrix
        # Based on "On the Continuity of Rotation Representations in Neural Networks"
        a1 = rot_6d[:3]
        a2 = rot_6d[3:6]
        
        # Normalize a1
        b1 = a1 / (np.linalg.norm(a1) + 1e-8)
        
        # Make a2 orthogonal to b1
        b2 = a2 - np.dot(b1, a2) * b1
        b2 = b2 / (np.linalg.norm(b2) + 1e-8)
        
        # Third axis via cross product
        b3 = np.cross(b1, b2)
        
        # Stack to create rotation matrix
        rotation = np.stack([b1, b2, b3], axis=1)  # 3x3
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        
        extrinsics.append(transform)
        
        # Estimate intrinsics (VGGT doesn't provide explicit intrinsics)
        # Using default values based on 518x518 output
        focal_length = 518 * 1.2  # Reasonable estimate
        intrinsic = np.array([
            [focal_length, 0, 259],  # 518/2 = 259
            [0, focal_length, 259],
            [0, 0, 1]
        ])
        intrinsics.append(intrinsic)
    
    return extrinsics, intrinsics


def process_all_images_together(
    image_paths: List[Path],
    model: VGGT,
    device: str,
    dtype: torch.dtype
) -> Dict:
    """Process all images in a single batch for proper multi-view geometry"""
    
    logger.info(f"Loading and preprocessing {len(image_paths)} images...")
    
    # Load all images at once
    image_paths_str = [str(p) for p in image_paths]
    images = load_and_preprocess_images(image_paths_str).to(device)
    
    logger.info(f"Images tensor shape: {images.shape}")
    
    # Run VGGT inference on all images together
    with torch.no_grad():
        with torch.amp.autocast(device, dtype=dtype):
            predictions = model(images)
    
    # Log what we got
    logger.info("VGGT outputs:")
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    # Extract results
    results = {
        'depth_maps': [],
        'world_points': [],
        'extrinsics': [],
        'intrinsics': [],
        'images': images.cpu(),
        'pose_encodings': []
    }
    
    # Process pose encodings
    if 'pose_enc' in predictions:
        pose_enc = predictions['pose_enc'].cpu().numpy()
        logger.info(f"Pose encoding shape: {pose_enc.shape}")
        
        # Convert to matrices
        extrinsics, intrinsics = pose_encoding_to_matrix(pose_enc)
        results['extrinsics'] = extrinsics
        results['intrinsics'] = intrinsics
        results['pose_encodings'] = pose_enc
    
    # Process depth maps
    if 'depth' in predictions:
        depth = predictions['depth'].cpu().numpy()
        logger.info(f"Original depth shape: {depth.shape}")
        
        # Handle shape [1, 69, 518, 518, 1]
        if depth.shape[0] == 1:
            depth = depth[0]  # Remove batch dim -> [69, 518, 518, 1]
        if depth.shape[-1] == 1:
            depth = depth[..., 0]  # Remove last dim -> [69, 518, 518]
        
        # Now depth should be [69, 518, 518]
        for i in range(depth.shape[0]):
            results['depth_maps'].append(depth[i])
    
    # Process world points
    if 'world_points' in predictions:
        points = predictions['world_points'].cpu().numpy()
        logger.info(f"Original world_points shape: {points.shape}")
        
        # Handle shape [1, 69, 518, 518, 3]
        if points.shape[0] == 1:
            points = points[0]  # Remove batch dim -> [69, 518, 518, 3]
        
        for i in range(points.shape[0]):
            results['world_points'].append(points[i])
    
    return results


def save_results(results: Dict, image_paths: List[Path], output_dir: Path):
    """Save all results in Neuralangelo format"""
    # Create directories
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    (output_dir / "depth").mkdir(parents=True, exist_ok=True)
    
    # Save processed images
    logger.info("Saving processed images...")
    images_tensor = results['images']
    
    # Handle image tensor shape
    if images_tensor.shape[0] == 1:
        images_tensor = images_tensor[0]  # Remove batch dim if present
    
    for idx in range(images_tensor.shape[0]):
        img_array = images_tensor[idx].permute(1, 2, 0).numpy()
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        img.save(output_dir / "images" / f"frame_{idx:06d}.png")
        
        # Copy mask if it exists
        orig_path = image_paths[idx]
        mask_path = orig_path.parent.parent / "masks" / f"{orig_path.stem}.png"
        if mask_path.exists():
            mask = Image.open(mask_path).convert("L")
            mask_resized = mask.resize((518, 518), Image.NEAREST)
            mask_resized.save(output_dir / "masks" / f"frame_{idx:06d}.png")
    
    # Save depth maps
    logger.info("Saving depth maps...")
    for idx, depth in enumerate(results['depth_maps']):
        # Save raw depth
        np.save(output_dir / "depth" / f"frame_{idx:06d}_depth.npy", depth)
        
        # Save visualization
        depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        depth_vis = (depth_vis * 255).astype(np.uint8)
        Image.fromarray(depth_vis).save(output_dir / "depth" / f"frame_{idx:06d}_depth.png")
    
    # Save camera parameters
    logger.info("Saving camera parameters...")
    transforms = {
        "camera_model": "PINHOLE",
        "frames": []
    }
    
    for idx in range(len(image_paths)):
        frame_data = {
            "file_path": f"images/frame_{idx:06d}.png",
            "transform_matrix": results['extrinsics'][idx].tolist() if idx < len(results['extrinsics']) else None,
            "intrinsics": {
                "fx": float(results['intrinsics'][idx][0, 0]),
                "fy": float(results['intrinsics'][idx][1, 1]),
                "cx": float(results['intrinsics'][idx][0, 2]),
                "cy": float(results['intrinsics'][idx][1, 2])
            } if idx < len(results['intrinsics']) else None
        }
        transforms["frames"].append(frame_data)
    
    with open(output_dir / "transforms.json", 'w') as f:
        json.dump(transforms, f, indent=2)
    
    # Verify all frames have poses
    null_count = sum(1 for frame in transforms["frames"] if frame["transform_matrix"] is None)
    logger.info(f"Frames with camera poses: {len(transforms['frames']) - null_count}/{len(transforms['frames'])}")
    
    if null_count > 0:
        logger.warning(f"{null_count} frames are missing camera poses!")
    else:
        logger.info("✅ All frames have camera poses!")
    
    # Save metadata
    metadata = {
        "num_frames": len(image_paths),
        "processing_mode": "batch_multiview",
        "pose_encodings_shape": list(results['pose_encodings'].shape) if len(results['pose_encodings']) > 0 else None,
        "all_poses_estimated": null_count == 0
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Process all images together through VGGT")
    parser.add_argument("images", type=Path, help="Input images directory")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--metadata", type=Path, help="Path to metadata.json from prep_crop.py")
    
    args = parser.parse_args()
    
    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Get image paths
    image_paths = sorted(args.images.glob("*.png")) + sorted(args.images.glob("*.jpg"))
    logger.info(f"Found {len(image_paths)} images")
    
    if not image_paths:
        logger.error(f"No images found in {args.images}")
        return
    
    # Load model
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    logger.info(f"Loading VGGT model on {args.device} with dtype {dtype}")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(args.device)
    model.eval()
    
    # Process all images together
    results = process_all_images_together(image_paths, model, args.device, dtype)
    
    # Save results
    save_results(results, image_paths, args.output_dir)
    
    logger.info(f"\n✅ Batch processing complete!")
    logger.info(f"Output: {args.output_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Verify all frames have camera poses in transforms.json")
    logger.info("2. Update Neuralangelo config to use this directory")
    logger.info("3. Run Neuralangelo training")


if __name__ == "__main__":
    main()
