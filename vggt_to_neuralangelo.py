#!/usr/bin/env python3
"""
VGGT to Neuralangelo Pipeline
Processes images through VGGT to generate camera poses and depth maps for Neuralangelo
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# VGGT imports
try:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
except ImportError:
    print("Error: VGGT not found. Please install VGGT first:")
    print("git clone https://github.com/facebookresearch/vggt.git")
    print("cd vggt && pip install -e .")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_vggt_model(device: str = "cuda", dtype: Optional[torch.dtype] = None) -> VGGT:
    """Load pretrained VGGT model"""
    if dtype is None:
        # Use bfloat16 on Ampere GPUs (Compute Capability 8.0+), otherwise float16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    
    logger.info(f"Loading VGGT model on {device} with dtype {dtype}")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    
    return model, dtype


def process_images_batch(
    image_paths: List[Path],
    model: VGGT,
    device: str,
    dtype: torch.dtype,
    batch_size: int = 8
) -> Dict:
    """Process images through VGGT in batches"""
    results = {
        'cameras': [],
        'depth_maps': [],
        'point_maps': [],
        'intrinsics': [],
        'extrinsics': [],
        'images': []
    }
    
    # Process in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i + batch_size]
        
        # Load and preprocess images
        images = load_and_preprocess_images([str(p) for p in batch_paths]).to(device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                # Get VGGT predictions
                predictions = model(images)
        
        # Debug: print what VGGT actually returns
        if i == 0:  # Only print once
            logger.info(f"VGGT output keys: {predictions.keys()}")
            for key, value in predictions.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        
        # Extract results based on actual VGGT outputs
        # Common VGGT outputs include: depth, points3d, camera_intrinsics, camera_extrinsics, etc.
        if 'depth' in predictions:
            depth_maps = predictions['depth'].cpu().numpy()
            # Handle batched outputs - split into individual depth maps
            if depth_maps.ndim == 4:  # (batch, channel, height, width)
                for j in range(depth_maps.shape[0]):
                    results['depth_maps'].append(depth_maps[j])
            elif depth_maps.ndim == 3:  # (batch, height, width)
                for j in range(depth_maps.shape[0]):
                    results['depth_maps'].append(depth_maps[j])
            else:
                results['depth_maps'].append(depth_maps)
        
        if 'points3d' in predictions:
            point_maps = predictions['points3d'].cpu().numpy()
            # Handle batched outputs
            if point_maps.ndim >= 3 and point_maps.shape[0] > 1:
                for j in range(point_maps.shape[0]):
                    results['point_maps'].append(point_maps[j])
            else:
                results['point_maps'].append(point_maps)
        
        # Extract camera poses from pose_enc
        if 'pose_enc' in predictions:
            pose_enc = predictions['pose_enc'].cpu().numpy()
            # pose_enc is typically [batch, num_views, 9] where 9 = rotation (6) + translation (3)
            if pose_enc.ndim >= 2:
                for j in range(pose_enc.shape[0]):
                    if pose_enc.shape[1] == 1:  # Single view
                        pose = pose_enc[j, 0]  # Get the 9 values
                    else:
                        pose = pose_enc[j]  # Multiple views
                    
                    # Convert pose encoding to rotation matrix and translation
                    # VGGT uses 6D rotation representation + 3D translation
                    if len(pose) == 9:
                        # Extract rotation (first 6) and translation (last 3)
                        rot_6d = pose[:6]
                        translation = pose[6:9]
                        
                        # Convert 6D rotation to 3x3 matrix
                        # Based on "On the Continuity of Rotation Representations in Neural Networks"
                        a1, a2 = rot_6d[:3], rot_6d[3:6]
                        b1 = a1 / (np.linalg.norm(a1) + 1e-8)
                        b2 = a2 - np.dot(b1, a2) * b1
                        b2 = b2 / (np.linalg.norm(b2) + 1e-8)
                        b3 = np.cross(b1, b2)
                        
                        rotation = np.stack([b1, b2, b3], axis=1)  # 3x3
                        
                        # Create 4x4 transformation matrix
                        transform = np.eye(4)
                        transform[:3, :3] = rotation
                        transform[:3, 3] = translation
                        
                        results['extrinsics'].append(transform)
                        
                        # Also estimate intrinsics from image size
                        # This is a rough estimate - VGGT might not provide explicit intrinsics
                        img_size = 518  # VGGT's output size
                        focal_length = img_size * 1.2  # Rough estimate
                        intrinsic = np.array([
                            [focal_length, 0, img_size/2],
                            [0, focal_length, img_size/2],
                            [0, 0, 1]
                        ])
                        results['intrinsics'].append(intrinsic)
            else:
                logger.warning(f"Unexpected pose_enc shape: {pose_enc.shape}")
        
        # Also extract world points if available
        if 'world_points' in predictions:
            world_points = predictions['world_points'].cpu().numpy()
            if world_points.ndim >= 3 and world_points.shape[0] > 1:
                for j in range(world_points.shape[0]):
                    results['point_maps'].append(world_points[j])
            else:
                results['point_maps'].append(world_points)
        
        # Store processed images
        processed_images = images.cpu()
        # Handle batched images
        if processed_images.ndim == 4:  # (batch, channels, height, width)
            for j in range(processed_images.shape[0]):
                results['images'].append(processed_images[j])
        else:
            results['images'].append(processed_images)
    
    return results


def save_neuralangelo_format(
    results: Dict,
    image_paths: List[Path],
    output_dir: Path,
    original_metadata: Optional[Dict] = None
):
    """Save results in Neuralangelo format"""
    # Create output directories
    out_images = output_dir / "images"
    out_masks = output_dir / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    
    # Save processed images (518x518 or whatever VGGT outputs)
    logger.info("Saving processed images...")
    for idx, (img_tensor, orig_path) in enumerate(zip(results['images'], image_paths)):
        # Convert tensor to PIL image
        if img_tensor.ndim == 4:  # (batch, channels, height, width)
            img_tensor = img_tensor.squeeze(0)  # Remove batch dimension
        
        img_array = img_tensor.permute(1, 2, 0).numpy()
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # Save with consistent naming
        out_name = f"frame_{idx:06d}"
        img.save(out_images / f"{out_name}.png")
        
        # Copy mask if it exists
        mask_path = orig_path.parent.parent / "masks" / f"{orig_path.stem}.png"
        if mask_path.exists():
            # Resize mask to match VGGT output size
            mask = Image.open(mask_path).convert("L")
            mask_resized = mask.resize(img.size, Image.NEAREST)
            mask_resized.save(out_masks / f"{out_name}.png")
    
    # Save camera parameters
    logger.info("Saving camera parameters...")
    camera_data = {
        "camera_model": "PINHOLE",
        "frames": []
    }
    
    for idx, orig_path in enumerate(image_paths):
        frame_data = {
            "file_path": f"images/frame_{idx:06d}.png",
            "transform_matrix": None,  # Will be filled below
            "intrinsics": None
        }
        
        # Add camera parameters if available
        if idx < len(results['extrinsics']):
            # Convert extrinsics to 4x4 matrix
            extrinsic = results['extrinsics'][idx]
            if extrinsic.shape == (3, 4):
                transform = np.eye(4)
                transform[:3, :] = extrinsic
            else:
                transform = extrinsic
            frame_data["transform_matrix"] = transform.tolist()
        
        if idx < len(results['intrinsics']):
            intrinsic = results['intrinsics'][idx]
            frame_data["intrinsics"] = {
                "fx": float(intrinsic[0, 0]),
                "fy": float(intrinsic[1, 1]),
                "cx": float(intrinsic[0, 2]),
                "cy": float(intrinsic[1, 2])
            }
        
        camera_data["frames"].append(frame_data)
    
    # Save camera JSON
    with open(output_dir / "transforms.json", 'w') as f:
        json.dump(camera_data, f, indent=2)
    
    # Save depth maps if available
    if results['depth_maps']:
        depth_dir = output_dir / "depth"
        depth_dir.mkdir(exist_ok=True)
        
        logger.info("Saving depth maps...")
        for idx, depth in enumerate(results['depth_maps']):
            # Handle different depth map shapes from VGGT
            if depth.ndim == 3:  # (channels, height, width)
                depth = depth.squeeze(0) if depth.shape[0] == 1 else depth[0]
            elif depth.ndim == 4:  # Should not happen after our fix above
                depth = depth.squeeze()
            
            # Ensure depth is 2D
            if depth.ndim != 2:
                logger.warning(f"Unexpected depth shape: {depth.shape}, attempting to make 2D")
                while depth.ndim > 2:
                    depth = depth.squeeze(0) if depth.shape[0] == 1 else depth[0]
            
            # Normalize depth for visualization
            depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
            depth_vis = (depth_vis * 255).astype(np.uint8)
            
            depth_img = Image.fromarray(depth_vis)
            depth_img.save(depth_dir / f"frame_{idx:06d}_depth.png")
            
            # Also save raw depth as .npy
            np.save(depth_dir / f"frame_{idx:06d}_depth.npy", depth)
    
    # Save metadata
    metadata = {
        "num_frames": len(image_paths),
        "vggt_version": "1B",
        "processing_info": {
            "input_resolution": "varied",
            "output_resolution": results['images'][0].shape[1:] if results['images'] else None,
            "device": str(torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu")
        }
    }
    
    # Include original metadata if provided
    if original_metadata:
        metadata["original_metadata"] = original_metadata
    
    with open(output_dir / "vggt_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Process images through VGGT for Neuralangelo")
    parser.add_argument("images", type=Path, help="Input images directory")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth image")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for VGGT processing")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--metadata", type=Path, help="Path to metadata.json from prep_crop.py")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Load metadata if provided
    original_metadata = None
    if args.metadata and args.metadata.exists():
        with open(args.metadata, 'r') as f:
            original_metadata = json.load(f)
        logger.info(f"Loaded metadata from {args.metadata}")
    
    # Get image paths
    image_extensions = ['.jpg', '.JPG', '.png', '.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(sorted(args.images.glob(f"*{ext}")))
    
    # Apply stride
    image_paths = image_paths[::args.stride]
    
    logger.info(f"Found {len(image_paths)} images to process (stride={args.stride})")
    
    if not image_paths:
        logger.error(f"No images found in {args.images}")
        return
    
    # Load VGGT model
    model, dtype = load_vggt_model(args.device)
    
    # Process images
    results = process_images_batch(
        image_paths,
        model,
        args.device,
        dtype,
        args.batch_size
    )
    
    # Save results
    save_neuralangelo_format(
        results,
        image_paths,
        args.output_dir,
        original_metadata
    )
    
    # Create Neuralangelo config
    logger.info("Creating Neuralangelo configuration...")
    neuralangelo_config = {
        "dataset": {
            "root": str(args.output_dir),
            "type": "neuralangelo",
            "num_images": len(image_paths)
        },
        "model": {
            "type": "neuralangelo",
            "object": {
                "sdf": {
                    "encoding": {
                        "type": "hashgrid",
                        "n_levels": 16,
                        "n_features_per_level": 2,
                        "log2_hashmap_size": 22
                    }
                }
            }
        }
    }
    
    with open(args.output_dir / "neuralangelo_config.yaml", 'w') as f:
        import yaml
        yaml.dump(neuralangelo_config, f, default_flow_style=False)
    
    logger.info("\n✅ VGGT processing complete!")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Total frames processed: {len(image_paths)}")
    logger.info("\nNext steps:")
    logger.info("1. Review the generated camera poses in transforms.json")
    logger.info("2. Check depth maps in the depth/ subdirectory")
    logger.info("3. Run Neuralangelo training with the generated config")


if __name__ == "__main__":
    main()
