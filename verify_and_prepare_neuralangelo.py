#!/usr/bin/env python3
"""
Verify the scaled transforms and optionally prepare VGGT depth maps for Neuralangelo initialization
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def verify_intrinsics(transforms_path: Path, sample_image_path: Path):
    """Verify that intrinsics are properly scaled"""
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)
    
    # Get first frame's intrinsics
    frame0 = transforms['frames'][0]
    
    # Handle different intrinsics formats
    if 'intrinsics' in frame0 and frame0['intrinsics'] is not None:
        intrinsics_data = frame0['intrinsics']
        
        # Check if it's a matrix or dict format
        if isinstance(intrinsics_data, list):
            # Matrix format [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            intrinsics = {
                'fx': intrinsics_data[0][0],
                'fy': intrinsics_data[1][1],
                'cx': intrinsics_data[0][2],
                'cy': intrinsics_data[1][2]
            }
        elif isinstance(intrinsics_data, dict):
            # Already in dict format
            intrinsics = intrinsics_data
        else:
            raise ValueError(f"Unknown intrinsics format: {type(intrinsics_data)}")
    else:
        # Try to get from global transforms data
        if 'fl_x' in transforms and 'fl_y' in transforms:
            intrinsics = {
                'fx': transforms['fl_x'],
                'fy': transforms['fl_y'],
                'cx': transforms.get('cx', transforms.get('w', 0) / 2),
                'cy': transforms.get('cy', transforms.get('h', 0) / 2)
            }
        else:
            raise KeyError("No intrinsics found in transforms.json. Please run fix_intrinsics.py first.")
    
    # Load sample image to check dimensions
    img = Image.open(sample_image_path)
    width, height = img.size
    
    logger.info(f"\nIntrinsics verification:")
    logger.info(f"Image dimensions: {width}x{height}")
    logger.info(f"Focal length: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")
    logger.info(f"Principal point: cx={intrinsics['cx']:.1f}, cy={intrinsics['cy']:.1f}")
    
    # Sanity checks
    cx_centered = abs(intrinsics['cx'] - width/2) < width * 0.1
    cy_centered = abs(intrinsics['cy'] - height/2) < height * 0.1
    
    logger.info(f"\nSanity checks:")
    logger.info(f"Principal point roughly centered: {'✓' if cx_centered and cy_centered else '✗'}")
    logger.info(f"Focal length reasonable: {'✓' if intrinsics['fx'] > width*0.5 and intrinsics['fx'] < width*3 else '✗'}")
    
    # Check all frames have valid transforms (not intrinsics necessarily)
    null_count = sum(1 for frame in transforms['frames'] if frame.get('transform_matrix') is None)
    logger.info(f"Frames with valid transforms: {len(transforms['frames']) - null_count}/{len(transforms['frames'])}")
    
    return intrinsics, (width, height)


def prepare_depth_initialization(vggt_depth_dir: Path, output_dir: Path, 
                                crop_metadata_path: Path, target_size: tuple):
    """
    Prepare VGGT depth maps as initialization for Neuralangelo
    Scale depth maps from 518x518 to original resolution
    """
    if not vggt_depth_dir.exists():
        logger.warning(f"VGGT depth directory not found: {vggt_depth_dir}")
        return
    
    logger.info("\nPreparing depth maps for initialization...")
    
    # Load crop metadata
    with open(crop_metadata_path, 'r') as f:
        crop_metadata = json.load(f)
    
    crop_box = crop_metadata['global_box']
    
    # Create output directory
    depth_output = output_dir / "depth_init"
    depth_output.mkdir(parents=True, exist_ok=True)
    
    # Process depth maps
    depth_files = sorted(vggt_depth_dir.glob("*_depth.npy"))
    
    for depth_file in depth_files[:3]:  # Process first 3 as examples
        # Load VGGT depth
        depth_vggt = np.load(depth_file)
        
        # Create full resolution depth map
        depth_full = np.zeros(target_size[::-1])  # (height, width)
        
        # If images were made square, account for padding
        if 'square_size' in crop_metadata:
            square_size = crop_metadata['square_size']
            pad_x, pad_y = crop_metadata['padding_offsets']
            
            # Remove padding from depth map
            unpad_width = int(518 * (square_size - 2*pad_x) / square_size)
            unpad_height = int(518 * (square_size - 2*pad_y) / square_size)
            
            depth_unpadded = depth_vggt[
                int(pad_y * 518 / square_size):int(pad_y * 518 / square_size) + unpad_height,
                int(pad_x * 518 / square_size):int(pad_x * 518 / square_size) + unpad_width
            ]
        else:
            depth_unpadded = depth_vggt
        
        # Resize to crop size
        from scipy.ndimage import zoom
        crop_width = crop_box[2] - crop_box[0] + 1
        crop_height = crop_box[3] - crop_box[1] + 1
        
        zoom_y = crop_height / depth_unpadded.shape[0]
        zoom_x = crop_width / depth_unpadded.shape[1]
        
        depth_resized = zoom(depth_unpadded, (zoom_y, zoom_x), order=1)
        
        # Place in full resolution image
        depth_full[crop_box[1]:crop_box[3]+1, crop_box[0]:crop_box[2]+1] = depth_resized
        
        # Save
        output_name = depth_file.stem.replace('_depth', '') + '_depth_fullres.npy'
        np.save(depth_output / output_name, depth_full)
        
        # Also save as image for visualization
        depth_vis = (depth_full - depth_full.min()) / (depth_full.max() - depth_full.min() + 1e-6)
        depth_vis = (depth_vis * 255).astype(np.uint8)
        Image.fromarray(depth_vis).save(depth_output / output_name.replace('.npy', '.png'))
        
        logger.info(f"Saved {output_name}")


def create_neuralangelo_config(output_dir: Path, image_size: tuple, batch_size: int = 1):
    """Create optimized Neuralangelo config for high-res training"""
    config = {
        "data": {
            "root": str(output_dir),
            "type": "neuralangelo",
            "num_images": len(list((output_dir / "images").glob("*.jpg"))),
            "train": {
                "batch_size": batch_size,
                "image_size": list(image_size[::-1]),  # [height, width]
                "rays_per_image": 1024,  # Reduced for memory
            },
            "val": {
                "batch_size": 1,
                "image_size": list(image_size[::-1]),
                "rays_per_image": 1024,
            }
        },
        "model": {
            "type": "neuralangelo",
            "object": {
                "sdf": {
                    "encoding": {
                        "type": "hashgrid",
                        "n_levels": 16,
                        "n_features_per_level": 2,
                        "log2_hashmap_size": 22,  # May need to reduce for memory
                    }
                }
            }
        },
        "memory_optimization": {
            "gradient_checkpointing": True,  # Enable for large images
            "empty_cache_freq": 100,
        },
        "optim": {
            "params": {
                "lr": 0.001,  # Slightly lower for stability with large images
            }
        }
    }
    
    config_path = output_dir / "neuralangelo_config.yaml"
    
    # Write as YAML
    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Created Neuralangelo config: {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(description="Verify pipeline and prepare for Neuralangelo")
    parser.add_argument("scaled_dir", type=Path, help="Directory with scaled transforms")
    parser.add_argument("--vggt_depth", type=Path, help="VGGT depth directory")
    parser.add_argument("--crop_metadata", type=Path, help="Crop metadata.json")
    parser.add_argument("--prepare_depth", action="store_true", help="Prepare depth initialization")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for Neuralangelo")
    
    args = parser.parse_args()
    
    # Verify transforms
    transforms_path = args.scaled_dir / "transforms.json"
    sample_image = list((args.scaled_dir / "images").glob("*.jpg"))[0]
    
    try:
        intrinsics, image_size = verify_intrinsics(transforms_path, sample_image)
    except KeyError as e:
        logger.error(f"Error: {e}")
        logger.error("Please run fix_intrinsics.py to add missing intrinsics to transforms.json")
        return 1
    
    # Prepare depth if requested
    if args.prepare_depth and args.vggt_depth and args.crop_metadata:
        prepare_depth_initialization(
            args.vggt_depth,
            args.scaled_dir,
            args.crop_metadata,
            image_size
        )
    
    # Create optimized config
    config_path = create_neuralangelo_config(args.scaled_dir, image_size, args.batch_size)
    
    # Memory estimate
    pixels_per_image = image_size[0] * image_size[1]
    memory_gb = (pixels_per_image * args.batch_size * 4 * 3) / (1024**3)  # Rough estimate
    
    logger.info(f"\n{'='*60}")
    logger.info("Pipeline verification complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Image resolution: {image_size[0]}x{image_size[1]}")
    logger.info(f"Estimated memory per batch: ~{memory_gb:.1f} GB")
    logger.info(f"\nRecommendations:")
    logger.info(f"- For {image_size[0]}x{image_size[1]} images, use batch_size=1")
    logger.info(f"- Enable gradient checkpointing (already set in config)")
    logger.info(f"- Monitor GPU memory and reduce rays_per_image if needed")
    logger.info(f"\nTo start training:")
    logger.info(f"cd {args.scaled_dir}")
    logger.info(f"python train.py --config neuralangelo_config.yaml")


if __name__ == "__main__":
    main()
