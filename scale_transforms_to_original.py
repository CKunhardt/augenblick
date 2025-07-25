#!/usr/bin/env python3
"""
Scale VGGT transforms from cropped coordinates back to original image coordinates
"""

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_crop_metadata(metadata_path: Path) -> Dict:
    """Load cropping metadata to understand the transformation"""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Crop metadata:")
    logger.info(f"  Global crop box: {metadata['global_box']}")
    logger.info(f"  Crop dimensions: {metadata['crop_dimensions']}")
    if 'square_size' in metadata:
        logger.info(f"  Square size: {metadata['square_size']}")
        logger.info(f"  Padding offsets: {metadata['padding_offsets']}")
    
    return metadata


def scale_intrinsics(intrinsics: Dict, crop_metadata: Dict, 
                    original_size: Tuple[int, int], vggt_size: int = 518) -> Dict:
    """
    Scale camera intrinsics from cropped/resized to original coordinates
    
    Critical: Must handle both the crop offset AND the scale factor
    """
    if not intrinsics:
        return None
    
    # Extract crop parameters
    crop_box = crop_metadata['global_box']  # [xmin, ymin, xmax, ymax]
    crop_xmin, crop_ymin = crop_box[0], crop_box[1]
    crop_width = crop_box[2] - crop_box[0] + 1
    crop_height = crop_box[3] - crop_box[1] + 1
    
    # Original image dimensions
    orig_width, orig_height = original_size
    
    # Handle square padding if present
    if 'square_size' in crop_metadata:
        square_size = crop_metadata['square_size']
        pad_x, pad_y = crop_metadata['padding_offsets']
        
        # VGGT saw the image with padding, so we need to account for that
        # The actual content starts at (pad_x, pad_y) in the square image
        # Scale from VGGT coordinates to square coordinates
        vggt_to_square = square_size / vggt_size
        
        # Principal point in square coordinates
        cx_square = intrinsics['cx'] * vggt_to_square
        cy_square = intrinsics['cy'] * vggt_to_square
        
        # Remove padding offset to get to crop coordinates
        cx_crop = cx_square - pad_x
        cy_crop = cy_square - pad_y
        
        # Scale factors from crop to original
        scale_x = 1.0  # No additional scaling needed
        scale_y = 1.0
        
        # Focal lengths scale with the resize
        fx = intrinsics['fx'] * vggt_to_square
        fy = intrinsics['fy'] * vggt_to_square
    else:
        # Direct scaling from VGGT to crop size
        scale_x = crop_width / vggt_size
        scale_y = crop_height / vggt_size
        
        # Scale focal lengths
        fx = intrinsics['fx'] * scale_x
        fy = intrinsics['fy'] * scale_y
        
        # Scale principal points to crop coordinates
        cx_crop = intrinsics['cx'] * scale_x
        cy_crop = intrinsics['cy'] * scale_y
    
    # Add crop offset to get to original image coordinates
    cx_original = cx_crop + crop_xmin
    cy_original = cy_crop + crop_ymin
    
    logger.info(f"Intrinsics scaling:")
    logger.info(f"  VGGT: fx={intrinsics['fx']:.1f}, cx={intrinsics['cx']:.1f}, cy={intrinsics['cy']:.1f}")
    logger.info(f"  Original: fx={fx:.1f}, cx={cx_original:.1f}, cy={cy_original:.1f}")
    logger.info(f"  Crop offset: ({crop_xmin}, {crop_ymin})")
    
    return {
        'fx': fx,
        'fy': fy,
        'cx': cx_original,
        'cy': cy_original,
        'width': orig_width,
        'height': orig_height
    }


def adjust_transform_matrix(transform: List[List[float]], crop_metadata: Dict,
                          original_size: Tuple[int, int], vggt_size: int = 518) -> List[List[float]]:
    """
    Adjust transformation matrix to account for cropping and scaling
    
    The transformation needs to account for:
    1. VGGT processing at 518x518
    2. Original crop from full image
    3. Potential square padding
    """
    if not transform:
        return None
    
    # Convert to numpy for easier manipulation
    T = np.array(transform)
    
    # Get crop parameters
    crop_box = crop_metadata['global_box']  # [xmin, ymin, xmax, ymax]
    crop_width = crop_box[2] - crop_box[0] + 1
    crop_height = crop_box[3] - crop_box[1] + 1
    
    # Handle square padding if present
    if 'square_size' in crop_metadata:
        square_size = crop_metadata['square_size']
        pad_x, pad_y = crop_metadata['padding_offsets']
        
        # Account for padding in the crop
        effective_crop_x = crop_box[0] - pad_x * (crop_width / (square_size - 2 * pad_x))
        effective_crop_y = crop_box[1] - pad_y * (crop_height / (square_size - 2 * pad_y))
        effective_scale_x = square_size / vggt_size
        effective_scale_y = square_size / vggt_size
    else:
        effective_crop_x = crop_box[0]
        effective_crop_y = crop_box[1]
        effective_scale_x = crop_width / vggt_size
        effective_scale_y = crop_height / vggt_size
    
    # The transform might need adjustment based on the preprocessing
    # This is a simplified version - you might need to refine based on actual results
    # For now, we'll keep the rotation/translation as-is since VGGT should handle
    # the object-centric coordinate system correctly
    
    return T.tolist()


def process_transforms(
    vggt_transforms_path: Path,
    crop_metadata_path: Path,
    original_images_dir: Path,
    output_dir: Path
):
    """Process VGGT transforms and scale them to original image dimensions"""
    
    # Load transforms and metadata
    with open(vggt_transforms_path, 'r') as f:
        transforms = json.load(f)
    
    crop_metadata = load_crop_metadata(crop_metadata_path)
    
    # Get original image size from first image
    sample_images = list(original_images_dir.glob("*.jpg")) + list(original_images_dir.glob("*.JPG"))
    if not sample_images:
        raise ValueError(f"No images found in {original_images_dir}")
    
    sample_img = Image.open(sample_images[0])
    original_width, original_height = sample_img.size
    logger.info(f"Original image size: {original_width}x{original_height}")
    
    # Calculate scaling factors
    crop_box = crop_metadata['global_box']
    crop_width = crop_box[2] - crop_box[0] + 1
    crop_height = crop_box[3] - crop_box[1] + 1
    
    # If images were made square, account for that
    if 'square_size' in crop_metadata:
        square_size = crop_metadata['square_size']
        pad_x, pad_y = crop_metadata['padding_offsets']
        
        # Scale from VGGT (518) to square crop size to original crop size
        scale_x = crop_width / (518 - 2 * pad_x * 518 / square_size)
        scale_y = crop_height / (518 - 2 * pad_y * 518 / square_size)
    else:
        # Direct scale from VGGT size to crop size
        scale_x = crop_width / 518
        scale_y = crop_height / 518
    
    # Offset for the crop position in original image
    offset_x = crop_box[0]
    offset_y = crop_box[1]
    
    logger.info(f"Scaling factors: x={scale_x:.3f}, y={scale_y:.3f}")
    logger.info(f"Offsets: x={offset_x}, y={offset_y}")
    
    # Create new transforms with scaled intrinsics
    new_transforms = {
        "camera_model": transforms["camera_model"],
        "frames": []
    }
    
    # Get frame mapping to find original image names
    frame_mapping = crop_metadata.get('frame_mapping', {})
    
    # Process first frame to show detailed scaling info
    show_details = True
    
    for idx, frame in enumerate(transforms["frames"]):
        # Find original image name
        frame_key = f"frame_{idx:04d}"
        if frame_key in frame_mapping:
            original_name = frame_mapping[frame_key]['original']
        else:
            # Fallback to sequential naming
            original_name = f"frame_{idx:04d}.jpg"
        
        # Scale intrinsics with proper offset handling
        scaled_intrinsics = scale_intrinsics(
            frame.get("intrinsics"),
            crop_metadata,
            (original_width, original_height),
            vggt_size=518
        )
        
        # Use transform matrix as-is (VGGT handles object-centric coordinates)
        # The extrinsics should remain the same as they represent camera pose
        # relative to the object, not the image
        transform_matrix = frame.get("transform_matrix")
        
        new_frame = {
            "file_path": f"images/{original_name}",
            "transform_matrix": transform_matrix,
            "intrinsics": scaled_intrinsics
        }
        new_transforms["frames"].append(new_frame)
        
        # Only show details for first frame
        if show_details:
            show_details = False
    
    # Save scaled transforms
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "transforms.json", 'w') as f:
        json.dump(new_transforms, f, indent=2)
    
    # Create symbolic links to original images
    (output_dir / "images").mkdir(exist_ok=True)
    for img_path in original_images_dir.glob("*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            link_path = output_dir / "images" / img_path.name
            if not link_path.exists():
                link_path.symlink_to(img_path.resolve())
    
    # Copy masks if they exist
    original_masks = original_images_dir.parent / "masks"
    if original_masks.exists():
        (output_dir / "masks").mkdir(exist_ok=True)
        for mask_path in original_masks.glob("*.png"):
            link_path = output_dir / "masks" / mask_path.name
            if not link_path.exists():
                link_path.symlink_to(mask_path.resolve())
    
    logger.info(f"\n✅ Transform scaling complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Original image resolution: {original_width}x{original_height}")
    logger.info(f"Ready for Neuralangelo with high-resolution images!")


def main():
    parser = argparse.ArgumentParser(
        description="Scale VGGT transforms from cropped to original image coordinates"
    )
    parser.add_argument("vggt_transforms", type=Path, 
                       help="Path to VGGT transforms.json")
    parser.add_argument("crop_metadata", type=Path,
                       help="Path to crop metadata.json")
    parser.add_argument("original_images", type=Path,
                       help="Path to original (full-res) images directory")
    parser.add_argument("--output_dir", type=Path, required=True,
                       help="Output directory for scaled transforms")
    
    args = parser.parse_args()
    
    process_transforms(
        args.vggt_transforms,
        args.crop_metadata,
        args.original_images,
        args.output_dir
    )


if __name__ == "__main__":
    main()
