import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_camera_groups(image_files: List[Path], frames_per_camera: int = 46) -> Dict[int, List[Path]]:
    """Group images by camera based on sequential ordering"""
    camera_groups = {}
    
    for i, img_path in enumerate(image_files):
        camera_id = i // frames_per_camera  # Camera 0, 1, 2
        if camera_id not in camera_groups:
            camera_groups[camera_id] = []
        camera_groups[camera_id].append(img_path)
    
    return camera_groups

def sample_within_cameras(camera_groups: Dict[int, List[Path]], stride: int) -> List[Path]:
    """Sample every Nth frame within each camera group"""
    sampled_images = []
    
    for camera_id in sorted(camera_groups.keys()):
        camera_images = camera_groups[camera_id]
        # Sample every stride-th image from this camera
        sampled = camera_images[::stride]
        sampled_images.extend(sampled)
        logger.info(f"Camera {camera_id + 1}: selected {len(sampled)} frames from {len(camera_images)}")
    
    return sampled_images

def compute_global_bbox_from_selection(image_paths: List[Path], mask_dir: Path, 
                                      padding: int = 50) -> Tuple[int, int, int, int]:
    """Compute global bounding box from selected images"""
    logger.info(f"Computing global bounding box from {len(image_paths)} selected masks...")
    
    if not image_paths:
        raise ValueError("No images found to compute bounding box!")
    
    global_xmin, global_ymin = float('inf'), float('inf')
    global_xmax, global_ymax = 0, 0
    
    valid_masks = 0
    for img_path in tqdm(image_paths, desc="Analyzing masks"):
        mask_path = mask_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            logger.warning(f"Mask not found: {mask_path}")
            continue
            
        mask = np.array(Image.open(mask_path).convert("L"))
        ys, xs = np.nonzero(mask > 127)
        
        if xs.size > 0:
            valid_masks += 1
            global_xmin = min(global_xmin, xs.min())
            global_xmax = max(global_xmax, xs.max())
            global_ymin = min(global_ymin, ys.min())
            global_ymax = max(global_ymax, ys.max())
    
    if valid_masks == 0:
        raise ValueError("No valid masks found!")
    
    logger.info(f"Found {valid_masks} valid masks")
    
    # Add padding
    h, w = mask.shape if 'mask' in locals() else (4160, 6240)  # Your actual dimensions
    global_xmin = max(0, int(global_xmin - padding))
    global_ymin = max(0, int(global_ymin - padding))
    global_xmax = min(w - 1, int(global_xmax + padding))
    global_ymax = min(h - 1, int(global_ymax + padding))
    
    return global_xmin, global_ymin, global_xmax, global_ymax

def main():
    parser = argparse.ArgumentParser(description="Camera-aware cropping for multi-camera datasets")
    parser.add_argument("images", type=Path, help="Images directory")
    parser.add_argument("masks", type=Path, help="Masks directory")
    parser.add_argument("--stride", type=int, default=3, 
                       help="Sample every Nth frame WITHIN each camera")
    parser.add_argument("--cameras-per-group", type=int, default=46,
                       help="Number of frames per camera")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--make-square", action="store_true", 
                       help="Make crops square by padding shorter dimension")
    
    args = parser.parse_args()
    
    if args.out is None:
        args.out = args.images.parent / f"{args.images.parent.name}_cropped_cameras"
    
    # Create output directories
    out_images = args.out / "images"
    out_masks = args.out / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    
    # Get all images - check both .jpg and .JPG
    image_files = sorted(args.images.glob("frame_*.jpg"))
    if not image_files:
        image_files = sorted(args.images.glob("frame_*.JPG"))
    if not image_files:
        # Try numbered pattern
        image_files = sorted(args.images.glob("frame_????.jpg"))
        if not image_files:
            image_files = sorted(args.images.glob("frame_????.JPG"))
    
    logger.info(f"Found {len(image_files)} total images")
    
    if not image_files:
        logger.error(f"No images found in {args.images}")
        logger.error("Looking for patterns: frame_*.jpg, frame_*.JPG, frame_????.jpg, frame_????.JPG")
        return
    
    # Group by camera
    camera_groups = detect_camera_groups(image_files, args.cameras_per_group)
    logger.info(f"Detected {len(camera_groups)} cameras")
    
    # Sample within each camera
    selected_images = sample_within_cameras(camera_groups, args.stride)
    logger.info(f"Total selected frames: {len(selected_images)}")
    
    # Compute global bounding box
    global_box = compute_global_bbox_from_selection(selected_images, args.masks)
    xmin, ymin, xmax, ymax = global_box
    crop_width = xmax - xmin + 1
    crop_height = ymax - ymin + 1
    
    logger.info(f"Global crop box: {global_box}")
    logger.info(f"Crop dimensions: {crop_width} x {crop_height}")
    
    # Process selected images
    output_metadata = {
        "total_frames": len(selected_images),
        "cameras": len(camera_groups),
        "frames_per_camera": args.cameras_per_group,
        "stride": args.stride,
        "global_box": list(global_box),
        "crop_dimensions": [crop_width, crop_height],
        "make_square": args.make_square,
        "camera_assignments": {},
        "frame_mapping": {}
    }
    
    output_idx = 0
    for camera_id in sorted(camera_groups.keys()):
        camera_images = camera_groups[camera_id]
        sampled = camera_images[::args.stride]
        
        camera_frames = []
        for img_path in tqdm(sampled, desc=f"Processing camera {camera_id + 1}"):
            # Load and crop
            img = Image.open(img_path)
            mask_path = args.masks / f"{img_path.stem}.png"
            
            if mask_path.exists():
                mask = Image.open(mask_path).convert("L")
                
                # Crop both
                img_cropped = img.crop((xmin, ymin, xmax + 1, ymax + 1))
                mask_cropped = mask.crop((xmin, ymin, xmax + 1, ymax + 1))
                
                # Make square if requested
                if args.make_square:
                    # Determine target size (use the larger dimension)
                    target_size = max(img_cropped.width, img_cropped.height)
                    
                    # Create square images with black padding
                    img_square = Image.new('RGB', (target_size, target_size), (0, 0, 0))
                    mask_square = Image.new('L', (target_size, target_size), 0)
                    
                    # Calculate position to paste (center the image)
                    x_offset = (target_size - img_cropped.width) // 2
                    y_offset = (target_size - img_cropped.height) // 2
                    
                    # Paste cropped images into square canvases
                    img_square.paste(img_cropped, (x_offset, y_offset))
                    mask_square.paste(mask_cropped, (x_offset, y_offset))
                    
                    img_cropped = img_square
                    mask_cropped = mask_square
                    
                    output_metadata["square_size"] = target_size
                    output_metadata["padding_offsets"] = [x_offset, y_offset]
                
                # Save with new sequential names
                out_name = f"frame_{output_idx:04d}"
                img_cropped.save(out_images / f"{out_name}.jpg", quality=95)
                mask_cropped.save(out_masks / f"{out_name}.png")
                
                camera_frames.append(output_idx)
                output_metadata["frame_mapping"][out_name] = {
                    "original": img_path.name,
                    "camera": camera_id + 1,
                    "camera_frame_idx": camera_images.index(img_path)
                }
                
                output_idx += 1
        
        output_metadata["camera_assignments"][f"camera_{camera_id + 1}"] = camera_frames
    
    # Save metadata
    with open(args.out / "metadata.json", 'w') as f:
        json.dump(output_metadata, f, indent=2)
    
    logger.info(f"\n✅ Camera-aware cropping complete!")
    logger.info(f"Output: {args.out}")
    logger.info(f"Frames per camera: {len(selected_images) // len(camera_groups)}")
    
    # Show camera distribution
    for cam_name, frames in output_metadata["camera_assignments"].items():
        logger.info(f"{cam_name}: frames {frames[0]}-{frames[-1]} ({len(frames)} total)")

if __name__ == "__main__":
    main()
