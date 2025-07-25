"""
Complete masked 3D reconstruction pipeline with VGGT initialization and depth map support
Compatible with pycolmap 3.x and COLMAP 3.12+
"""

import os
import sys
import json
import shutil
import subprocess
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import traceback
import time
import random
import gc
import torch  # Added for GPU memory management
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def organize_dataset(input_dir: Path, output_dir: Path) -> Tuple[Path, Path, int, int]:
    """Separate images and masks, return organized paths"""
    logger.info("Organizing dataset...")
    
    # Create output directories
    organized_dir = output_dir / "organized"
    images_dir = organized_dir / "images"
    masks_dir = organized_dir / "masks"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all files
    all_files = list(input_dir.glob("*"))
    image_files = []
    mask_files = []
    
    for f in all_files:
        if f.is_file():
            if ".mask.png" in f.name:
                mask_files.append(f)
            elif f.suffix.upper() in ['.JPG', '.JPEG', '.PNG'] and '.mask' not in f.name:
                image_files.append(f)
    
    logger.info(f"Found {len(image_files)} images and {len(mask_files)} masks")
    
    # Copy files to organized directories
    image_count = 0
    mask_count = 0
    
    for img_file in image_files:
        # Copy image
        dst_image = images_dir / img_file.name
        if not dst_image.exists():
            shutil.copy2(img_file, dst_image)
        image_count += 1
        
        # Find corresponding mask
        base_name = img_file.stem
        mask_patterns = [
            f"{base_name}.jpg.mask.png",
            f"{base_name}.JPG.mask.png"
        ]
        
        for mask_file in mask_files:
            if mask_file.name in mask_patterns:
                dst_mask = masks_dir / f"{base_name}.png"
                if not dst_mask.exists():
                    shutil.copy2(mask_file, dst_mask)
                mask_count += 1
                break
    
    logger.info(f"Organized: {image_count} images, {mask_count} masks")
    return images_dir, masks_dir, image_count, mask_count


def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion [qw, qx, qy, qz]"""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    return np.array([qw, qx, qy, qz])


def verify_depth_maps(neuralangelo_dir: Path) -> bool:
    """Verify that depth maps are in the correct format for Neuralangelo"""
    depth_dir = neuralangelo_dir / "depth_maps"
    
    if not depth_dir.exists():
        return False
    
    depth_files = sorted(list(depth_dir.glob("depth_*.npy")))
    conf_files = sorted(list(depth_dir.glob("conf_*.npy")))
    
    if not depth_files:
        logger.warning("No depth maps found")
        return False
    
    # Check a sample depth map
    sample_depth = np.load(depth_files[0])
    logger.info(f"Sample depth map shape: {sample_depth.shape}, dtype: {sample_depth.dtype}")
    
    if conf_files:
        sample_conf = np.load(conf_files[0])
        logger.info(f"Sample confidence map shape: {sample_conf.shape}, dtype: {sample_conf.dtype}")
    
    # Verify naming convention matches transforms.json
    transforms_file = neuralangelo_dir / "transforms.json"
    if transforms_file.exists():
        with open(transforms_file, 'r') as f:
            transforms = json.load(f)
        
        num_frames = len(transforms.get('frames', []))
        if len(depth_files) != num_frames:
            logger.warning(f"Mismatch: {len(depth_files)} depth maps but {num_frames} frames in transforms.json")
            return False
    
    logger.info(f"✓ Depth maps verified: {len(depth_files)} depth maps ready for use")
    return True


def step2_run_vggt_direct(images_dir: Path, output_dir: Path, vggt_script_path: Path, 
                          device: str = "cuda:0", batch_size: Optional[int] = None) -> Optional[Path]:
    """Run VGGT script that outputs Neuralangelo format directly with depth maps"""
    logger.info("\n=== Step 2: Running VGGT (Direct Neuralangelo Output with Depth) ===")
    
    # Check if VGGT script exists
    if not vggt_script_path.exists():
        logger.error(f"VGGT script not found at {vggt_script_path}")
        return None
    
    vggt_dir = output_dir / "vggt"
    vggt_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    neuralangelo_data_dir = vggt_dir / "neuralangelo_data"
    if neuralangelo_data_dir.exists() and (neuralangelo_data_dir / "transforms.json").exists():
        logger.info("VGGT output already exists, using existing data...")
        # Check if depth maps exist
        depth_dir = neuralangelo_data_dir / "depth_maps"
        if depth_dir.exists() and list(depth_dir.glob("*.npy")):
            logger.info(f"Found {len(list(depth_dir.glob('*.npy')))} depth maps")
        return neuralangelo_data_dir
    
    try:
        # Prepare input directory structure for VGGT script
        vggt_input_dir = vggt_dir / "input"
        vggt_input_images = vggt_input_dir / "images"
        vggt_input_images.mkdir(parents=True, exist_ok=True)
        
        # Copy images to VGGT input structure
        logger.info(f"Copying images to VGGT input directory...")
        image_files = list(images_dir.glob("*"))
        image_count = 0
        for img_file in image_files:
            if img_file.suffix.upper() in ['.JPG', '.JPEG', '.PNG']:
                shutil.copy2(img_file, vggt_input_images / img_file.name)
                image_count += 1
        
        logger.info(f"Copied {image_count} images")
        
        # If batch size is specified and we have many images, process in batches
        if batch_size and image_count > batch_size:
            logger.warning(f"Processing {image_count} images in batches of {batch_size} is not yet implemented")
            logger.warning("Processing all images at once - may cause OOM")
        
        # Run VGGT script with save_for_neuralangelo flag to ensure depth maps are saved
        cmd = [
            sys.executable,  # Use current Python interpreter
            str(vggt_script_path),
            str(vggt_input_dir),
            "--output_dir", str(vggt_dir),
            "--save_for_neuralangelo",  # This flag ensures depth maps are saved
            "--skip_glb",
            "--conf_thres", "50.0"
        ]
        
        logger.info(f"Running VGGT command: {' '.join(cmd)}")
        
        # Set environment to use correct GPU and manage memory
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Help with fragmentation
        
        # Run the script
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            logger.error(f"VGGT script failed with return code {result.returncode}")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            return None
        
        logger.info(f"VGGT stdout: {result.stdout}")
        
        # Find the generated neuralangelo_data directory
        neuralangelo_dirs = list(vggt_dir.glob("**/neuralangelo_data"))
        if neuralangelo_dirs:
            neuralangelo_data_dir = neuralangelo_dirs[0]
            logger.info(f"✓ VGGT successfully created Neuralangelo data at: {neuralangelo_data_dir}")
            
            # Verify depth maps were created
            depth_dir = neuralangelo_data_dir / "depth_maps"
            if depth_dir.exists():
                depth_files = list(depth_dir.glob("*.npy"))
                logger.info(f"✓ Found {len(depth_files)} depth maps in {depth_dir}")
                
                # Also check for confidence maps
                conf_files = list(depth_dir.glob("conf_*.npy"))
                if conf_files:
                    logger.info(f"✓ Found {len(conf_files)} confidence maps")
            else:
                logger.warning("No depth maps directory found - depth supervision will not be available")
            
            return neuralangelo_data_dir
        else:
            logger.error("VGGT script completed but no neuralangelo_data directory found")
            return None
            
    except Exception as e:
        logger.error(f"VGGT processing failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def step2_run_vggt(images_dir: Path, output_dir: Path, vggt_path: Path, device: str = "cuda:0") -> Optional[Path]:
    """Run VGGT model to get initial camera poses and depth maps"""
    logger.info("\n=== Step 2: Running VGGT ===")
    
    # Check if VGGT source exists
    if not vggt_path.exists():
        logger.warning(f"VGGT source not found at {vggt_path}")
        return None
    
    vggt_dir = output_dir / "vggt"
    vggt_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    vggt_sparse_dir = vggt_dir / "sparse"
    if vggt_sparse_dir.exists() and (vggt_sparse_dir / "cameras.bin").exists():
        logger.info("VGGT output already exists, skipping...")
        return vggt_sparse_dir
    
    try:
        # Add VGGT to path
        sys.path.insert(0, str(vggt_path))
        
        import torch
        import torch.nn.functional as F
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images_square
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        import pycolmap
        import trimesh
        
        # Set device and dtype
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        logger.info(f"Using device: {device}, dtype: {dtype}")
        
        # Load model
        logger.info("Loading VGGT model...")
        model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        model.eval()
        model = model.to(device)
        logger.info("VGGT model loaded successfully")
        
        # Prepare image paths
        image_paths = sorted([str(p) for p in images_dir.glob("*") if p.suffix.upper() in ['.JPG', '.JPEG', '.PNG']])
        logger.info(f"Processing {len(image_paths)} images with VGGT")
        
        # VGGT settings
        fixed_resolution = 518
        img_load_resolution = 1024
        
        # Load and preprocess images
        images, original_coords = load_and_preprocess_images_square(image_paths, img_load_resolution)
        images = images.to(device)
        
        # Resize for VGGT
        images_resized = F.interpolate(
            images, 
            size=(fixed_resolution, fixed_resolution),
            mode="bilinear", 
            align_corners=False
        )
        
        # Run inference
        logger.info("Running VGGT inference...")
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                images_batch = images_resized[None]
                aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
                
                # Predict Cameras
                pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_batch.shape[-2:])
                
                # Predict Depth Maps  
                depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_batch, ps_idx)
        
        # Convert to numpy
        extrinsic = extrinsic.squeeze(0).cpu().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().numpy()
        depth_map = depth_map.squeeze(0).cpu().numpy()
        depth_conf = depth_conf.squeeze(0).cpu().numpy()
        
        # Debug original shapes
        logger.info(f"Raw shapes after squeeze(0) - depth_map: {depth_map.shape}, depth_conf: {depth_conf.shape}")
        logger.info(f"Extrinsic: {extrinsic.shape}, Intrinsic: {intrinsic.shape}")
        
        # Fix depth_map shape - VGGT's unproject expects (S, H, W) or (S, H, W, 1)
        if depth_map.ndim == 4:
            # Could be (num_frames, H, W, 1) or (num_frames, 1, H, W)
            if depth_map.shape[-1] == 1:
                # Shape is (num_frames, H, W, 1) - this is fine for VGGT
                pass
            elif depth_map.shape[1] == 1:
                # Shape is (num_frames, 1, H, W) - need to transpose
                depth_map = depth_map.squeeze(1)  # -> (num_frames, H, W)
            else:
                # Unexpected 4D shape
                logger.warning(f"Unexpected 4D depth_map shape: {depth_map.shape}")
        elif depth_map.ndim == 3:
            # Shape is already (num_frames, H, W) - this is fine
            pass
        else:
            logger.error(f"Unexpected depth_map dimensions: {depth_map.ndim}, shape: {depth_map.shape}")
        
        # Fix depth_conf shape similarly
        if depth_conf.ndim == 4:
            if depth_conf.shape[-1] == 1:
                depth_conf = depth_conf.squeeze(-1)  # -> (num_frames, H, W)
            elif depth_conf.shape[1] == 1:
                depth_conf = depth_conf.squeeze(1)  # -> (num_frames, H, W)
        elif depth_conf.ndim == 3:
            # Already (num_frames, H, W)
            pass
        
        # Ensure we have the right number of frames
        num_frames = len(image_paths)
        if depth_map.shape[0] != num_frames:
            logger.error(f"Mismatch: depth_map has {depth_map.shape[0]} frames but we have {num_frames} images")
            # If depth_map is transposed, fix it
            if depth_map.shape[-1] == num_frames:
                logger.info("Attempting to transpose depth_map to fix frame count")
                depth_map = depth_map.transpose(2, 0, 1)  # (H, W, S) -> (S, H, W)
                depth_conf = depth_conf.transpose(2, 0, 1) if depth_conf.shape[-1] == num_frames else depth_conf
        
        logger.info(f"Final shapes - depth_map: {depth_map.shape}, depth_conf: {depth_conf.shape}")
        
        # Unproject to 3D
        points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
        
        # Get RGB values
        points_rgb = F.interpolate(
            images, 
            size=(fixed_resolution, fixed_resolution),
            mode="bilinear", 
            align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)
        
        # Filter by confidence (lowered threshold for better results)
        conf_threshold = 2.0  # Lowered from 5.0
        max_points = 100000
        
        # Ensure depth_map and depth_conf are 3D (S, H, W) for coordinate grid creation
        if depth_map.ndim == 4 and depth_map.shape[-1] == 1:
            depth_map = depth_map.squeeze(-1)  # (S, H, W, 1) -> (S, H, W)
        if depth_conf.ndim == 4 and depth_conf.shape[-1] == 1:
            depth_conf = depth_conf.squeeze(-1)  # Keep arrays in sync
        
        # Create coordinate grid for tracking
        num_frames, height, width = depth_map.shape
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
        frame_indices = np.arange(num_frames)[:, np.newaxis, np.newaxis]
        
        x_coords = np.broadcast_to(x_coords, (num_frames, height, width))
        y_coords = np.broadcast_to(y_coords, (num_frames, height, width))
        frame_coords = np.broadcast_to(frame_indices, (num_frames, height, width))
        
        points_xyf = np.stack([x_coords, y_coords, frame_coords], axis=-1).astype(np.float32)
        
        # Apply confidence mask
        conf_mask = depth_conf >= conf_threshold
        
        # Randomly limit if too many points
        if conf_mask.sum() > max_points:
            true_indices = np.where(conf_mask.ravel())[0]
            selected_indices = np.random.choice(true_indices, max_points, replace=False)
            new_mask = np.zeros_like(conf_mask.ravel(), dtype=bool)
            new_mask[selected_indices] = True
            conf_mask = new_mask.reshape(conf_mask.shape)
        
        points_3d_filtered = points_3d[conf_mask]
        points_rgb_filtered = points_rgb[conf_mask]
        points_xyf_filtered = points_xyf[conf_mask]
        
        logger.info(f"Filtered to {len(points_3d_filtered)} confident 3D points")
        
        # Convert to COLMAP format using newer pycolmap API
        logger.info("Converting to COLMAP format...")
        
        # Create reconstruction
        reconstruction = pycolmap.Reconstruction()
        
        # Set up image size
        image_size = np.array([fixed_resolution, fixed_resolution])
        
        # Add cameras
        if intrinsic.ndim == 3 and np.allclose(intrinsic[0], intrinsic[1:]):
            # Shared camera model
            camera_id = 1
            camera = pycolmap.Camera(
                model='PINHOLE',
                width=int(image_size[0]),
                height=int(image_size[1]),
                params=[float(intrinsic[0, 0, 0]), float(intrinsic[0, 1, 1]), 
                       float(intrinsic[0, 0, 2]), float(intrinsic[0, 1, 2])]
            )
            camera.camera_id = camera_id  # Set the ID explicitly
            reconstruction.add_camera(camera)
            camera_ids = [camera_id] * len(image_paths)
        else:
            # Individual cameras
            camera_ids = []
            for i in range(len(image_paths)):
                camera_id = i + 1
                K = intrinsic[i] if intrinsic.ndim == 3 else intrinsic
                camera = pycolmap.Camera(
                    model='PINHOLE',
                    width=int(image_size[0]),
                    height=int(image_size[1]),
                    params=[float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]
                )
                camera.camera_id = camera_id  # Set unique ID for each camera
                reconstruction.add_camera(camera)
                camera_ids.append(camera_id)
        
        # Add images - simplified approach for your pycolmap version
        base_image_names = [Path(p).name for p in image_paths]
        image_ids = []
        
        # For turntable setup, we typically have one camera position
        # Log some diagnostic info about camera positions
        if extrinsic.ndim == 3:
            camera_positions = extrinsic[:, :3, 3]  # Extract translations
            cam_pos_std = np.std(camera_positions, axis=0)
            logger.info(f"Camera position std dev: {cam_pos_std}")
            logger.info(f"Likely turntable setup: {np.max(cam_pos_std) < 0.1}")
        
        logger.info("Adding images to reconstruction...")
        for i in range(len(image_paths)):
            # Create a simple image object
            try:
                image = pycolmap.Image(
                    name=base_image_names[i],
                    camera_id=camera_ids[i]
                )
                reconstruction.add_image(image)
                image_ids.append(image.image_id)
                logger.debug(f"Added image {i}: {base_image_names[i]}")
            except Exception as e:
                logger.error(f"Failed to add image {i}: {e}")
        
        logger.info(f"Added {len(image_ids)} images to reconstruction")
        
        # Add 3D points with the correct API
        if len(points_3d_filtered) > 0:
            logger.info(f"Adding {len(points_3d_filtered)} 3D points...")
            points_added = 0
            for i, (point3d, color) in enumerate(zip(points_3d_filtered, points_rgb_filtered)):
                try:
                    # Your pycolmap expects: xyz as numpy array, track, and color
                    xyz = np.array(point3d, dtype=np.float64)
                    color_uint8 = np.array(color, dtype=np.uint8)
                    track = pycolmap.Track()  # Empty track
                    
                    point_id = reconstruction.add_point3D(xyz, track, color_uint8)
                    points_added += 1
                except Exception as e:
                    if i == 0:  # Only log first error to avoid spam
                        logger.warning(f"Failed to add 3D point: {e}")
        
        # Update camera parameters for original resolution
        rescale_camera = True
        for img_id in image_ids:
            if rescale_camera:
                image = reconstruction.images[img_id]
                camera = reconstruction.cameras[image.camera_id]
                
                # Get original image size
                img_idx = img_id - 1  # Assuming sequential IDs
                real_image_size = original_coords[img_idx, -2:]
                resize_ratio = max(real_image_size) / fixed_resolution
                
                # Update camera parameters
                params = list(camera.params)
                params[0] *= resize_ratio  # fx
                params[1] *= resize_ratio  # fy
                params[2] = real_image_size[0] / 2  # cx
                params[3] = real_image_size[1] / 2  # cy
                
                # Update camera
                camera.width = int(real_image_size[0])
                camera.height = int(real_image_size[1])
                camera.params = params
                
                if len(set(camera_ids)) == 1:
                    # Only rescale once for shared camera
                    rescale_camera = False
        
        # Save COLMAP reconstruction
        vggt_sparse_dir.mkdir(parents=True, exist_ok=True)
        
        # Since we can't store poses in the reconstruction with this pycolmap version,
        # let's save the poses separately for later use
        poses_file = vggt_sparse_dir / "vggt_poses.npz"
        np.savez(poses_file, 
                 extrinsic=extrinsic,
                 intrinsic=intrinsic,
                 image_names=base_image_names)
        logger.info(f"Saved VGGT poses to {poses_file}")
        
        # Try to save what we can of the reconstruction
        try:
            reconstruction.write(str(vggt_sparse_dir))
            logger.info(f"Saved partial reconstruction to {vggt_sparse_dir}")
        except Exception as e:
            logger.warning(f"Could not save full reconstruction: {e}")
            # Save cameras and images in text format manually
            text_dir = vggt_sparse_dir / "text"
            text_dir.mkdir(exist_ok=True)
            
            # Save cameras.txt
            with open(text_dir / "cameras.txt", 'w') as f:
                f.write("# Camera list with one line of data per camera:\n")
                f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
                for cam_id, camera in reconstruction.cameras.items():
                    f.write(f"{cam_id} PINHOLE {camera.width} {camera.height} ")
                    f.write(" ".join(map(str, camera.params)) + "\n")
            
            # Save images.txt with poses from VGGT
            with open(text_dir / "images.txt", 'w') as f:
                f.write("# Image list with two lines of data per image:\n")
                f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
                f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
                for i, (img_id, img_name) in enumerate(zip(image_ids, base_image_names)):
                    # Get VGGT pose
                    E_c2w = extrinsic[i] if extrinsic.ndim == 3 else extrinsic
                    R_c2w = E_c2w[:3, :3]
                    t_c2w = E_c2w[:3, 3]
                    
                    # Convert to world-to-camera for COLMAP
                    R_w2c = R_c2w.T
                    t_w2c = -R_w2c @ t_c2w
                    qvec = rotation_matrix_to_quaternion(R_w2c)
                    
                    f.write(f"{img_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} ")
                    f.write(f"{t_w2c[0]} {t_w2c[1]} {t_w2c[2]} {camera_ids[i]} {img_name}\n")
                    f.write("\n")  # Empty line for points2D
            
            logger.info("Saved COLMAP text files as fallback")
        
        # Save point cloud
        if len(points_3d_filtered) > 0:
            trimesh.PointCloud(points_3d_filtered, colors=points_rgb_filtered).export(
                str(vggt_sparse_dir / "points.ply")
            )
        
        logger.info(f"VGGT reconstruction saved to {vggt_sparse_dir}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return vggt_sparse_dir
        
    except Exception as e:
        logger.error(f"VGGT processing failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None
    finally:
        # Remove VGGT from path
        if str(vggt_path) in sys.path:
            sys.path.remove(str(vggt_path))


def step3_run_colmap(images_dir: Path, output_dir: Path, vggt_sparse_dir: Optional[Path] = None, 
                     use_module: bool = False) -> Optional[Path]:
    """Run COLMAP reconstruction"""
    logger.info("\n=== Step 3: Running COLMAP ===")
    
    colmap_dir = output_dir / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    
    database_path = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)
    
    # Check if already exists
    if (sparse_dir / "0" / "cameras.bin").exists():
        logger.info("COLMAP reconstruction already exists, skipping...")
        return sparse_dir / "0"
    
    # Create COLMAP script
    if vggt_sparse_dir and vggt_sparse_dir.exists():
        logger.info("Using VGGT initialization for COLMAP")
        
        # Copy VGGT output as initialization
        init_sparse_dir = colmap_dir / "sparse_init"
        init_sparse_dir.mkdir(exist_ok=True)
        
        for file_name in ["cameras.bin", "images.bin", "points3D.bin"]:
            src_file = vggt_sparse_dir / file_name
            if src_file.exists():
                shutil.copy2(src_file, init_sparse_dir / file_name)
        
        script_content = f"""#!/bin/bash
{"module load colmap/3.11" if use_module else ""}

# Create database from existing model
colmap database_creator \\
    --database_path {database_path}

# Import existing model from VGGT
colmap feature_extractor \\
    --database_path {database_path} \\
    --image_path {images_dir} \\
    --ImageReader.camera_model SIMPLE_PINHOLE \\
    --ImageReader.single_camera 1 \\
    --SiftExtraction.use_gpu 0 \\
    --SiftExtraction.num_threads 8

# Feature matching
colmap exhaustive_matcher \\
    --database_path {database_path} \\
    --SiftMatching.use_gpu 0 \\
    --SiftMatching.num_threads 8

# Triangulation with VGGT initialization
colmap point_triangulator \\
    --database_path {database_path} \\
    --image_path {images_dir} \\
    --input_path {init_sparse_dir} \\
    --output_path {sparse_dir}/0

# Bundle adjustment
colmap bundle_adjuster \\
    --input_path {sparse_dir}/0 \\
    --output_path {sparse_dir}/0 \\
    --BundleAdjustment.refine_focal_length 1 \\
    --BundleAdjustment.refine_principal_point 0 \\
    --BundleAdjustment.refine_extra_params 0

# Convert to text format
mkdir -p {colmap_dir}/text
colmap model_converter \\
    --input_path {sparse_dir}/0 \\
    --output_path {colmap_dir}/text \\
    --output_type TXT
"""
    else:
        logger.info("Running standard COLMAP pipeline")
        script_content = f"""#!/bin/bash
{"module load colmap/3.11" if use_module else ""}

# Feature extraction
colmap feature_extractor \\
    --database_path {database_path} \\
    --image_path {images_dir} \\
    --ImageReader.camera_model SIMPLE_PINHOLE \\
    --ImageReader.single_camera 1 \\
    --SiftExtraction.use_gpu 0 \\
    --SiftExtraction.num_threads 8 \\
    --SiftExtraction.max_image_size 3200 \\
    --SiftExtraction.max_num_features 8192

# Feature matching
colmap exhaustive_matcher \\
    --database_path {database_path} \\
    --SiftMatching.use_gpu 0 \\
    --SiftMatching.num_threads 8

# Reconstruction
colmap mapper \\
    --database_path {database_path} \\
    --image_path {images_dir} \\
    --output_path {sparse_dir} \\
    --Mapper.num_threads 16

# Convert to text
mkdir -p {colmap_dir}/text
colmap model_converter \\
    --input_path {sparse_dir}/0 \\
    --output_path {colmap_dir}/text \\
    --output_type TXT
"""
    
    # Save and run script
    script_path = output_dir / "run_colmap.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    
    try:
        logger.info("Executing COLMAP script...")
        subprocess.run(["bash", str(script_path)], check=True)
        
        # Verify output
        if (sparse_dir / "0" / "cameras.bin").exists():
            logger.info("COLMAP reconstruction successful!")
            return sparse_dir / "0"
        else:
            logger.error("COLMAP output not found")
            return None
            
    except subprocess.CalledProcessError as e:
        logger.error(f"COLMAP failed: {e}")
        return None


def qvec_to_rotation_matrix(qvec):
    """Convert quaternion to rotation matrix"""
    qw, qx, qy, qz = qvec
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def parse_colmap_cameras(cameras_file: Path) -> Dict[int, Dict[str, Any]]:
    """Parse COLMAP cameras.txt file"""
    cameras = {}
    with open(cameras_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            
            if model == "SIMPLE_PINHOLE":
                cameras[camera_id] = {
                    'model': model,
                    'width': width,
                    'height': height,
                    'f': float(parts[4]),
                    'cx': float(parts[5]),
                    'cy': float(parts[6])
                }
            elif model == "PINHOLE":
                cameras[camera_id] = {
                    'model': model,
                    'width': width,
                    'height': height,
                    'fx': float(parts[4]),
                    'fy': float(parts[5]),
                    'cx': float(parts[6]),
                    'cy': float(parts[7])
                }
    
    return cameras


def parse_colmap_images(images_file: Path) -> Dict[int, Dict[str, Any]]:
    """Parse COLMAP images.txt file"""
    images = {}
    with open(images_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or not line:
            i += 1
            continue
        
        parts = line.split()
        if len(parts) >= 10:
            image_id = int(parts[0])
            image_name = ' '.join(parts[9:])
            
            images[image_id] = {
                'name': image_name,
                'camera_id': int(parts[8]),
                'qvec': [float(parts[j]) for j in range(1, 5)],
                'tvec': [float(parts[j]) for j in range(5, 8)]
            }
            i += 2  # Skip points2D line
        else:
            i += 1
    
    return images


def step4_convert_to_neuralangelo(colmap_dir: Path, images_dir: Path, masks_dir: Path, 
                                 output_dir: Path) -> Path:
    """Convert COLMAP output to Neuralangelo format"""
    logger.info("\n=== Step 4: Converting to Neuralangelo format ===")
    
    neuralangelo_dir = output_dir / "neuralangelo"
    neuralangelo_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    ngp_images = neuralangelo_dir / "images"
    ngp_masks = neuralangelo_dir / "masks"
    ngp_images.mkdir(exist_ok=True)
    ngp_masks.mkdir(exist_ok=True)
    
    # Find COLMAP text files
    text_dir = colmap_dir.parent / "text"
    if not text_dir.exists():
        text_dir = colmap_dir  # Try sparse directory directly
    
    cameras_file = text_dir / "cameras.txt"
    images_file = text_dir / "images.txt"
    
    if not cameras_file.exists() or not images_file.exists():
        raise FileNotFoundError(f"COLMAP text files not found in {text_dir}")
    
    # Parse COLMAP output
    cameras = parse_colmap_cameras(cameras_file)
    images = parse_colmap_images(images_file)
    
    # Get camera info (assuming single camera)
    camera_id = list(cameras.keys())[0]
    camera = cameras[camera_id]
    
    # Create transforms.json
    transforms = {
        "camera_model": "OPENCV",
        "frames": []
    }
    
    # Set camera intrinsics
    if camera['model'] == "SIMPLE_PINHOLE":
        transforms["fl_x"] = camera['f']
        transforms["fl_y"] = camera['f']
    else:
        transforms["fl_x"] = camera['fx']
        transforms["fl_y"] = camera['fy']
    
    transforms["cx"] = camera['cx']
    transforms["cy"] = camera['cy']
    transforms["w"] = camera['width']
    transforms["h"] = camera['height']
    
    # Add additional camera parameters
    transforms["sk_x"] = 0.0
    transforms["sk_y"] = 0.0
    transforms["k1"] = 0.0
    transforms["k2"] = 0.0
    transforms["k3"] = 0.0
    transforms["k4"] = 0.0
    transforms["p1"] = 0.0
    transforms["p2"] = 0.0
    
    # Calculate bounding sphere from camera positions
    camera_positions = []
    for img_info in images.values():
        t = np.array(img_info['tvec'])
        camera_positions.append(t)
    
    camera_positions = np.array(camera_positions)
    object_center = np.mean(camera_positions, axis=0)
    distances = np.linalg.norm(camera_positions - object_center, axis=1)
    sphere_radius = max(np.max(distances) * 1.5, 1.0)
    
    transforms["sphere_center"] = object_center.tolist()
    transforms["sphere_radius"] = float(sphere_radius)
    
    # Process each image
    successful_frames = 0
    for img_id, img_info in sorted(images.items()):
        # Get transformation matrix
        R = qvec_to_rotation_matrix(img_info['qvec'])
        t = np.array(img_info['tvec'])
        
        # Create 4x4 transformation matrix (camera-to-world)
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = t
        
        # Convert coordinate system (COLMAP to NeRF/NGP)
        coord_change = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        c2w = c2w @ coord_change
        
        # Find image and mask files
        image_name = img_info['name']
        src_image = images_dir / image_name
        
        # Try different extensions if exact match not found
        if not src_image.exists():
            for ext in ['.JPG', '.jpg', '.PNG', '.png', '.JPEG', '.jpeg']:
                candidate = images_dir / (image_name + ext)
                if candidate.exists():
                    src_image = candidate
                    image_name = candidate.name
                    break
        
        if src_image.exists():
            # Copy image
            dst_image = ngp_images / image_name
            if not dst_image.exists():
                shutil.copy2(src_image, dst_image)
            
            # Find and copy mask
            base_name = src_image.stem
            src_mask = masks_dir / f"{base_name}.png"
            if src_mask.exists():
                dst_mask = ngp_masks / f"{base_name}.png"
                if not dst_mask.exists():
                    shutil.copy2(src_mask, dst_mask)
                
                # Add to transforms
                frame = {
                    "file_path": f"images/{image_name}",
                    "transform_matrix": c2w.tolist(),
                    "mask_path": f"masks/{base_name}.png"
                }
                transforms["frames"].append(frame)
                successful_frames += 1
    
    # Save transforms.json
    with open(neuralangelo_dir / "transforms.json", 'w') as f:
        json.dump(transforms, f, indent=2)
    
    logger.info(f"Converted {successful_frames} frames to Neuralangelo format")
    return neuralangelo_dir


def detect_gpu_memory(gpu_index: int) -> int:
    """Auto-detect GPU memory in GB"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", f"--id={gpu_index}"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            memory_mb = float(result.stdout.strip())
            # Check for B200 GPU
            name_result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", f"--id={gpu_index}"], capture_output=True, text=True)
            if "B200" in name_result.stdout:
                logger.info("Detected B200 GPU - using high-res optimizations")
                return 180  # Force B200 settings
            return int(memory_mb / 1024)
    except:
        pass
    return 24  # Default conservative estimate


def calculate_training_params(gpu_memory_gb: int, num_images: int, max_steps: int = 50000) -> Dict[str, Any]:
    """Calculate optimal batch size and ray count based on GPU memory"""
    if gpu_memory_gb >= 160:
        # B200 or better - OPTIMIZED FOR HIGH-RES SPECIMEN
        num_images_per_batch = min(num_images, 8)  # Reduced for stability
        rays_per_batch = 32768  # Doubled from 16384 for denser sampling
        rays_per_image = rays_per_batch // num_images_per_batch  # Will be 8192 with batch=4
        train_image_size = [4160, 6240]  # Full resolution
        val_image_size = [2080, 3120]
    elif gpu_memory_gb >= 80:
        # A100/H100 class
        num_images_per_batch = min(num_images, 4, 8)
        rays_per_batch = 16384
        rays_per_image = rays_per_batch // num_images_per_batch
        train_image_size = [1560, 2340]
        val_image_size = [780, 1170]
    elif gpu_memory_gb >= 40:
        # A6000/A40 class  
        num_images_per_batch = min(num_images, 3, 5)
        rays_per_batch = 8192
        rays_per_image = rays_per_batch // num_images_per_batch
        train_image_size = [1040, 1560]
        val_image_size = [520, 780]
    else:
        # Consumer GPUs
        num_images_per_batch = min(num_images, 2, 3)
        rays_per_batch = 4096
        rays_per_image = rays_per_batch // num_images_per_batch
        train_image_size = [1040, 1560]
        val_image_size = [520, 780]
    
    adjusted_max_steps = max_steps
    
    return {
        'num_images_per_batch': num_images_per_batch,
        'rays_per_batch': rays_per_batch,
        'rays_per_image': rays_per_image,
        'train_image_size': train_image_size,
        'val_image_size': val_image_size,
        'adjusted_max_steps': adjusted_max_steps
    }


def create_neuralangelo_config(neuralangelo_dir: Path, output_dir: Path, neuralangelo_source: Path,
                              gpu_memory_gb: int, num_images: int, max_steps: int = 50000,
                              sphere_center: List[float] = None, sphere_radius: float = None,
                              config_template: Optional[Path] = None, use_depth: bool = True):
    """Create Neuralangelo configuration file with depth supervision support"""
    
    # Count actual images in the dataset
    ngp_images_dir = neuralangelo_dir / "images"
    image_count = len(list(ngp_images_dir.glob("*"))) if ngp_images_dir.exists() else num_images
    
    # Check if depth maps are available
    depth_dir = neuralangelo_dir / "depth_maps"
    has_depth = depth_dir.exists() and len(list(depth_dir.glob("*.npy"))) > 0
    
    if use_depth and has_depth:
        logger.info(f"✓ Depth maps found - will use depth supervision")
        # Use depth-enabled template if not specified
        if not config_template or "depth" not in str(config_template):
            # Try to find depth template
            depth_template = Path("/home/jhennessy7.gatech/augenblick/src/neuralangelo/projects/neuralangelo/configs/b200_depth_template.yaml")
            if depth_template.exists():
                config_template = depth_template
                logger.info(f"Using depth-enabled config template")
    elif use_depth and not has_depth:
        logger.warning("Depth supervision requested but no depth maps found")
    
    # Prepare sphere parameters
    if sphere_center is not None and sphere_radius is not None:
        sphere_center_list = [sphere_center[0], sphere_center[1], sphere_center[2]]
        sphere_scale = 1.0 / sphere_radius
    else:
        sphere_center_list = [0.0, 0.0, 0.0]
        sphere_scale = 1.0
    
    # Calculate iterations per epoch based on actual dataset
    params = calculate_training_params(gpu_memory_gb, num_images, max_steps)
    batch_size = params['num_images_per_batch']
    
    base_iterations = 50 #max(image_count // batch_size, 1)
    
    if gpu_memory_gb >= 160:
        # B200: many iterations per epoch
        num_iterations_per_epoch = base_iterations * 40
    else:
        # Other GPUs
        num_iterations_per_epoch = base_iterations * 20
    
    # Check if template is provided
    if not config_template:
        raise ValueError("Config template path must be provided with --config-template argument")
    
    if not config_template.exists():
        raise FileNotFoundError(f"Config template not found: {config_template}")
    
    # Load template
    logger.info(f"Using config template: {config_template}")
    with open(config_template, 'r') as f:
        template_content = f.read()
    
    # Replace placeholders using string substitution
    replacements = {
        '${data_root}': str(neuralangelo_dir),
        '${num_images}': str(image_count),
        '${sphere_center}': str(sphere_center_list),
        '${sphere_scale}': str(sphere_scale),
        '${max_iter}': str(max_steps),
        '${num_iterations_per_epoch}': str(num_iterations_per_epoch)
    }
    
    config_content = template_content
    for placeholder, value in replacements.items():
        config_content = config_content.replace(placeholder, value)
    
    # Parse to validate YAML
    try:
        config_dict = yaml.safe_load(config_content)
        
        # If depth is available and enabled, ensure depth supervision is configured
        if has_depth and use_depth:
            if 'model' in config_dict and 'depth_supervision' in config_dict['model']:
                config_dict['model']['depth_supervision']['enabled'] = True
                logger.info("✓ Depth supervision enabled in config")
        
        logger.info(f"YAML validation successful")
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML generated: {e}")
        raise
    
    # Write final config
    config_path = neuralangelo_dir / "config.yaml"
    with open(config_path, 'w') as f:
        # If we modified the dict, dump it as YAML
        if has_depth and use_depth and 'model' in config_dict:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        else:
            # Otherwise use the template string as-is
            f.write(config_content)
    
    # Also save to logs directory for reference
    logs_config_path = output_dir / "logs" / "config.yaml"
    logs_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(logs_config_path, 'w') as f:
        if has_depth and use_depth and 'model' in config_dict:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        else:
            f.write(config_content)
    
    # Log configuration details
    logger.info(f"Created config with:")
    if 'data' in config_dict and 'train' in config_dict['data']:
        logger.info(f"  - Batch size: {config_dict['data']['train']['batch_size']}")
    if 'model' in config_dict:
        logger.info(f"  - Ray samples: {config_dict['model'].get('n_rays', 'N/A')}")
    logger.info(f"  - Iterations per epoch: {num_iterations_per_epoch}")
    logger.info(f"  - Max iterations: {max_steps}")
    logger.info(f"  - Depth supervision: {has_depth and use_depth}")
    logger.info(f"  - Config saved to: {config_path}")
    logger.info(f"  - Template saved to: {output_dir}/logs/config.yaml")
    
    return max_steps


def step5_train_neuralangelo(neuralangelo_dir: Path, output_dir: Path, neuralangelo_source: Path,
                            gpu_index: int = 0, max_steps: int = 50000, 
                            config_template: Optional[Path] = None,
                            use_depth: bool = True) -> Optional[Path]:
    """Train Neuralangelo model with optional depth supervision"""
    logger.info("\n=== Step 5: Training Neuralangelo ===")
    if use_depth:
        logger.info("Depth supervision enabled")
    
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Check for existing checkpoints
    checkpoint_dir = logs_dir / "checkpoints"
    if checkpoint_dir.exists() and list(checkpoint_dir.glob("*.pth")):
        logger.info("Found existing checkpoints, skipping training...")
        checkpoints = sorted(checkpoint_dir.glob("*.pth"), key=os.path.getmtime)
        return checkpoints[-1]
    
    # Verify Neuralangelo source
    train_script = neuralangelo_source / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Neuralangelo train.py not found at: {train_script}")
    
    # Auto-detect GPU memory
    gpu_memory = detect_gpu_memory(gpu_index)
    
    # Read transforms to get image count and sphere parameters
    transforms_file = neuralangelo_dir / "transforms.json"
    with open(transforms_file, 'r') as f:
        transforms = json.load(f)
    num_images = len(transforms['frames'])
    
    # Extract sphere parameters from transforms.json
    sphere_center = transforms.get('sphere_center', None)
    sphere_radius = transforms.get('sphere_radius', None)
    
    # Create config with depth support
    adjusted_steps = create_neuralangelo_config(
        neuralangelo_dir, output_dir, neuralangelo_source,
        gpu_memory, num_images, max_steps,
        sphere_center, sphere_radius,
        config_template=config_template,
        use_depth=use_depth
    )
 
    # Run training
    original_cwd = os.getcwd()
    try:
        os.chdir(neuralangelo_source)
        
        # Use random port to avoid conflicts
        port = random.randint(29600, 29700)
        
        cmd = [
            "torchrun", "--nproc_per_node=1", "--master_port", str(port),
            "train.py",
            "--logdir", str(logs_dir),
            "--config", str(logs_dir / "config.yaml"),
            "--show_pbar"
        ]
        
        logger.info(f"Training Neuralangelo ({adjusted_steps} steps, depth={'enabled' if use_depth else 'disabled'})...")
        logger.info(f"Command: {' '.join(cmd)}")
        
        env = os.environ.copy()
        subprocess.run(cmd, check=True, env=env)

        logger.info("Training complete!")
        
        # Return latest checkpoint
        checkpoints = sorted(checkpoint_dir.glob("*.pth"), key=os.path.getmtime)
        return checkpoints[-1] if checkpoints else None
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        return None
    finally:
        os.chdir(original_cwd)


def step6_extract_mesh(checkpoint_path: Path, config_path: Path, output_path: Path,
                      neuralangelo_source: Path, resolution: int = 2048, 
                      block_res: int = 128) -> Optional[Path]:
    """Extract mesh from trained model"""
    logger.info("\n=== Step 6: Extracting Mesh ===")
    logger.info(f"Resolution: {resolution}, Block resolution: {block_res}")
    
    # Verify extraction script
    extract_script = neuralangelo_source / "projects" / "neuralangelo" / "scripts" / "extract_mesh.py"
    if not extract_script.exists():
        raise FileNotFoundError(f"Mesh extraction script not found at: {extract_script}")
    
    # Run extraction
    original_cwd = os.getcwd()
    try:
        os.chdir(neuralangelo_source)
        
        # Use random port
        port = random.randint(29700, 29800)
        
        cmd = [
            "torchrun", "--nproc_per_node=1", "--master_port", str(port),
            "projects/neuralangelo/scripts/extract_mesh.py",
            "--config", str(config_path),
            "--checkpoint", str(checkpoint_path),
            "--output_file", str(output_path),
            "--resolution", str(resolution),
            "--block_res", str(block_res)
        ]
        
        logger.info(f"Using checkpoint: {checkpoint_path.name}")
        logger.info("Extracting mesh...")
        env = os.environ.copy()
        subprocess.run(cmd, check=True, env=env)

        
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ Mesh saved to: {output_path}")
            logger.info(f"  File size: {file_size:.1f} MB")
            return output_path
        else:
            logger.error("Mesh file not created!")
            return None
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Mesh extraction failed: {e}")
        return None
    finally:
        os.chdir(original_cwd)


class MaskedReconstructionPipeline:
    """Complete pipeline for masked 3D reconstruction"""
    
    def __init__(self, input_dir: Path, output_dir: Path, 
                 vggt_path: Path = Path("/home/jhennessy7.gatech/augenblick/src/vggt"),
                 neuralangelo_path: Path = Path("/home/jhennessy7.gatech/augenblick/src/neuralangelo"),
                 gpu_index: int = 0,
                 vggt_script_path: Path = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.vggt_path = Path(vggt_path)
        self.neuralangelo_path = Path(neuralangelo_path)
        self.gpu_index = gpu_index
        self.vggt_script_path = vggt_script_path or (Path.home() / "augenblick/src/vggt/your_vggt_script.py")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_full_pipeline(self, skip_colmap: bool = False, skip_vggt: bool = False,
                         max_steps: int = 50000, mesh_resolution: int = 2048,
                         block_res: int = 128, use_module_colmap: bool = False,
                         use_vggt_script: bool = True, config_template: Optional[Path] = None,
                         use_depth: bool = True):
        """Run the complete reconstruction pipeline with depth map support"""
        start_time = time.time()
        
        logger.info("="*60)
        logger.info("Masked 3D Reconstruction Pipeline with VGGT and Depth Maps")
        logger.info("="*60)
        logger.info(f"Input: {self.input_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"GPU: {self.gpu_index}")
        logger.info(f"Skip COLMAP: {skip_colmap}")
        logger.info(f"Skip VGGT: {skip_vggt}")
        logger.info(f"Use VGGT Script: {use_vggt_script}")
        logger.info(f"Use Depth Maps: {use_depth}")
        logger.info(f"Max steps: {max_steps}")
        logger.info(f"Mesh resolution: {mesh_resolution}")
        
        try:
            # Step 1: Organize dataset
            logger.info("\n=== Step 1: Organizing Dataset ===")
            images_dir, masks_dir, image_count, mask_count = organize_dataset(
                self.input_dir, self.output_dir
            )
            
            if image_count == 0:
                raise RuntimeError("No images found in dataset")
            
            # Step 2: Run VGGT (optional)
            vggt_output_dir = None
            vggt_sparse_dir = None  # Initialize this variable
            
            if not skip_vggt:
                if use_vggt_script and self.vggt_script_path.exists():
                    # Use the direct VGGT script that outputs Neuralangelo format
                    try:
                        # Clear GPU memory before running VGGT
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            gc.collect()
                            logger.info(f"Cleared GPU memory before VGGT")
                        
                        # Use the depth-aware VGGT function
                        vggt_output_dir = step2_run_vggt_direct(
                            images_dir, self.output_dir, self.vggt_script_path,
                            f"cuda:{self.gpu_index}"
                        )
                        
                    except Exception as e:
                        logger.error(f"VGGT script failed: {e}")
                        if "out of memory" in str(e).lower():
                            logger.error("GPU out of memory. Try:")
                            logger.error("1. Reducing batch size in VGGT script")
                            logger.error("2. Using a different GPU with more memory")
                            logger.error("3. Closing other GPU processes")
                        vggt_output_dir = None
                    
                    if vggt_output_dir:
                        # Skip COLMAP and conversion steps - we already have Neuralangelo format
                        logger.info("VGGT script produced Neuralangelo format directly")
                        
                        # Copy masks if available
                        if mask_count > 0:
                            ngp_masks_dir = vggt_output_dir / "masks"
                            ngp_masks_dir.mkdir(exist_ok=True)
                            for mask_file in masks_dir.glob("*.png"):
                                shutil.copy2(mask_file, ngp_masks_dir / mask_file.name)
                            logger.info(f"Copied {mask_count} masks to Neuralangelo directory")
                        
                        # Verify depth maps if depth usage is enabled
                        depth_available = False
                        if use_depth:
                            depth_available = verify_depth_maps(vggt_output_dir)
                            if not depth_available:
                                logger.warning("Depth maps not available or invalid - continuing without depth supervision")
                        
                        # Use this as our neuralangelo_dir
                        neuralangelo_dir = vggt_output_dir
                        
                        # Determine config template
                        if use_depth and config_template is None:
                            # Use depth-enabled template
                            config_template = Path("/home/jhennessy7.gatech/augenblick/src/neuralangelo/projects/neuralangelo/configs/b200_depth_template.yaml")
                            if not config_template.exists():
                                logger.warning("Depth template not found, using standard template")
                                config_template = Path("/home/jhennessy7.gatech/augenblick/src/neuralangelo/projects/neuralangelo/configs/b200_template.yaml")
                        
                        # Skip to training
                        checkpoint = step5_train_neuralangelo(
                            neuralangelo_dir, self.output_dir, self.neuralangelo_path,
                            self.gpu_index, max_steps, config_template=config_template,
                            use_depth=use_depth
                        )
                        
                        if not checkpoint:
                            raise RuntimeError("Training failed to produce checkpoint")
                        
                        # Extract mesh
                        mesh_path = self.output_dir / "final_mesh.ply"
                        config_path = self.output_dir / "logs" / "config.yaml"
                        
                        extracted_mesh = step6_extract_mesh(
                            checkpoint, config_path, mesh_path,
                            self.neuralangelo_path, mesh_resolution, block_res
                        )
                        
                        if not extracted_mesh:
                            raise RuntimeError("Mesh extraction failed")
                        
                        # Success!
                        elapsed_time = time.time() - start_time
                        logger.info("\n" + "="*60)
                        logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY!")
                        logger.info("="*60)
                        logger.info(f"Total time: {elapsed_time/3600:.1f} hours")
                        logger.info(f"Output directory: {self.output_dir}")
                        logger.info(f"Final mesh: {mesh_path}")
                        logger.info(f"Depth supervision used: {use_depth and depth_available}")
                        return
                else:
                    # Fall back to original VGGT approach
                    vggt_sparse_dir = step2_run_vggt(
                        images_dir, self.output_dir, self.vggt_path,
                        f"cuda:{self.gpu_index}"
                    )
                    if not vggt_sparse_dir:
                        logger.warning("VGGT step failed, but continuing...")
            
            # Step 3: Run COLMAP (optional) - only if we didn't use VGGT script
            if not skip_colmap and not vggt_output_dir:
                colmap_sparse_dir = step3_run_colmap(
                    images_dir, self.output_dir, vggt_sparse_dir,
                    use_module=use_module_colmap
                )
                if not colmap_sparse_dir:
                    raise RuntimeError("COLMAP failed to produce output")
            else:
                logger.info("\n=== Step 3: Skipping COLMAP (--skip-colmap flag set or VGGT script used) ===")
                if vggt_sparse_dir:
                    # Use VGGT output directly
                    colmap_sparse_dir = vggt_sparse_dir
                else:
                    logger.error("No VGGT output found and COLMAP is skipped!")
                    logger.error("Either run VGGT first or enable COLMAP")
                    raise RuntimeError("Camera pose estimation failed (neither VGGT nor COLMAP produced output)")
            
            # Step 4: Convert to Neuralangelo format (if not using VGGT script)
            if not vggt_output_dir:
                neuralangelo_dir = step4_convert_to_neuralangelo(
                    colmap_sparse_dir, images_dir, masks_dir, self.output_dir
                )
            
            # Step 5: Train Neuralangelo
            checkpoint = step5_train_neuralangelo(
                neuralangelo_dir, self.output_dir, self.neuralangelo_path,
                self.gpu_index, max_steps, config_template=config_template,
                use_depth=use_depth
            )
            
            if not checkpoint:
                raise RuntimeError("Training failed to produce checkpoint")
            
            # Step 6: Extract mesh
            mesh_path = self.output_dir / "final_mesh.ply"
            config_path = self.output_dir / "logs" / "config.yaml"
            
            extracted_mesh = step6_extract_mesh(
                checkpoint, config_path, mesh_path,
                self.neuralangelo_path, mesh_resolution, block_res
            )
            
            if not extracted_mesh:
                raise RuntimeError("Mesh extraction failed")
            
            # Success!
            elapsed_time = time.time() - start_time
            logger.info("\n" + "="*60)
            logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            logger.info(f"Total time: {elapsed_time/3600:.1f} hours")
            logger.info(f"Output directory: {self.output_dir}")
            logger.info(f"Final mesh: {mesh_path}")
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"\n✗ Pipeline failed: {e}")
            logger.error(f"Failed after {elapsed_time/60:.1f} minutes")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Complete masked 3D reconstruction pipeline with VGGT and depth supervision"
    )
    parser.add_argument("input_dir", help="Directory with images and masks")
    parser.add_argument("output_dir", help="Output directory for all results")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--max_steps", type=int, default=50000, help="Training steps")
    parser.add_argument("--mesh_resolution", type=int, default=2048, help="Mesh resolution")
    parser.add_argument(
        "--block_res", type=int, default=128, 
        help="Block resolution for mesh extraction (default: 128, use 256 for powerful GPUs)"
    )
    parser.add_argument(
        "--skip-colmap", action="store_true", 
        help="Skip COLMAP step and use VGGT output directly"
    )
    parser.add_argument(
        "--skip-vggt", action="store_true",
        help="Skip VGGT step and use COLMAP only"
    )
    parser.add_argument(
        "--vggt-path", type=str, default="/home/jhennessy7.gatech/augenblick/src/vggt",
        help="Path to VGGT source code"
    )
    parser.add_argument(
        "--vggt-script", type=str, default=None,
        help="Path to VGGT script that outputs Neuralangelo format directly (e.g., your_vggt_script.py)"
    )
    parser.add_argument(
        "--neuralangelo-path", type=str, default="/home/jhennessy7.gatech/augenblick/src/neuralangelo",
        help="Path to Neuralangelo source code"
    )
    parser.add_argument(
        "--use-module-colmap", action="store_true",
        help="Use 'module load colmap' instead of system colmap"
    )
    parser.add_argument(
        "--use-vggt-script", action="store_true",
        help="Use VGGT script that outputs Neuralangelo format directly (recommended for turntable)"
    )
    parser.add_argument(
        "--config-template", type=str, default=None,
        help="Path to YAML config template (e.g., configs/b200_template.yaml)"
    )
    parser.add_argument(
        "--use-depth", action="store_true", default=True,
        help="Use depth maps from VGGT for supervision during training (default: True)"
    )
    parser.add_argument(
        "--no-depth", action="store_true",
        help="Disable depth supervision even if depth maps are available"
    )
    parser.add_argument(
        "--depth-weight", type=float, default=0.1,
        help="Weight for depth loss (default: 0.1)"
    )
 
    args = parser.parse_args()
    
    # Handle depth flag logic
    use_depth = args.use_depth and not args.no_depth
    
    # Auto-detect high-resolution mode
    if args.mesh_resolution >= 4096 and args.block_res == 128:
        logger.info(f"High resolution mode detected (resolution={args.mesh_resolution})")
        logger.info("Consider using --block_res 256 for better performance on B200 GPU")
    
    # Default VGGT script path if not provided
    if args.vggt_script is None:
        # Try common locations
        possible_paths = [
            Path.home() / "augenblick/src/vggt/your_vggt_script.py",
            Path("/blue/arthur.porto-biocosmos/jhennessy7.gatech/scratch/vggt/your_vggt_script.py"),
            Path("./vggt/your_vggt_script.py"),
        ]
        for path in possible_paths:
            if path.exists():
                args.vggt_script = str(path)
                logger.info(f"Auto-detected VGGT script at: {path}")
                break
    
    # Auto-select config template based on depth usage
    if args.config_template is None:
        if use_depth:
            config_template = Path("/home/jhennessy7.gatech/augenblick/src/neuralangelo/projects/neuralangelo/configs/b200_depth_template.yaml")
            if not config_template.exists():
                logger.warning("Depth template not found, using standard template")
                config_template = Path("/home/jhennessy7.gatech/augenblick/src/neuralangelo/projects/neuralangelo/configs/b200_template.yaml")
            else:
                logger.info("Using depth-enabled config template")
        else:
            config_template = Path("/home/jhennessy7.gatech/augenblick/src/neuralangelo/projects/neuralangelo/configs/b200_template.yaml")
            logger.info("Using standard config template")
    else:
        config_template = Path(args.config_template)
    
    # Create pipeline
    pipeline = MaskedReconstructionPipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        vggt_path=Path(args.vggt_path),
        neuralangelo_path=Path(args.neuralangelo_path),
        gpu_index=args.gpu,
        vggt_script_path=Path(args.vggt_script) if args.vggt_script else None
    )
    
    # Run pipeline with depth support
    pipeline.run_full_pipeline(
        skip_colmap=args.skip_colmap,
        skip_vggt=args.skip_vggt,
        max_steps=args.max_steps,
        mesh_resolution=args.mesh_resolution,
        block_res=args.block_res,
        use_module_colmap=args.use_module_colmap,
        use_vggt_script=args.use_vggt_script or (args.vggt_script is not None),
        config_template=config_template,
        use_depth=use_depth
    )


if __name__ == "__main__":
    main()
