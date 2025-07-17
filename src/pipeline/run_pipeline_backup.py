# Main script to run the VGGT model pipeline with batch processing
# Modified to process images in batches to avoid OOM errors
# Now saves predictions for NeuS2 conversion
# Author: Modified for batch processing

import os
import torch
import numpy as np
import sys
import glob
import time
import gc
import pickle
import argparse

# Fix for CUDA allocation error
# Remove problematic PYTORCH_CUDA_ALLOC_CONF settings
if 'PYTORCH_CUDA_ALLOC_CONF' in os.environ:
    current_conf = os.environ['PYTORCH_CUDA_ALLOC_CONF']
    if 'max_split_size_mb' in current_conf:
        # Remove the problematic option
        parts = current_conf.split(',')
        new_parts = [p for p in parts if 'max_split_size_mb' not in p]
        if new_parts:
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = ','.join(new_parts)
        else:
            del os.environ['PYTORCH_CUDA_ALLOC_CONF']

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing and loading VGGT model...")

model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

model.eval()
model = model.to(device)

def run_model_batched(target_dir, model, batch_size=4):
    """
    Modified version of run_model that processes images in batches
    """
    print(f"Processing images from {target_dir} in batches of {batch_size}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load image names
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    # Initialize storage for all predictions
    all_predictions = None
    all_images_raw = []
    
    # Process in batches
    num_batches = (len(image_names) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_names))
        batch_image_names = image_names[start_idx:end_idx]
        
        print(f"Processing batch {batch_idx + 1}/{num_batches} (images {start_idx + 1}-{end_idx})")
        
        # Log individual image names being processed
        for idx, img_name in enumerate(batch_image_names, start=start_idx + 1):
            print(f"  Loading image {idx}: {os.path.basename(img_name)}")
        
        # Load and preprocess batch
        images = load_and_preprocess_images(batch_image_names).to(device)
        print(f"Batch shape: {images.shape}")
        
        # Store raw images for NeuS2
        all_images_raw.append(images.cpu())
        
        # Run inference
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                batch_predictions = model(images)
        
        # Debug model output
        print(f"  Model output shapes:")
        for key, value in batch_predictions.items():
            print(f"    {key}: {value.shape}")
        
        # Convert pose encoding to extrinsic and intrinsic matrices
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            batch_predictions["pose_enc"], 
            images.shape[-2:]
        )
        batch_predictions["extrinsic"] = extrinsic
        batch_predictions["intrinsic"] = intrinsic
        
        # Move to CPU immediately to free GPU memory
        for key in batch_predictions.keys():
            batch_predictions[key] = batch_predictions[key].cpu()
        
        # Accumulate predictions
        if all_predictions is None:
            all_predictions = {}
            for key in batch_predictions.keys():
                all_predictions[key] = [batch_predictions[key]]
        else:
            for key in batch_predictions.keys():
                all_predictions[key].append(batch_predictions[key])
        
        # Clean up GPU memory
        del images, batch_predictions
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Batch {batch_idx + 1} complete. GPU memory cleared.")
    
    # Concatenate all batches
    print("Concatenating all batch results...")
    for key in all_predictions.keys():
        print(f"  Concatenating {key}:")
        print(f"    Number of batches: {len(all_predictions[key])}")
        for i, batch in enumerate(all_predictions[key]):
            print(f"    Batch {i} shape: {batch.shape}")
        
        all_predictions[key] = torch.cat(all_predictions[key], dim=0)
        print(f"    Concatenated shape: {all_predictions[key].shape}")
        
        # Convert to numpy (don't squeeze - we have multiple images)
        if isinstance(all_predictions[key], torch.Tensor):
            all_predictions[key] = all_predictions[key].numpy()
            print(f"    Final shape after numpy conversion: {all_predictions[key].shape}")
    
    # Concatenate raw images
    all_images_raw = torch.cat(all_images_raw, dim=0)
    
    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = all_predictions["depth"]  # Should be (S, H, W) or (S, H, W, 1)
    extrinsics = all_predictions["extrinsic"]  # (S, 4, 4)
    intrinsics = all_predictions["intrinsic"]  # (S, 3, 3)
    
    print(f"Initial depth map shape: {depth_map.shape}")
    print(f"Extrinsic shape: {extrinsics.shape}")
    print(f"Intrinsic shape: {intrinsics.shape}")
    
    # Handle the specific case where depth might be (S, C, H, W) format
    if depth_map.ndim == 4:
        # Check if it's channel-first format (S, C, H, W)
        if depth_map.shape[1] in [1, 3, 4] and depth_map.shape[2] > 10 and depth_map.shape[3] > 10:
            print(f"Detected channel-first format (S, C, H, W)")
            if depth_map.shape[1] == 1:
                # Single channel depth
                depth_map = depth_map.squeeze(1)  # (S, H, W)
            else:
                # Multi-channel depth - take first channel as depth
                print(f"Warning: Multi-channel depth map with {depth_map.shape[1]} channels, using first channel")
                depth_map = depth_map[:, 0, :, :]  # (S, H, W)
            print(f"Converted to shape: {depth_map.shape}")
        # Check if it's channel-last format (S, H, W, C)
        elif depth_map.shape[-1] in [1, 3, 4] and depth_map.shape[1] > 10 and depth_map.shape[2] > 10:
            print(f"Detected channel-last format (S, H, W, C)")
            if depth_map.shape[-1] == 1:
                depth_map = depth_map.squeeze(-1)  # (S, H, W)
            else:
                # Multi-channel depth - take first channel
                print(f"Warning: Multi-channel depth map with {depth_map.shape[-1]} channels, using first channel")
                depth_map = depth_map[..., 0]  # (S, H, W)
            print(f"Converted to shape: {depth_map.shape}")
    elif depth_map.ndim == 3:
        # Check if it's actually (C, H, W) for a single image
        if depth_map.shape[0] in [1, 3, 4] and depth_map.shape[1] > 10 and depth_map.shape[2] > 10:
            print(f"Detected single image in channel-first format (C, H, W)")
            if depth_map.shape[0] == 1:
                depth_map = depth_map[0]  # (H, W)
            else:
                print(f"Warning: Multi-channel single depth map with {depth_map.shape[0]} channels, using first channel")
                depth_map = depth_map[0]  # (H, W)
            depth_map = depth_map[np.newaxis, ...]  # Add batch dimension back
            print(f"Converted to shape: {depth_map.shape}")
    elif depth_map.ndim == 2:
        # Single (H, W) depth map
        depth_map = depth_map[np.newaxis, ...]  # Add batch dimension
        print(f"Added batch dimension, new shape: {depth_map.shape}")
    
    # Final shape check
    if depth_map.ndim != 3:
        raise ValueError(f"After processing, depth map should be 3D (S, H, W), got shape {depth_map.shape}")
    
    # Convert extrinsics from 4x4 to 3x4 (remove last row)
    extrinsics_3x4 = extrinsics[:, :3, :]  # (S, 3, 4)
    print(f"Extrinsic 3x4 shape: {extrinsics_3x4.shape}")
    
    # The function handles batches internally, so we can pass all at once
    world_points = unproject_depth_map_to_point_map(
        depth_map, 
        extrinsics_3x4, 
        intrinsics
    )
    
    all_predictions["world_points_from_depth"] = world_points
    print(f"Final world_points_from_depth shape: {world_points.shape}")
    
    # Store raw images in predictions for save_predictions_for_neus2
    all_predictions["_raw_images"] = all_images_raw
    
    # Final cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    return all_predictions

def save_predictions_for_neus2(predictions, output_dir, input_dir):
    """
    Save VGG-T predictions in format compatible with NeuS2 converter.
    Modified to handle raw images from predictions dict.
    """
    print("Preparing predictions for NeuS2 conversion...")
    
    # Extract raw images if available
    images_raw = predictions.pop("_raw_images", None)
    
    # Create the predictions dictionary in the format expected by the converter
    neus2_predictions = {}
    
    # 1. World points - use both pointmap and depth-based points
    if "world_points" in predictions:
        neus2_predictions["world_points"] = predictions["world_points"]
        print(f"Added world_points: {predictions['world_points'].shape}")
    
    if "world_points_from_depth" in predictions:
        neus2_predictions["world_points_from_depth"] = predictions["world_points_from_depth"]
        print(f"Added world_points_from_depth: {predictions['world_points_from_depth'].shape}")
    
    # 2. Confidence scores
    if "world_points_conf" in predictions:
        neus2_predictions["world_points_conf"] = predictions["world_points_conf"]
        print(f"Added world_points_conf: {predictions['world_points_conf'].shape}")
    elif "depth_conf" in predictions:
        neus2_predictions["depth_conf"] = predictions["depth_conf"]
        print(f"Added depth_conf: {predictions['depth_conf'].shape}")
    else:
        # Create uniform confidence if not available
        if "world_points" in predictions:
            conf_shape = predictions["world_points"].shape[:-1]  # Remove last dimension (xyz)
            neus2_predictions["world_points_conf"] = np.ones(conf_shape, dtype=np.float32)
            print(f"Created uniform confidence: {conf_shape}")
    
    # 3. Images - convert back to uint8 format for NeuS2
    if images_raw is not None:
        # Assume images_raw is preprocessed tensor (S, C, H, W) normalized to [-1, 1] or [0, 1]
        images_np = images_raw.numpy() if isinstance(images_raw, torch.Tensor) else images_raw
        
        # Convert from (S, C, H, W) to (S, H, W, C)
        if images_np.ndim == 4 and images_np.shape[1] == 3:
            images_np = np.transpose(images_np, (0, 2, 3, 1))
        
        # Denormalize to [0, 255] range
        if images_np.max() <= 1.0:
            # Assuming normalized to [0, 1]
            images_np = (images_np * 255).astype(np.uint8)
        elif images_np.min() >= -1.0 and images_np.max() <= 1.0:
            # Assuming normalized to [-1, 1]
            images_np = ((images_np + 1) * 127.5).astype(np.uint8)
        else:
            # Already in reasonable range
            images_np = np.clip(images_np, 0, 255).astype(np.uint8)
        
        neus2_predictions["images"] = images_np
        print(f"Added images: {images_np.shape}")
    
    # 4. Camera matrices
    neus2_predictions["extrinsic"] = predictions["extrinsic"]
    neus2_predictions["intrinsic"] = predictions["intrinsic"]
    print(f"Added extrinsic: {predictions['extrinsic'].shape}")
    print(f"Added intrinsic: {predictions['intrinsic'].shape}")
    
    # 5. Additional useful data
    if "depth" in predictions:
        neus2_predictions["depth"] = predictions["depth"]
        print(f"Added depth: {predictions['depth'].shape}")
    
    if "pose_enc" in predictions:
        neus2_predictions["pose_enc"] = predictions["pose_enc"]
        print(f"Added pose_enc: {predictions['pose_enc'].shape}")
    
    # 6. Save metadata
    neus2_predictions["metadata"] = {
        "source": "VGGT",
        "input_dir": input_dir,
        "timestamp": time.time(),
        "num_frames": neus2_predictions["extrinsic"].shape[0],
        "image_size": [neus2_predictions["images"].shape[2], neus2_predictions["images"].shape[1]]  # [W, H]
    }
    
    # Save to pickle file
    timestamp = int(time.time())
    predictions_file = os.path.join(output_dir, f"vggt_predictions_{timestamp}.pkl")
    
    with open(predictions_file, 'wb') as f:
        pickle.dump(neus2_predictions, f)
    
    print(f"✅ Saved VGG-T predictions to: {predictions_file}")
    print(f"📊 Prediction summary:")
    for key, value in neus2_predictions.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape} ({value.dtype})")
        elif key == "metadata":
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {type(value)}")
    
    return predictions_file

def parse_arguments():
    """Parse command line arguments for input and output directories"""
    parser = argparse.ArgumentParser(
        description="Run VGGT photogrammetry pipeline on a directory of images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input_dir",
        type=str,
        help="Input directory containing an 'images' subdirectory with photos"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for generated GLB file (defaults to input_dir)"
    )
    
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=50.0,
        help="Confidence threshold for GLB generation"
    )
    
    parser.add_argument(
        "--prediction_mode",
        type=str,
        default="Depthmap and Camera Branch",
        choices=["Depthmap and Camera Branch", "Points Only", "Cameras Only"],
        help="Prediction mode for GLB generation"
    )
    
    parser.add_argument(
        "--save_for_neus2",
        action="store_true",
        help="Save predictions in format suitable for NeuS2 conversion"
    )
    
    parser.add_argument(
        "--skip_glb",
        action="store_true", 
        help="Skip GLB generation (useful when only saving predictions)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of images to process at once (reduce if OOM)"
    )
    
    return parser.parse_args()

def validate_input_directory(input_dir):
    """Validate that input directory exists and contains images subdirectory"""
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    images_dir = os.path.join(input_dir, "images")
    if not os.path.exists(images_dir):
        raise ValueError(f"Input directory must contain an 'images' subdirectory: {images_dir}")
    
    # Check if images directory has any files
    image_files = glob.glob(os.path.join(images_dir, "*"))
    if len(image_files) == 0:
        raise ValueError(f"No images found in: {images_dir}")
    
    print(f"Found {len(image_files)} files in images directory")
    return True

def main():
    """Main function with command line argument handling"""
    args = parse_arguments()
    
    # Validate input directory
    try:
        validate_input_directory(args.input_dir)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Confidence threshold: {args.conf_thres}")
    print(f"Prediction mode: {args.prediction_mode}")
    print(f"Save for NeuS2: {args.save_for_neus2}")
    print(f"Skip GLB: {args.skip_glb}")
    print(f"Batch size: {args.batch_size}")
    
    try:
        start_time = time.time()
        
        # Run the model with batching
        predictions = run_model_batched(args.input_dir, model, batch_size=args.batch_size)
        total_time = time.time() - start_time
        
        print(f"SUCCESS! Processing took {total_time:.2f} seconds")
        
        # Save predictions for NeuS2 conversion
        if args.save_for_neus2:
            predictions_file = save_predictions_for_neus2(
                predictions, 
                output_dir, 
                args.input_dir
            )
            
            # Print conversion command
            print("\n" + "="*60)
            print("🎯 Ready for NeuS2 conversion!")
            print("="*60)
            print("Run this command to convert to NeuS2 format:")
            print()
            print(f"python vggt_to_neus2_converter.py \\")
            print(f"    {predictions_file} \\")
            print(f"    ./neus2_data \\")
            print(f"    --images_dir {args.input_dir}/images")
            print()
            print("="*60)
        
        # Generate GLB (optional)
        if not args.skip_glb:
            # Generate output filename with timestamp
            timestamp = int(time.time())
            glb_filename = f"vggt_output_{timestamp}.glb"
            glb_path = os.path.join(output_dir, glb_filename)
            
            print(f"Saving GLB to {glb_path}")
            
            scene = predictions_to_glb(
                predictions, 
                conf_thres=args.conf_thres,
                target_dir=args.input_dir,
                prediction_mode=args.prediction_mode
            )
            scene.export(glb_path)
            
            print(f"Successfully saved GLB to: {glb_path}")
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
