# BYTE-FOR-BYTE copy of demo_gradio.py approach
# Load model globally like demo, use exact same function structure

import os
import torch
import numpy as np
import sys
import glob
import time
import gc

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"

# EXACT same global model loading as demo_gradio.py
print("Initializing and loading VGGT model...")
# model = VGGT.from_pretrained("facebook/VGGT-1B")  # another way to load the model

model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

model.eval()
model = model.to(device)

def run_model(target_dir, model) -> dict:
    """
    EXACT BYTE-FOR-BYTE copy of demo_gradio.py run_model function
    """
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    # Clean up
    torch.cuda.empty_cache()
    return predictions

def main():
    """Simple main that just calls the exact demo function"""
    target_dir = "C:\\repos\\gatech\\photogrammetry\\buddha"
    
    try:
        start_time = time.time()
        predictions = run_model(target_dir, model)  # Use global model
        total_time = time.time() - start_time
        
        print(f"SUCCESS! Processing took {total_time:.2f} seconds")
        
        # Save GLB - Use correct prediction mode like demo
        glb_path = os.path.join(target_dir, f"correct_prediction_mode_{int(time.time())}.glb")
        print(f"Saving GLB to {glb_path}")
        
        scene = predictions_to_glb(
            predictions, 
            conf_thres=50.0,  # Use demo default of 50 instead of 30
            target_dir=target_dir,
            prediction_mode="Depthmap and Camera Branch"  # Match demo_gradio.py
        )
        scene.export(glb_path)
        
        print(f"Saved to: {glb_path}")
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()