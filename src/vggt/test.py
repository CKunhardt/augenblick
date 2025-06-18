# Adapted from by demo_gradio.py from the VGGT repository
# Author: Clinton T. Kunhardt

import os
import cv2
import torch
import numpy as np
import sys
import glob
import time
import open3d as o3d

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing and loading VGGT model...")
# model = VGGT.from_pretrained("facebook/VGGT-1B")  # another way to load the model
startTime = time.time()
model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))


model.eval()
model = model.to(device)
endTime = time.time()
print(f"Model loaded in {endTime - startTime:.2f} seconds")

def run_model(target_dir, model) -> dict:
    """
    Run the VGGT model on images in the 'target_dir/images' folder and return predictions.
    """
    print(f"Processing images from {target_dir}")
    startTimeAll = time.time()
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
    startTime = time.time()
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    endTime = time.time()
    print(f"Inference completed in {endTime - startTime:.2f} seconds")

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    startTime = time.time()
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    endTime = time.time()
    print(f"Pose encoding conversion completed in {endTime - startTime:.2f} seconds")

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension

    # Generate world points from depth map
    print("Computing world points from depth map...")
    startTime = time.time()
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points
    endTime = time.time()
    print(f"World points computed in {endTime - startTime:.2f} seconds")
    print(f"Total processing time: {endTime - startTimeAll:.2f} seconds")
    # Clean up
    torch.cuda.empty_cache()
    return predictions


print("Starting VGGT model inference...")
target_dir = "C:\\repos\\gatech\\photogrammetry\\south-building"  # Change this to your target directory
predictions = run_model(target_dir, model)

# Save predictions to GLB file
glb_path = os.path.join(target_dir, "predictions.glb")
print(f"Saving predictions to {glb_path}")
predictions_to_glb(predictions, glb_path)

# Optionally visualize the point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(predictions["world_points_from_depth"].reshape(-1, 3))
o3d.visualization.draw_geometries([pcd])