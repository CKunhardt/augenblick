import os
import cv2
import numpy as np
from read_write_model import read_cameras_binary, read_images_binary
from tqdm import tqdm
# Path to COLMAP outputs
colmap_path = "data/bird_ivory_2/"  # e.g., './data/scene'
input_dir = os.path.join(colmap_path, "input")  # distorted images
output_dir = os.path.join(colmap_path, "images")  # undistorted images
sparse_dir = os.path.join(colmap_path, "sparse", "0")
os.makedirs(output_dir, exist_ok=True)

# Load cameras and image metadata
cameras = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
images = read_images_binary(os.path.join(sparse_dir, "images.bin"))

# Loop over each image
for image_id, image in tqdm(images.items(), desc="Undistorting images"):
    cam = cameras[image.camera_id]
    
    assert cam.model == 'PINHOLE', "Camera model must be PINHOLE"

    fx, fy, cx, cy = cam.params
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])
    
    # No distortion for PINHOLE, but we assume original images are distorted
    dist_coeffs = np.array([0, 0, 0, 0])  # no distortion

    image_path = os.path.join(input_dir, image.name)
    img = cv2.imread(image_path)

    undistorted = cv2.undistort(img, K, dist_coeffs, None, K)

    out_path = os.path.join(output_dir, image.name)
    cv2.imwrite(out_path, undistorted)

print("Done: All images copied/undistorted into", output_dir)
