import os
import cv2
import numpy as np
from read_write_model import read_cameras_binary, read_images_binary, read_points3D_binary
from pathlib import Path
from tqdm import tqdm

# === Paths ===
colmap_sparse_dir = "data/UF_mammals_36342_skull/noscale/sparse/0"
undistorted_image_dir = "data/UF_mammals_36342_skull/noscale/images"
save_output_dir = "data/UF_mammals_36342_skull/noscale/visualized"

Path(save_output_dir).mkdir(parents=True, exist_ok=True)

# === Load COLMAP metadata ===
cameras = read_cameras_binary(os.path.join(colmap_sparse_dir, "cameras.bin"))
images = read_images_binary(os.path.join(colmap_sparse_dir, "images.bin"))
points3D = read_points3D_binary(os.path.join(colmap_sparse_dir, "points3D.bin"))

# === Helper ===
def qvec2rotmat(qvec):
    q = qvec / np.linalg.norm(qvec)
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,       1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,       2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
    ])

# === Process ===
for img_id, img in tqdm(images.items(), desc="Reprojecting 3D points"):
    cam = cameras[img.camera_id]

    if cam.model != "PINHOLE":
        print(f"Unsupported camera model: {cam.model}")
        continue

    fx, fy, cx, cy = cam.params
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    R = qvec2rotmat(img.qvec)
    t = img.tvec.reshape(3, 1)

    img_path = os.path.join(undistorted_image_dir, img.name)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Image not found: {img_path}")
        continue

    for pid, pt in points3D.items():
        X = pt.xyz.reshape(3, 1)
        X_cam = R @ X + t
        if X_cam[2] <= 0:
            continue  # Skip behind-camera points
        x_proj = K @ X_cam
        x_proj = (x_proj[:2] / x_proj[2]).flatten().astype(int)

        x, y = x_proj
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    out_path = os.path.join(save_output_dir, f"aligned_{os.path.basename(img.name)}")
    cv2.imwrite(out_path, image)

print("Done.")
