import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for headless environment
import matplotlib.pyplot as plt
import argparse
import open3d as o3d
import traceback
from sugar_scene.gs_model import GaussianSplattingWrapper
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor
)

def parse_args():
    parser = argparse.ArgumentParser(description='Render VGGT sparse reconstruction from PLY files')
    
    # Basic parameters
    parser.add_argument('--source_path', type=str, required=True,
                        help='Path to the scene data')
    parser.add_argument('--gs_checkpoint_path', type=str, required=True,
                        help='Path to the vanilla Gaussian Splatting checkpoint')
    parser.add_argument('--sparse_ply_path', type=str, required=True,
                        help='Path to the sparse point cloud PLY file')
    
    # Camera & rendering options
    parser.add_argument('--camera_index', type=int, default=-1,
                        help='Camera index to render (-1 for random)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index')
    parser.add_argument('--plot_ratio', type=float, default=2.0,
                        help='Size ratio for output images')
    parser.add_argument('--point_size', type=float, default=0.01,
                        help='Size of points in the rendering')
    parser.add_argument('--output_dir', type=str, default='./renders',
                        help='Directory to save rendered images')
    parser.add_argument('--eval_split', action='store_true',
                        help='Use evaluation split')
    parser.add_argument('--skip_images', type=int, default=8,
                        help='Number of images to skip for eval split')
    parser.add_argument('--render_multi_view', action='store_true',
                        help='Render from multiple camera views')
    parser.add_argument('--num_views', type=int, default=5,
                        help='Number of views to render (only if render_multi_view is set)')
    
    return parser.parse_args()

def render_point_cloud(point_cloud_path, cameras, camera_idx, device, point_radius=0.01):
    """Render a point cloud from a specific camera view"""
    # Load the point cloud
    print(f"Loading point cloud from {point_cloud_path}...")
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    
    # Convert to torch tensors
    points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device=device)
    print(f"Point cloud contains {len(points)} points")
    
    # Handle colors - if the point cloud has colors, use them
    if pcd.has_colors():
        print("Point cloud has color information")
        colors = torch.tensor(np.asarray(pcd.colors), dtype=torch.float32, device=device)
    else:
        # Generate colors based on position (for visualization)
        print("Point cloud has no color information, generating colors based on position")
        points_np = points.cpu().numpy()
        min_vals = points_np.min(axis=0)
        max_vals = points_np.max(axis=0)
        normalized_points = (points_np - min_vals) / (max_vals - min_vals + 1e-6)
        colors = torch.tensor(normalized_points, dtype=torch.float32, device=device)
    
    # Create a Pointclouds object
    point_cloud = Pointclouds(points=[points], features=[colors])
    
    # Get camera dimensions from the camera object
    # These lines need to be fixed:
    camera_width = cameras.gs_cameras[camera_idx].image_width
    camera_height = cameras.gs_cameras[camera_idx].image_height
    
    # Point cloud rendering settings with correct image size
    raster_settings = PointsRasterizationSettings(
        image_size=(camera_height, camera_width),  # Use the camera-specific dimensions
        radius=point_radius,
        points_per_pixel=10
    )
    
    # Create renderer
    p3d_cameras = cameras.p3d_cameras[camera_idx]
    rasterizer = PointsRasterizer(cameras=p3d_cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )
    
    # Render the point cloud
    with torch.no_grad():
        image = renderer(point_cloud)
    
    return image[0, ..., :3]  # Return RGB only, no alpha

def main():
    args = parse_args()
    
    # Set GPU
    torch.cuda.set_device(args.gpu)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data and camera information
    print(f"\nLoading config {args.gs_checkpoint_path}...")
    nerfmodel = GaussianSplattingWrapper(
        source_path=args.source_path,
        output_path=args.gs_checkpoint_path,
        iteration_to_load=7000,  # Default iteration for vanilla GS
        load_gt_images=False,
        eval_split=args.eval_split,
        eval_split_interval=args.skip_images,
    )
    
    print(f'{len(nerfmodel.training_cameras)} training images detected.')
    
    # Select cameras for rendering
    cameras_to_use = nerfmodel.training_cameras
    
    # Generate the full path to the PLY file if not absolute
    sparse_path = args.sparse_ply_path
    # Check if PLY file exists
    if not os.path.exists(sparse_path):
        print(f"Error: Sparse point cloud file not found at {sparse_path}")
        sys.exit(1)
    
    try:
        if args.render_multi_view:
            # Render from multiple camera views
            num_cameras = len(cameras_to_use.gs_cameras)
            if args.num_views > num_cameras:
                print(f"Warning: Requested {args.num_views} views but only {num_cameras} cameras available")
                step = 1
            else:
                step = max(1, num_cameras // args.num_views)
            
            camera_indices = list(range(0, num_cameras, step))[:args.num_views]
            print(f"Rendering {len(camera_indices)} different camera views")
            
            for i, cam_idx in enumerate(camera_indices):
                print(f"\nRendering view {i+1}/{len(camera_indices)} from camera {cam_idx}")
                print("Image name:", cameras_to_use.gs_cameras[cam_idx].image_name)
                
                # Render the point cloud
                sparse_image = render_point_cloud(
                    sparse_path,
                    cameras_to_use,
                    cam_idx,
                    nerfmodel.device,
                    point_radius=args.point_size
                )
                
                plt.figure(figsize=(10 * args.plot_ratio, 10 * args.plot_ratio))
                plt.axis("off")
                plt.title(f"Sparse Reconstruction - View {i+1} (Camera {cam_idx})")
                plt.imshow(sparse_image.cpu().numpy())
                
                # Save the image
                sparse_output_path = os.path.join(args.output_dir, f"sparse_cam{cam_idx:03d}.png")
                plt.savefig(sparse_output_path, bbox_inches='tight')
                plt.close()
                print(f"Sparse reconstruction render saved to {sparse_output_path}")
        else:
            # Render from a single camera view
            if args.camera_index == -1:
                cam_idx = np.random.randint(0, len(cameras_to_use.gs_cameras))
            else:
                cam_idx = args.camera_index
            
            print(f"\nRendering from camera {cam_idx}")
            print("Image name:", cameras_to_use.gs_cameras[cam_idx].image_name)
            
            # Render the point cloud
            sparse_image = render_point_cloud(
                sparse_path,
                cameras_to_use,
                cam_idx,
                nerfmodel.device,
                point_radius=args.point_size
            )
            
            plt.figure(figsize=(10 * args.plot_ratio, 10 * args.plot_ratio))
            plt.axis("off")
            plt.title("Sparse Reconstruction")
            plt.imshow(sparse_image.cpu().numpy())
            
            # Save the image
            sparse_output_path = os.path.join(args.output_dir, f"sparse_cam{cam_idx:03d}.png")
            plt.savefig(sparse_output_path, bbox_inches='tight')
            plt.close()
            print(f"Sparse reconstruction render saved to {sparse_output_path}")
            
    except Exception as e:
        print(f"Error rendering sparse point cloud: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
    
    print("\nDone!")

if __name__ == "__main__":
    main()