# Adapted from by demo_gradio.py from the VGGT repository
# Author: Clinton T. Kunhardt
# Optimized for memory efficiency

import os
import cv2
import torch
import numpy as np
import sys
import glob
import time
import open3d as o3d
from typing import Dict, List, Optional, Tuple
import gc
from scipy.spatial.distance import cdist

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"

class MeshGenerationPipeline:
    """
    Converts VGGT point clouds and depth maps to textured meshes
    """
    def __init__(self, method='tsdf', voxel_size=0.005):
        self.method = method
        self.voxel_size = voxel_size
        
    def generate_mesh(self, predictions: Dict) -> o3d.geometry.TriangleMesh:
        """Generate textured mesh from VGGT predictions"""
        print(f"Generating mesh using {self.method} method...")
        
        if self.method == 'tsdf':
            return self._generate_mesh_tsdf(predictions)
        elif self.method == 'poisson':
            return self._generate_mesh_poisson(predictions)
        elif self.method == 'alpha_shapes':
            return self._generate_mesh_alpha_shapes(predictions)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _generate_mesh_tsdf(self, predictions: Dict) -> o3d.geometry.TriangleMesh:
        """Generate mesh using TSDF fusion of depth maps"""
        depth_maps = predictions["depth"]  # Shape: (S, H, W)
        images = predictions["images"]     # Shape: (S, H, W, 3) or (S, 3, H, W)
        extrinsics = predictions["extrinsic"]  # Shape: (S, 3, 4)
        intrinsics = predictions["intrinsic"]  # Shape: (S, 3, 3)
        
        # Handle different image formats
        if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
            images = np.transpose(images, (0, 2, 3, 1))  # Convert to NHWC
        
        # Ensure images are in [0, 255] range and uint8
        if images.max() <= 1.0:
            images = (images * 255).astype(np.uint8)
        else:
            images = images.astype(np.uint8)
        
        # Create TSDF volume
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            sdf_trunc=self.voxel_size * 8,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        
        print(f"Integrating {len(depth_maps)} depth maps...")
        for i in range(len(depth_maps)):
            # Ensure depth is float32 and contiguous
            depth_array = np.ascontiguousarray(depth_maps[i].astype(np.float32))
            color_array = np.ascontiguousarray(images[i])
            
            # Convert to Open3D format
            depth_o3d = o3d.geometry.Image(depth_array)
            color_o3d = o3d.geometry.Image(color_array)
            
            # Create RGBD image
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d,
                depth_scale=1.0,
                depth_trunc=50.0,
                convert_rgb_to_intensity=False
            )
            
            # Convert camera parameters
            intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
            intrinsic_o3d.intrinsic_matrix = intrinsics[i]
            
            # Convert extrinsic (3x4) to 4x4 transformation matrix
            extrinsic_4x4 = np.eye(4)
            extrinsic_4x4[:3, :] = extrinsics[i]
            
            # Integrate into TSDF volume
            volume.integrate(rgbd, intrinsic_o3d, extrinsic_4x4)
        
        # Extract mesh
        print("Extracting mesh from TSDF volume...")
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        
        # Clean up mesh
        mesh = self._clean_mesh(mesh)
        
        return mesh
    
    def _generate_mesh_poisson(self, predictions: Dict) -> o3d.geometry.TriangleMesh:
        """Generate mesh using Poisson surface reconstruction"""
        world_points = predictions["world_points_from_depth"]
        images = predictions["images"]
        
        # Handle different image formats
        if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
            images = np.transpose(images, (0, 2, 3, 1))
        
        # Reshape points and colors
        points = world_points.reshape(-1, 3)
        colors = images.reshape(-1, 3)
        
        # Filter invalid points
        valid_mask = ~np.any(np.isnan(points) | np.isinf(points), axis=1)
        points = points[valid_mask]
        colors = colors[valid_mask]
        
        # Subsample if too many points
        if len(points) > 500000:
            indices = np.random.choice(len(points), 500000, replace=False)
            points = points[indices]
            colors = colors[indices]
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Estimate normals
        print("Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(100)
        
        # Poisson reconstruction
        print("Running Poisson reconstruction...")
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=10, width=0, scale=1.1, linear_fit=False
        )
        
        # Clean up mesh
        mesh = self._clean_mesh(mesh)
        
        return mesh
    
    def _generate_mesh_alpha_shapes(self, predictions: Dict) -> o3d.geometry.TriangleMesh:
        """Generate mesh using Alpha Shapes"""
        world_points = predictions["world_points_from_depth"]
        images = predictions["images"]
        
        # Handle different image formats
        if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
            images = np.transpose(images, (0, 2, 3, 1))
        
        # Reshape points and colors
        points = world_points.reshape(-1, 3)
        colors = images.reshape(-1, 3)
        
        # Filter invalid points
        valid_mask = ~np.any(np.isnan(points) | np.isinf(points), axis=1)
        points = points[valid_mask]
        colors = colors[valid_mask]
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Estimate alpha value
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        alpha = avg_dist * 2
        
        print(f"Creating alpha shape with alpha={alpha:.4f}...")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        
        # Clean up mesh
        mesh = self._clean_mesh(mesh)
        
        return mesh
    
    def _clean_mesh(self, mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
        """Clean up mesh by removing small components and smoothing"""
        print("Cleaning mesh...")
        
        # Remove degenerate triangles
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        # Keep only largest connected component
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles()
        )
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        
        if len(cluster_n_triangles) > 0:
            largest_cluster_idx = cluster_n_triangles.argmax()
            triangles_to_remove = triangle_clusters != largest_cluster_idx
            mesh.remove_triangles_by_mask(triangles_to_remove)
        
        # Smooth mesh slightly
        mesh = mesh.filter_smooth_simple(number_of_iterations=2)
        
        # Recompute normals
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        
        print(f"Final mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        
        return mesh

class VGGTPhotogrammetryPipeline:
    def __init__(self, max_images: int = 24, max_resolution: int = 320):
        """
        Initialize VGGT pipeline with memory constraints
        
        Args:
            max_images: Maximum number of images to process at once
            max_resolution: Maximum image resolution (images will be resized if larger)
        """
        self.max_images = max_images
        self.max_resolution = max_resolution
        self.device = device
        self.model = None
        
    def load_model(self):
        """Load VGGT model with memory optimization"""
        print("Initializing and loading VGGT model...")
        start_time = time.time()
        
        # Clear any existing CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        
        # Load model state dict
        state_dict = torch.hub.load_state_dict_from_url(_URL, map_location='cpu')
        self.model.load_state_dict(state_dict)
        
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # Set memory efficient settings
        if hasattr(self.model, 'gradient_checkpointing'):
            self.model.gradient_checkpointing = True
            
        end_time = time.time()
        print(f"Model loaded in {end_time - start_time:.2f} seconds")
        
        # Print memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU memory allocated after model loading: {memory_allocated:.2f} GB")
    
    def preprocess_images(self, image_paths: List[str]) -> Tuple[torch.Tensor, List[str]]:
        """
        Load and preprocess images with memory optimization
        """
        print(f"Found {len(image_paths)} images")
        
        # Limit number of images
        if len(image_paths) > self.max_images:
            print(f"Limiting to {self.max_images} images (found {len(image_paths)})")
            # Take evenly spaced images
            indices = np.linspace(0, len(image_paths) - 1, self.max_images, dtype=int)
            image_paths = [image_paths[i] for i in indices]
        
        # Check image sizes and resize if necessary
        processed_paths = []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            h, w = img.shape[:2]
            max_dim = max(h, w)
            
            if max_dim > self.max_resolution:
                # Resize image
                scale = self.max_resolution / max_dim
                new_w, new_h = int(w * scale), int(h * scale)
                img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Save resized image temporarily
                base_name = os.path.basename(img_path)
                temp_path = f"temp_resized_{base_name}"
                cv2.imwrite(temp_path, img_resized)
                processed_paths.append(temp_path)
                print(f"Resized {base_name}: {w}x{h} -> {new_w}x{new_h}")
            else:
                processed_paths.append(img_path)
        
        print(f"Processing {len(processed_paths)} images")
        
        # Load and preprocess
        try:
            images = load_and_preprocess_images(processed_paths).to(self.device)
            print(f"Preprocessed images shape: {images.shape}")
            
            # Check memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"GPU memory after image loading: {memory_allocated:.2f} GB")
                
            return images, processed_paths
            
        finally:
            # Clean up temporary files
            for path in processed_paths:
                if path.startswith("temp_resized_") and os.path.exists(path):
                    os.remove(path)
    
    def run_inference_batched(self, images: torch.Tensor) -> Dict:
        """
        Run inference - process all images together for coherent reconstruction
        """
        print(f"Processing all {images.shape[0]} images")
        return self._run_single_batch(images)
    
    def _run_single_batch(self, images: torch.Tensor) -> Dict:
        """Run inference on a single batch"""
        print("Running inference...")
        start_time = time.time()
        
        # Use mixed precision for memory efficiency
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = self.model(images)
        
        end_time = time.time()
        print(f"Inference completed in {end_time - start_time:.2f} seconds")
        
        # Convert pose encoding
        print("Converting pose encoding...")
        start_time = time.time()
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        end_time = time.time()
        print(f"Pose encoding conversion completed in {end_time - start_time:.2f} seconds")
        
        # Convert to numpy and move to CPU
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)
        
        # Generate world points
        print("Computing world points from depth map...")
        start_time = time.time()
        depth_map = predictions["depth"]
        world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
        predictions["world_points_from_depth"] = world_points
        end_time = time.time()
        print(f"World points computed in {end_time - start_time:.2f} seconds")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        return predictions
    
    def process_directory(self, target_dir: str) -> Dict:
        """Process all images in a directory"""
        print(f"Processing images from {target_dir}")
        start_time_all = time.time()
        
        # Find images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(target_dir, "images", ext)))
            image_paths.extend(glob.glob(os.path.join(target_dir, "images", ext.upper())))
        
        image_paths = sorted(image_paths)
        
        if len(image_paths) == 0:
            raise ValueError("No images found. Check your target directory.")
        
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Preprocess images
        images, processed_paths = self.preprocess_images(image_paths)
        
        # Run inference
        predictions = self.run_inference_batched(images)
        
        end_time_all = time.time()
        print(f"Total processing time: {end_time_all - start_time_all:.2f} seconds")
        
        return predictions

def main():
    # Configure for your GPU memory - no batching, process all together
    pipeline = VGGTPhotogrammetryPipeline(
        max_images=128,          # Process up to 128 images together
        max_resolution=320      # Reduce resolution to save memory
    )
    
    target_dir = "C:\\repos\\gatech\\photogrammetry\\buddha"
    
    try:
        predictions = pipeline.process_directory(target_dir)
        
        # Save results with proper parameter types
        glb_path = os.path.join(target_dir, "predictions.glb")
        print(f"Saving predictions to {glb_path}")
        scene = predictions_to_glb(
            predictions, 
            conf_thres=50.0,  # Ensure it's a float
            target_dir=target_dir
        )
        
        # Actually save the GLB file
        scene.export(glb_path)
        print(f"GLB file saved to {glb_path}")
        
        # Generate mesh from predictions using different methods
        mesh_methods = ['poisson'] # ['tsdf', 'poisson', 'alpha_shapes']
        
        for method in mesh_methods:
            try:
                print(f"\n--- Generating mesh using {method} ---")
                mesh_pipeline = MeshGenerationPipeline(method=method, voxel_size=0.005)
                textured_mesh = mesh_pipeline.generate_mesh(predictions)
                
                # Save mesh
                mesh_path = os.path.join(target_dir, f"reconstructed_mesh_{method}.ply")
                o3d.io.write_triangle_mesh(mesh_path, textured_mesh)
                print(f"Saved {method} mesh to {mesh_path}")
                
            except Exception as e:
                print(f"Failed to generate {method} mesh: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Visualize point cloud
        pcd = o3d.geometry.PointCloud()
        world_points = predictions["world_points_from_depth"].reshape(-1, 3)
        
        # Filter out invalid points
        valid_mask = ~np.any(np.isnan(world_points) | np.isinf(world_points), axis=1)
        world_points = world_points[valid_mask]
        
        pcd.points = o3d.utility.Vector3dVector(world_points)
        print(f"Generated point cloud with {len(pcd.points)} points")
        
        print("Visualizing point cloud...")
        o3d.visualization.draw_geometries([pcd])
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"Still getting OOM error: {e}")
        print("Try reducing max_images or max_resolution further")
        print("Current GPU memory usage:")
        if torch.cuda.is_available():
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print("Suggested settings for 12GB GPU:")
        print("- max_images=12, max_resolution=320 (conservative)")
        print("- max_images=16, max_resolution=384 (balanced)")
        print("- max_images=20, max_resolution=448 (aggressive)")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()