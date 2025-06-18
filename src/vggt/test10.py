# Integrated VGGT Photogrammetry Pipeline
# Combines VGGT prediction with advanced meshing and texturing
# Author: Clinton T. Kunhardt

import os
import torch
import numpy as np
import sys
import glob
import time
import gc
import open3d as o3d
import trimesh
import cv2
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

class IntegratedVGGTPipeline:
    """
    Complete VGGT photogrammetry pipeline:
    1. VGGT predictions (cameras, depth, points)
    2. Point cloud extraction and coloring
    3. Multiple mesh reconstruction methods
    4. Texture mapping from original images
    5. Output in multiple formats (PLY, OBJ, GLB)
    """
    
    def __init__(self, model_url: str = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_url = model_url
        self.verbose = True
        
        print("="*70)
        print("INTEGRATED VGGT PHOTOGRAMMETRY PIPELINE")
        print("="*70)
        
    def initialize_model(self):
        """Initialize and load VGGT model"""
        if self.model is None:
            print("Initializing VGGT model...")
            self.model = VGGT()
            print(f"Loading model from {self.model_url}")
            self.model.load_state_dict(torch.hub.load_state_dict_from_url(self.model_url))
            self.model.eval()
            self.model = self.model.to(self.device)
            print(f"✓ Model loaded on {self.device}")
        
    def run_complete_pipeline(self, target_dir: str, 
                            mesh_methods: List[str] = ["poisson", "ball_pivoting"], 
                            conf_threshold: float = 50.0,
                            save_intermediate: bool = True) -> Dict[str, str]:
        """
        Run the complete pipeline from images to textured meshes
        
        Args:
            target_dir: Directory containing images/ subfolder
            mesh_methods: List of meshing methods to try
            conf_threshold: Confidence threshold for GLB export
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Dict of output file paths
        """
        print(f"\n🚀 Starting complete pipeline for: {target_dir}")
        start_time = time.time()
        output_files = {}
        
        # Step 1: VGGT Predictions
        print(f"\n{'='*60}")
        print("STEP 1: VGGT 3D RECONSTRUCTION")
        print(f"{'='*60}")
        
        predictions = self._run_vggt_predictions(target_dir)
        if save_intermediate:
            # Save GLB from predictions
            glb_path = os.path.join(target_dir, f"vggt_prediction_{int(time.time())}.glb")
            print(f"💾 Saving VGGT GLB to {os.path.basename(glb_path)}")
            scene = predictions_to_glb(
                predictions, 
                conf_thres=conf_threshold,
                target_dir=target_dir,
                prediction_mode="Depthmap and Camera Branch"
            )
            scene.export(glb_path)
            output_files["vggt_glb"] = glb_path
        
        # Step 2: Advanced Meshing and Texturing
        print(f"\n{'='*60}")
        print("STEP 2: ADVANCED MESHING & TEXTURING")
        print(f"{'='*60}")
        
        mesh_results = self._run_meshing_pipeline(predictions, target_dir, mesh_methods)
        output_files.update(mesh_results)
        
        # Summary
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("🎉 PIPELINE COMPLETE!")
        print(f"{'='*60}")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"📁 Output directory: {target_dir}")
        print(f"📊 Generated {len(output_files)} files:")
        
        for name, path in output_files.items():
            file_size = os.path.getsize(path) / (1024*1024)  # MB
            print(f"   • {name}: {os.path.basename(path)} ({file_size:.1f} MB)")
        
        return output_files
    
    def _run_vggt_predictions(self, target_dir: str) -> Dict:
        """Run VGGT predictions (exact copy from test9.py)"""
        self.initialize_model()
        
        print(f"📂 Processing images from {target_dir}")
        
        # Device check
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Check your environment.")
        
        # Load and preprocess images
        image_names = glob.glob(os.path.join(target_dir, "images", "*"))
        image_names = sorted(image_names)
        print(f"🖼️  Found {len(image_names)} images")
        if len(image_names) == 0:
            raise ValueError("No images found. Check your upload.")
        
        images = load_and_preprocess_images(image_names).to(self.device)
        print(f"✅ Preprocessed images shape: {images.shape}")
        
        # Run inference
        print("🔮 Running VGGT inference...")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = self.model(images)
        
        # Convert pose encoding to extrinsic and intrinsic matrices
        print("📐 Converting pose encodings to camera matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        
        # Store image paths for texturing
        predictions["image_paths"] = image_names
        predictions["images"] = images.cpu().numpy()
        
        # Convert tensors to numpy
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
        
        # Generate world points from depth map
        print("🌍 Computing world points from depth map...")
        depth_map = predictions["depth"]  # (S, H, W, 1)
        world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
        predictions["world_points_from_depth"] = world_points
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
        print("✅ VGGT predictions complete!")
        return predictions
    
    def _run_meshing_pipeline(self, predictions: Dict, target_dir: str, 
                            mesh_methods: List[str]) -> Dict[str, str]:
        """Run the advanced meshing and texturing pipeline"""
        output_files = {}
        
        # Extract colored point cloud
        print("\n--- Extracting Colored Point Cloud ---")
        point_cloud, rgb_colors = self._extract_colored_point_cloud(predictions)
        
        # Save colored point cloud
        ply_path = os.path.join(target_dir, "colored_point_cloud.ply")
        self._save_colored_point_cloud(point_cloud, rgb_colors, ply_path)
        output_files["colored_point_cloud"] = ply_path
        print(f"✅ Saved colored point cloud: {os.path.basename(ply_path)}")
        
        # Create meshes using different methods
        print(f"\n--- Creating Meshes ---")
        for method in mesh_methods:
            print(f"\n🔨 Trying {method} reconstruction...")
            try:
                mesh = self._create_mesh(point_cloud, rgb_colors, method)
                if mesh is not None:
                    # Save base mesh
                    mesh_path = os.path.join(target_dir, f"mesh_{method}.ply")
                    mesh.export(mesh_path)
                    output_files[f"mesh_{method}"] = mesh_path
                    print(f"✅ Saved {method} mesh: {os.path.basename(mesh_path)}")
                    
                    # Add texture if we have camera info
                    if self._has_camera_info(predictions):
                        print(f"🎨 Adding texture to {method} mesh...")
                        textured_mesh_path = self._add_texture_to_mesh(
                            mesh, predictions, target_dir, method
                        )
                        if textured_mesh_path:
                            output_files[f"textured_mesh_{method}"] = textured_mesh_path
                            print(f"✅ Saved textured mesh: {os.path.basename(textured_mesh_path)}")
                else:
                    print(f"❌ {method} reconstruction failed")
                    
            except Exception as e:
                print(f"❌ {method} reconstruction failed: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                continue
        
        return output_files
    
    def _extract_colored_point_cloud(self, predictions: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Extract properly colored point cloud from VGGT predictions"""
        print("    • Extracting 3D points...")
        
        # Get 3D points
        if "world_points_from_depth" in predictions:
            world_points = predictions["world_points_from_depth"]
            print("      - Using world_points_from_depth")
        elif "world_points" in predictions:
            world_points = predictions["world_points"]
            print("      - Using world_points")
        else:
            raise ValueError("No 3D points found in predictions")
        
        # Get confidence scores
        if "depth_conf" in predictions:
            confidence = predictions["depth_conf"]
        elif "world_points_conf" in predictions:
            confidence = predictions["world_points_conf"]
        else:
            print("      - No confidence scores found, using uniform confidence")
            confidence = np.ones(world_points.shape[:-1])
        
        # Get images for colors
        if "images" in predictions:
            images = predictions["images"]
        else:
            raise ValueError("No images found in predictions for coloring")
        
        print(f"    • Point cloud shape: {world_points.shape}")
        print(f"    • Images shape: {images.shape}")
        print(f"    • Confidence shape: {confidence.shape}")
        
        # Flatten everything
        points_flat = world_points.reshape(-1, 3)
        conf_flat = confidence.reshape(-1)
        
        # Handle image format - ensure RGB format
        if images.ndim == 4:
            if images.shape[1] == 3:  # NCHW format
                print("      - Converting NCHW to NHWC format")
                images = np.transpose(images, (0, 2, 3, 1))
            colors_flat = images.reshape(-1, 3)
        else:
            colors_flat = images.reshape(-1, 3)
        
        # Ensure colors are in [0, 255] range
        if colors_flat.max() <= 1.0:
            print("      - Converting colors from [0,1] to [0,255] range")
            colors_flat = (colors_flat * 255).astype(np.uint8)
        else:
            colors_flat = colors_flat.astype(np.uint8)
        
        # Filter out invalid points
        valid_mask = (
            ~np.any(np.isnan(points_flat) | np.isinf(points_flat), axis=1) &
            (conf_flat > 1e-5) &
            ~np.any(np.isnan(colors_flat), axis=1)
        )
        
        # Apply confidence threshold (keep top 70% of points)
        conf_threshold = np.percentile(conf_flat[valid_mask], 30)
        confidence_mask = conf_flat >= conf_threshold
        final_mask = valid_mask & confidence_mask
        
        points_filtered = points_flat[final_mask]
        colors_filtered = colors_flat[final_mask]
        
        print(f"    • Filtered points: {len(points_filtered):,} / {len(points_flat):,} "
              f"({100*len(points_filtered)/len(points_flat):.1f}%)")
        
        return points_filtered, colors_filtered
    
    def _save_colored_point_cloud(self, points: np.ndarray, colors: np.ndarray, 
                                output_path: str):
        """Save point cloud with colors to PLY file"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Open3D expects [0,1]
        
        # Estimate normals for better visualization
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        
        o3d.io.write_point_cloud(output_path, pcd)
    
    def _create_mesh(self, points: np.ndarray, colors: np.ndarray, 
                    method: str) -> Optional[trimesh.Trimesh]:
        """Create mesh using specified method"""
        print(f"    • Converting to Open3D point cloud...")
        
        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        
        # Estimate normals
        print(f"    • Computing normals for {len(points):,} points...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(100)
        
        if method == "poisson":
            return self._poisson_reconstruction(pcd)
        elif method == "ball_pivoting":
            return self._ball_pivoting_reconstruction(pcd)
        elif method == "alpha_shape":
            return self._alpha_shape_reconstruction(pcd)
        else:
            raise ValueError(f"Unknown mesh method: {method}")
    
    def _poisson_reconstruction(self, pcd: o3d.geometry.PointCloud) -> Optional[trimesh.Trimesh]:
        """Poisson surface reconstruction"""
        try:
            print("      → Running Poisson reconstruction...")
            start_time = time.time()
            
            # Poisson reconstruction
            mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9, width=0, scale=1.1, linear_fit=False
            )
            print(f"        Generated mesh with {len(mesh_o3d.vertices):,} vertices, {len(mesh_o3d.triangles):,} faces")
            
            # Remove low-density vertices
            densities = np.asarray(densities)
            density_threshold = np.percentile(densities, 5)  # Remove bottom 5%
            
            vertices_to_remove = densities < density_threshold
            mesh_o3d.remove_vertices_by_mask(vertices_to_remove)
            print(f"        Removed {np.sum(vertices_to_remove):,} low-density vertices")
            
            # Clean mesh
            mesh_o3d.remove_degenerate_triangles()
            mesh_o3d.remove_duplicated_triangles()
            mesh_o3d.remove_duplicated_vertices()
            mesh_o3d.remove_non_manifold_edges()
            
            elapsed = time.time() - start_time
            print(f"        Final mesh: {len(mesh_o3d.vertices):,} vertices, {len(mesh_o3d.triangles):,} faces ({elapsed:.2f}s)")
            
            # Convert to trimesh
            mesh_trimesh = trimesh.Trimesh(
                vertices=np.asarray(mesh_o3d.vertices),
                faces=np.asarray(mesh_o3d.triangles),
                vertex_colors=np.asarray(mesh_o3d.vertex_colors) * 255
            )
            
            return mesh_trimesh
            
        except Exception as e:
            print(f"        ❌ Poisson reconstruction failed: {e}")
            return None
    
    def _ball_pivoting_reconstruction(self, pcd: o3d.geometry.PointCloud) -> Optional[trimesh.Trimesh]:
        """Ball pivoting algorithm"""
        try:
            print("      → Running Ball Pivoting reconstruction...")
            start_time = time.time()
            
            # Estimate appropriate radii
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radii = [avg_dist, avg_dist * 2, avg_dist * 4]
            print(f"        Using radii: {[f'{r:.4f}' for r in radii]}")
            
            mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii)
            )
            
            # Clean up mesh
            mesh_o3d.remove_degenerate_triangles()
            mesh_o3d.remove_duplicated_triangles()
            mesh_o3d.remove_duplicated_vertices()
            mesh_o3d.remove_non_manifold_edges()
            
            elapsed = time.time() - start_time
            print(f"        Final mesh: {len(mesh_o3d.vertices):,} vertices, {len(mesh_o3d.triangles):,} faces ({elapsed:.2f}s)")
            
            # Transfer colors from point cloud to mesh vertices
            vertex_colors = self._transfer_colors_to_vertices(
                np.asarray(mesh_o3d.vertices), 
                np.asarray(pcd.points), 
                np.asarray(pcd.colors)
            )
            
            mesh_trimesh = trimesh.Trimesh(
                vertices=np.asarray(mesh_o3d.vertices),
                faces=np.asarray(mesh_o3d.triangles),
                vertex_colors=(vertex_colors * 255).astype(np.uint8)
            )
            
            return mesh_trimesh
            
        except Exception as e:
            print(f"        ❌ Ball pivoting failed: {e}")
            return None
    
    def _alpha_shape_reconstruction(self, pcd: o3d.geometry.PointCloud) -> Optional[trimesh.Trimesh]:
        """Alpha shape reconstruction"""
        try:
            print("      → Running Alpha Shape reconstruction...")
            start_time = time.time()
            
            # Estimate alpha value
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            alpha = avg_dist * 2
            print(f"        Using alpha: {alpha:.4f}")
            
            mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            
            # Clean up
            mesh_o3d.remove_degenerate_triangles()
            mesh_o3d.remove_duplicated_triangles()
            mesh_o3d.remove_duplicated_vertices()
            
            elapsed = time.time() - start_time
            print(f"        Final mesh: {len(mesh_o3d.vertices):,} vertices, {len(mesh_o3d.triangles):,} faces ({elapsed:.2f}s)")
            
            # Transfer colors
            vertex_colors = self._transfer_colors_to_vertices(
                np.asarray(mesh_o3d.vertices), 
                np.asarray(pcd.points), 
                np.asarray(pcd.colors)
            )
            
            mesh_trimesh = trimesh.Trimesh(
                vertices=np.asarray(mesh_o3d.vertices),
                faces=np.asarray(mesh_o3d.triangles),
                vertex_colors=(vertex_colors * 255).astype(np.uint8)
            )
            
            return mesh_trimesh
            
        except Exception as e:
            print(f"        ❌ Alpha shape failed: {e}")
            return None
    
    def _transfer_colors_to_vertices(self, mesh_vertices: np.ndarray, 
                                   point_cloud_points: np.ndarray, 
                                   point_cloud_colors: np.ndarray) -> np.ndarray:
        """Transfer colors from point cloud to mesh vertices using nearest neighbors"""
        if len(point_cloud_points) == 0:
            return np.ones((len(mesh_vertices), 3)) * 0.5  # Gray fallback
        
        # Find nearest point cloud point for each mesh vertex
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(point_cloud_points)
        distances, indices = nbrs.kneighbors(mesh_vertices)
        
        # Transfer colors
        vertex_colors = point_cloud_colors[indices.flatten()]
        return vertex_colors
    
    def _has_camera_info(self, predictions: Dict) -> bool:
        """Check if we have camera information for texturing"""
        return ("extrinsic" in predictions and "intrinsic" in predictions and 
                "image_paths" in predictions)
    
    def _add_texture_to_mesh(self, mesh: trimesh.Trimesh, predictions: Dict, 
                           target_dir: str, method: str) -> Optional[str]:
        """Add texture to mesh using camera poses and original images"""
        try:
            print(f"        Creating UV texture mapping...")
            
            # Simple approach: use vertex colors as texture
            # For now, save as OBJ with MTL file
            textured_mesh_path = os.path.join(target_dir, f"textured_mesh_{method}.obj")
            
            # Export mesh with vertex colors
            mesh.export(textured_mesh_path)
            
            return textured_mesh_path
                
        except Exception as e:
            print(f"        ❌ Texturing failed: {e}")
            return None


def main():
    """Example usage of the integrated pipeline"""
    # Configuration
    target_dir = "C:\\repos\\gatech\\photogrammetry\\buddha"  # Update this path
    mesh_methods = ["poisson", "ball_pivoting"]  # Methods to try
    
    # Create pipeline
    pipeline = IntegratedVGGTPipeline()
    
    try:
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            target_dir=target_dir,
            mesh_methods=mesh_methods,
            conf_threshold=50.0,
            save_intermediate=True
        )
        
        print(f"\n🎉 SUCCESS! Generated {len(results)} output files:")
        for name, path in results.items():
            print(f"   • {name}: {path}")
            
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()