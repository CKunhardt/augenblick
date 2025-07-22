#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import open3d as o3d
from tqdm import tqdm
import json
import torch
import cv2
from glob import glob
from pathlib import Path
from sugar_scene.gs_model import GaussianSplattingWrapper
from pytorch3d.io import load_objs_as_meshes
from sklearn.neighbors import NearestNeighbors

class ReconstructionBenchmark:
    """
    A class to benchmark and compare sparse and dense reconstructions
    based on point size, point density, surface coverage, and mesh complexity
    across different views.
    """
    
    def __init__(self, args):
        """Initialize the benchmark with command line arguments"""
        self.args = args
        self.results = {
            "sparse": {},
            "dense": {},
            "comparison": {}
        }
        
        # Set the device
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load camera information
        self.load_camera_info()
        
        # Load sparse reconstruction
        if args.sparse_ply_path:
            self.load_sparse_reconstruction()
        
        # Load dense reconstruction (mesh or point cloud)
        if args.dense_obj_path:
            self.load_dense_mesh()

        if args.dense_ply_path:
            self.load_dense_reconstruction()
            
        # Load rendered images if available
        if args.sparse_renders_dir and args.dense_renders_dir:
            self.load_rendered_images()
        else:
            self.sparse_renders = {}
            self.dense_renders = {}
            self.has_rendered_images = False
    
    def load_camera_info(self):
        """Load camera information from the SuGaR dataset"""
        print(f"Loading camera information from {self.args.gs_checkpoint_path}...")
        try:
            self.model = GaussianSplattingWrapper(
                source_path=self.args.source_path,
                output_path=self.args.gs_checkpoint_path,
                iteration_to_load=7000,  # Default iteration
                load_gt_images=False,
                eval_split=False
            )
            self.cameras = self.model.training_cameras
            print(f"Loaded {len(self.cameras.gs_cameras)} cameras")
        except Exception as e:
            print(f"Error loading camera information: {str(e)}")
            sys.exit(1)
    
    def load_sparse_reconstruction(self):
        """Load sparse reconstruction from PLY file"""
        try:
            print(f"Loading sparse reconstruction from {self.args.sparse_ply_path}...")
            self.sparse_pcd = o3d.io.read_point_cloud(self.args.sparse_ply_path)
            sparse_points = np.asarray(self.sparse_pcd.points)
            print(f"Loaded {len(sparse_points)} sparse points")
            
            # Calculate bounding box for the sparse point cloud
            sparse_bbox_min = sparse_points.min(axis=0)
            sparse_bbox_max = sparse_points.max(axis=0)
            sparse_bbox_size = sparse_bbox_max - sparse_bbox_min
            sparse_bbox_volume = np.prod(sparse_bbox_size)
            
            # Store results
            self.results["sparse"]["num_points"] = len(sparse_points)
            self.results["sparse"]["bbox_volume"] = float(sparse_bbox_volume)
            self.results["sparse"]["point_density"] = len(sparse_points) / sparse_bbox_volume
            
            # Get statistical information about the points
            if len(sparse_points) > 0:
                distances = np.sqrt(np.sum(np.diff(sparse_points, axis=0)**2, axis=1))
                if len(distances) > 0:
                    self.results["sparse"]["avg_point_distance"] = float(np.mean(distances))
                    self.results["sparse"]["min_point_distance"] = float(np.min(distances))
                    self.results["sparse"]["max_point_distance"] = float(np.max(distances))
                else:
                    self.results["sparse"]["avg_point_distance"] = 0
                    self.results["sparse"]["min_point_distance"] = 0
                    self.results["sparse"]["max_point_distance"] = 0
        except Exception as e:
            print(f"Error loading sparse reconstruction: {str(e)}")
            self.sparse_pcd = None
    
    def load_dense_reconstruction(self):
        """Load dense reconstruction from PLY file"""
        try:
            print(f"Loading dense reconstruction from {self.args.dense_ply_path}...")
            self.dense_pcd = o3d.io.read_point_cloud(self.args.dense_ply_path)
            dense_points = np.asarray(self.dense_pcd.points)
            print(f"Loaded {len(dense_points)} dense points")
            
            # Calculate bounding box for the dense point cloud
            dense_bbox_min = dense_points.min(axis=0)
            dense_bbox_max = dense_points.max(axis=0)
            dense_bbox_size = dense_bbox_max - dense_bbox_min
            dense_bbox_volume = np.prod(dense_bbox_size)
            
            # Store results
            self.results["dense"]["num_points"] = len(dense_points)
            self.results["dense"]["bbox_volume"] = float(dense_bbox_volume)
            self.results["dense"]["point_density"] = len(dense_points) / dense_bbox_volume
            
            # Get statistical information about the points
            if len(dense_points) > 0:
                distances = np.sqrt(np.sum(np.diff(dense_points, axis=0)**2, axis=1))
                if len(distances) > 0:
                    self.results["dense"]["avg_point_distance"] = float(np.mean(distances))
                    self.results["dense"]["min_point_distance"] = float(np.min(distances))
                    self.results["dense"]["max_point_distance"] = float(np.max(distances))
                else:
                    self.results["dense"]["avg_point_distance"] = 0
                    self.results["dense"]["min_point_distance"] = 0
                    self.results["dense"]["max_point_distance"] = 0
        except Exception as e:
            print(f"Error loading dense reconstruction: {str(e)}")
            self.dense_pcd = None
    
    def load_dense_mesh(self):
        """Load dense reconstruction from OBJ file"""
        try:
            print(f"Loading dense mesh from {self.args.dense_obj_path}...")
            self.dense_mesh = load_objs_as_meshes([self.args.dense_obj_path], device=self.device)
            
            # Get vertices and faces
            verts = self.dense_mesh.verts_list()[0].cpu().numpy()
            faces = self.dense_mesh.faces_list()[0].cpu().numpy()
            
            print(f"Loaded mesh with {len(verts)} vertices and {len(faces)} faces")
            
            # Calculate mesh complexity
            edge_count = len(faces) * 3 // 2  # Approximation for closed manifold meshes
            
            # Calculate bounding box
            bbox_min = verts.min(axis=0)
            bbox_max = verts.max(axis=0)
            bbox_size = bbox_max - bbox_min
            bbox_volume = np.prod(bbox_size)
            
            # Store results
            self.results["dense"]["num_vertices"] = len(verts)
            self.results["dense"]["num_faces"] = len(faces)
            self.results["dense"]["num_edges"] = edge_count
            self.results["dense"]["mesh_complexity"] = edge_count / len(verts)
            self.results["dense"]["bbox_volume"] = float(bbox_volume)
            self.results["dense"]["vertex_density"] = len(verts) / bbox_volume
            
            # Convert mesh to point cloud for comparison
            self.dense_pcd = o3d.geometry.PointCloud()
            self.dense_pcd.points = o3d.utility.Vector3dVector(verts)
            
        except Exception as e:
            print(f"Error loading dense mesh: {str(e)}")
            self.dense_mesh = None
            self.dense_pcd = None
    
    def load_rendered_images(self):
        """Load rendered images of sparse and dense reconstructions if available"""
        self.sparse_renders = {}
        self.dense_renders = {}
        self.has_rendered_images = False
        
        # Check if rendered image directories were provided
        if not self.args.sparse_renders_dir or not self.args.dense_renders_dir:
            print("Rendered image directories not provided, using distance-based metrics")
            return
            
        # Check if directories exist
        if not os.path.exists(self.args.sparse_renders_dir) or not os.path.exists(self.args.dense_renders_dir):
            print("One or both rendering directories not found, using distance-based metrics")
            return
            
        print("Loading rendered images for enhanced surface coverage comparison...")
        self.has_rendered_images = True
        
        # Get sparse renders
        sparse_files = sorted(glob(os.path.join(self.args.sparse_renders_dir, f"*{self.args.render_img_format}")))
        dense_files = sorted(glob(os.path.join(self.args.dense_renders_dir, f"*{self.args.render_img_format}")))

        # Process sparse renders
        for img_path in sparse_files:

            name = Path(img_path).stem
            self.sparse_renders[name] = cv2.imread(img_path)
        
        # Process dense renders
        for img_path in dense_files:
            if not img_path.split('/')[-1].startswith('sugar_'):
                continue
            name = Path(img_path).stem
            self.dense_renders[name] = cv2.imread(img_path)
            
        print(f"Loaded {len(self.sparse_renders)} sparse renders and {len(self.dense_renders)} dense renders")
    
    def get_topk_visible(self, points, center, k=5000):
        nbrs = NearestNeighbors(n_neighbors=min(k, len(points))).fit(points)
        dists, indices = nbrs.kneighbors(center.reshape(1, -1))
        return points[indices[0]]

    def calculate_direct_render_metrics(self):
        """Calculate metrics for each camera view"""
        if not hasattr(self, 'sparse_renders') or not hasattr(self, 'dense_renders'):
            print("Rendered images are required for direct comparison")
            return
        
        print("Calculating metrics directly from rendered images...")
        view_metrics = []

        # Only process views that have rendered images
        if not hasattr(self, 'has_rendered_images') or not self.has_rendered_images or len(self.sparse_renders) == 0:
            print("Error: No rendered images available. Cannot calculate per-view metrics.")
            print("Please provide --sparse_renders_dir and --dense_renders_dir arguments.")
            return
        
        print("Using rendered image views for metrics calculation")
        
        # Go through all cameras and find matching ones with renders
        views = [int(view.split('_')[-1]) for view in self.dense_renders]
        print(f"Found {len(views)} views with both sparse and dense renders")

        sparse_keys = sorted(list(self.sparse_renders.keys()))
        dense_keys = sorted(list(self.dense_renders.keys()))

        has_mesh = hasattr(self, 'dense_mesh') and self.dense_mesh is not None

        for i in tqdm(range(len(views)), desc="Processing views"):
            sparse_key = sparse_keys[i]
            dense_key = dense_keys[i]
            
            print(f"Comparing sparse render '{sparse_key}' with dense render '{dense_key}'")
            
            # Get images
            sparse_img = self.sparse_renders[sparse_key]
            dense_img = self.dense_renders[dense_key]

            # Convert images to grayscale for coverage calculation
            sparse_gray = cv2.cvtColor(sparse_img, cv2.COLOR_BGR2GRAY)
            dense_gray = cv2.cvtColor(dense_img, cv2.COLOR_BGR2GRAY)
            
            # Calculate coverage (number of non-black pixels)
            sparse_coverage = np.sum(sparse_gray > 10) / (sparse_gray.shape[0] * sparse_gray.shape[1])
            dense_coverage = np.sum(dense_gray > 10) / (dense_gray.shape[0] * dense_gray.shape[1])
       
            # Calculate coverage ratio
            surface_coverage_ratio = sparse_coverage / dense_coverage if dense_coverage > 0 else 0
            
            # We'll use camera 0's position for density calculation as we don't have specific camera info
            # This is just an approximation since we're focusing on image-based metrics
            # print("Available camera attributes:", dir(self.cameras.gs_cameras[0]))
  
            camera_center = self.cameras.gs_cameras[views[i]].camera_center
            print(f"Camera {i} center: {camera_center}")
            # Convert camera center from torch tensor to numpy array
            if isinstance(camera_center, torch.Tensor):
                camera_center = camera_center.detach().cpu().numpy()
            
            # Calculate point densities with the same method as calculate_view_metrics
            sparse_points = np.asarray(self.sparse_pcd.points)
            dense_points = np.asarray(self.dense_pcd.points)
            
   
            visible_sparse_points = self.get_topk_visible(sparse_points, camera_center, k=5000)
            visible_dense_points = self.get_topk_visible(dense_points, camera_center, k=5000)

            sparse_bbox_min = visible_sparse_points.min(axis=0)
            sparse_bbox_max = visible_sparse_points.max(axis=0)
            dense_bbox_min = visible_dense_points.min(axis=0)
            dense_bbox_max = visible_dense_points.max(axis=0)
            
            sparse_volume = np.prod(sparse_bbox_max - sparse_bbox_min)
            dense_volume = np.prod(dense_bbox_max - dense_bbox_min)
            
            sparse_density = len(visible_sparse_points) / sparse_volume if sparse_volume > 0 else 0
            dense_density = len(visible_dense_points) / dense_volume if dense_volume > 0 else 0
            point_density_ratio = sparse_density / dense_density if dense_density > 0 else 0
            print(f"Camera {i}: Sparse density = {sparse_density:.4f}, Dense density = {dense_density:.4f}, ")

               # Add mesh-specific metrics if available
            mesh_metrics = {}
            if has_mesh:
                # Calculate mesh quality for this view
                visible_face_count = 0
                visible_area = 0.0
                
                # For demonstration, we'll use a proxy metric: the number of visible pixels
                # In a real implementation, you could project the mesh and count visible faces
                mesh_coverage = dense_coverage  # Use image coverage as a proxy
                
                # Add mesh metrics to the result
                mesh_metrics = {
                    "mesh_visibility": float(dense_coverage * 100),
                    "mesh_coverage_ratio": float(dense_coverage / sparse_coverage if sparse_coverage > 0 else 0),
                    "visible_face_density": float(dense_density)  # Use point density as proxy
                }
                print(f"Camera {i}: Mesh visibility = {mesh_metrics['mesh_visibility']:.2f}%")

            # Store results
            metrics_dict = {
                "camera_idx": i,
                "sparse_key": sparse_key,
                "dense_key": dense_key,
                "sparse_visibility": int(round(sparse_coverage, 2) * 100),
                "dense_visibility": int(round(dense_coverage, 2) * 100),
                "surface_coverage_ratio": float(surface_coverage_ratio),
                "sparse_density": float(sparse_density),
                "dense_density": float(dense_density),
                "point_density_ratio": float(point_density_ratio),
            }
            
            # Add mesh metrics if available
            if mesh_metrics:
                metrics_dict.update(mesh_metrics)
                
            view_metrics.append(metrics_dict)
            print(f"Camera {i}: Sparse density = {sparse_density:.4f}, Dense density = {dense_density:.4f}")
    
        # Store metrics in results
        if view_metrics:
            self.results["per_view"] = view_metrics
            print(f"Processed metrics for {len(view_metrics)} render pairs")
    
    def calculate_global_metrics(self):
        """Calculate global comparison metrics"""
        if not hasattr(self, 'sparse_pcd') or not hasattr(self, 'dense_pcd'):
            return
            
        # Compare point counts
        sparse_point_count = self.results["sparse"]["num_points"]
        dense_point_count = self.results["dense"]["num_points"]
        self.results["comparison"]["point_size_ratio"] = sparse_point_count / dense_point_count
            
        # Compare point densities
        if "point_density" in self.results["sparse"] and "point_density" in self.results["dense"]:
            sparse_density = self.results["sparse"]["point_density"]
            dense_density = self.results["dense"]["point_density"]
            self.results["comparison"]["point_density_ratio"] = sparse_density / dense_density
            
    def generate_plots(self):
        """Generate plots comparing the metrics"""
        if not self.results.get("per_view"):
            print("No per-view metrics available for plotting")
            return
            
        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data for plots
        camera_indices = ["Camera" + str(m["camera_idx"]+1) for m in self.results["per_view"]]
        point_density_ratio = [m["point_density_ratio"] for m in self.results["per_view"]]
        sparse_visibility = [m["sparse_visibility"] for m in self.results["per_view"]]
        dense_visibility = [m["dense_visibility"] for m in self.results["per_view"]]
        sparse_density = [m["sparse_density"] for m in self.results["per_view"]]
        dense_density = [m["dense_density"] for m in self.results["per_view"]]
        
        # Create figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Add a general title for all four subplots
        fig.suptitle('3D Reconstruction Benchmark Results (VGGT + Sugar)', fontsize=16, y=0.98)
        
        # Plot 1: Point Size Comparison
        axs[0, 0].bar(['Sparse', 'Dense'], [self.results["sparse"]["num_points"], self.results["dense"]["num_points"]])
        axs[0, 0].set_title('Point Cloud Size Comparison')
        axs[0, 0].set_ylabel('Number of Points')
        
        # Plot 2: Point Density Comparison
        x = np.arange(len(camera_indices))
        width = 0.35
        axs[0, 1].bar(x, sparse_density, width, label='Sparse')
        axs[0, 1].bar(x + width, dense_density, width, label='Dense')
        axs[0, 1].set_title('Point Density Comparison')
        axs[0, 1].set_xlabel('Camera')
        axs[0, 1].set_yscale('log')
        axs[0, 1].set_ylabel('Points per Cubic Unit')
        axs[0, 1].set_xticks(x + width/2)
        axs[0, 1].set_xticklabels(camera_indices)
        axs[0, 1].legend()
        
        # Plot 3: Surface Coverage by View
        axs[1, 0].bar(x, sparse_visibility, width, label='Sparse')
        axs[1, 0].bar(x + width, dense_visibility, width, label='Dense')
        axs[1, 0].set_title('Surface Coverage Comparison')
        axs[1, 0].set_xlabel('Camera')
        axs[1, 0].set_ylabel('Coverage (%)')
        axs[1, 0].set_xticks(x + width/2)
        axs[1, 0].set_xticklabels(camera_indices)
        axs[1, 0].legend()
        
        # Plot 4: Mesh Complexity (if mesh data available) or Point Density Ratio
        if "num_vertices" in self.results["dense"]:
            # For meshes, use visible face count by view
            x = np.arange(len(camera_indices))
            width = 0.7
            
            # Extract mesh face counts or visibility based on what's available
            if "visible_face_density" in self.results["per_view"][0]:
                # Use the visible face density as a proxy for face count
                # Multiply by a factor to get approximate face count
                # More accurate approach
                total_faces = self.results["dense"]["num_faces"]
                approx_face_count = [m["mesh_visibility"] / 10 * total_faces for m in self.results["per_view"]]
                axs[1, 1].bar(x, approx_face_count, width)
                axs[1, 1].set_title('Estimated Visible Face Count')
                axs[1, 1].set_xlabel('Camera')
                axs[1, 1].set_ylabel('Number of Visible Faces')
            else:
                # Fallback to mesh visibility percentage
                mesh_metrics = [m.get("mesh_visibility", 0) for m in self.results["per_view"]]
                axs[1, 1].bar(x, mesh_metrics, width, color='darkgreen')
                axs[1, 1].set_title('Mesh Visibility by View')
                axs[1, 1].set_xlabel('Camera')
                axs[1, 1].set_ylabel('Mesh Visibility (%)')
            
            axs[1, 1].set_xticks(x)
            axs[1, 1].set_xticklabels(camera_indices)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reconstruction_metrics.png'))
        plt.close()
    
    def save_results(self):
        """Save benchmark results to a JSON file"""
        if self.results:
            output_file = os.path.join(self.args.output_dir, 'benchmark_results.json')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=4)
                
            print(f"Results saved to {output_file}")
    
    def run(self):
        """Run the complete benchmark process"""
        if hasattr(self, 'sparse_pcd') and (hasattr(self, 'dense_pcd') or hasattr(self, 'dense_mesh')):
            self.calculate_global_metrics()
            
            # Replace calculate_metrics_per_view with direct render comparison
            if hasattr(self, 'sparse_renders') and hasattr(self, 'dense_renders') and len(self.sparse_renders) > 0 and len(self.dense_renders) > 0:
                self.calculate_direct_render_metrics()
            else:
                ValueError("Rendered images are required for the comparison")
                
            self.generate_plots()
            self.save_results()
            print("Benchmark completed successfully!")
        else:
            print("Required reconstructions not available. Please check input paths.")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Benchmark sparse and dense reconstructions')
    
    # Input paths
    parser.add_argument('--source_path', type=str, required=True,
                      help='Path to the source data directory')
    parser.add_argument('--gs_checkpoint_path', type=str, required=True,
                      help='Path to the Gaussian Splatting checkpoint (for camera information)')
    parser.add_argument('--sparse_ply_path', type=str, required=True,
                      help='Path to the sparse reconstruction PLY file')
    
    # Dense reconstruction 
    parser.add_argument('--dense_ply_path', type=str, required=True,
                     help='Path to the dense point cloud PLY file')
    parser.add_argument('--dense_obj_path', type=str, required=True,
                     help='Path to the dense mesh OBJ file')
    
    # Rendered images (optional)
    parser.add_argument('--sparse_renders_dir', type=str, default=None,
                      help='Directory containing rendered images of sparse reconstruction')
    parser.add_argument('--dense_renders_dir', type=str, default=None,
                      help='Directory containing rendered images of dense reconstruction')
    parser.add_argument('--render_img_format', type=str, default='.png',
                      help='File extension of rendered images (default: .png)')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='./benchmark_output',
                      help='Directory to save benchmark results and plots')
    
    # Benchmark parameters
    parser.add_argument('--gpu', type=int, default=0,
                      help='GPU device index')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    benchmark = ReconstructionBenchmark(args)
    benchmark.run()
