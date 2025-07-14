#!/usr/bin/env python3
"""
VGG-T GLB Output Debug Script
Validates that a GLB file has all necessary components for HeatSDF pipeline
"""

import sys
import json
import numpy as np
import trimesh
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class ValidationReport:
    """Stores validation results"""
    has_point_cloud: bool = False
    has_colors: bool = False
    has_cameras: bool = False
    point_count: int = 0
    color_format: str = ""
    camera_count: int = 0
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    density_stats: Optional[Dict] = None
    warnings: List[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []

def analyze_glb(filepath: str) -> ValidationReport:
    """
    Analyze a GLB file and report on its contents
    """
    report = ValidationReport()
    
    try:
        # Load the GLB file
        scene = trimesh.load(filepath, force='scene')
        
        # Check if it's a single mesh or a scene
        if isinstance(scene, trimesh.Scene):
            print(f"✓ Loaded as Scene with {len(scene.geometry)} geometries")
            
            # VGG-T specific: geometry_0 is the point cloud, others are cameras
            camera_mesh_count = 0
            
            # Look for point cloud geometry
            for name, geom in scene.geometry.items():
                print(f"\n  Analyzing geometry '{name}':")
                print(f"    Type: {type(geom).__name__}")
                
                if isinstance(geom, trimesh.PointCloud):
                    report.has_point_cloud = True
                    report.point_count = len(geom.vertices)
                    
                    # Check for colors
                    if hasattr(geom, 'colors') and geom.colors is not None:
                        report.has_colors = True
                        report.color_format = f"shape={geom.colors.shape}, dtype={geom.colors.dtype}"
                        
                        # Validate color range
                        if geom.colors.max() > 1.0:
                            if geom.colors.max() > 255:
                                report.warnings.append("Colors appear to be in unexpected range (>255)")
                            else:
                                report.warnings.append("Colors appear to be in 0-255 range (will need normalization)")
                    
                    # Compute bounds
                    report.bounds = (geom.vertices.min(axis=0), geom.vertices.max(axis=0))
                    
                    # Analyze point density
                    report.density_stats = analyze_density(geom.vertices)
                    
                elif isinstance(geom, trimesh.Trimesh):
                    # VGG-T uses camera meshes (pyramids) to represent camera positions
                    if name != 'geometry_0':  # Not the main point cloud
                        camera_mesh_count += 1
                    
                    # Sometimes point clouds are stored as meshes with no faces
                    if len(geom.faces) == 0:
                        report.warnings.append(f"Found mesh '{name}' with no faces - might be a point cloud")
                        report.point_count = len(geom.vertices)
                        if hasattr(geom, 'vertex_colors') and geom.vertex_colors is not None:
                            report.has_colors = True
                            report.color_format = f"vertex_colors: shape={geom.vertex_colors.shape}"
            
            # If we found camera meshes, report them
            if camera_mesh_count > 0:
                report.has_cameras = True
                report.camera_count = camera_mesh_count
                print(f"\n✓ Found {camera_mesh_count} camera meshes (geometry_1 through geometry_{camera_mesh_count})")
            
            # Look for cameras in the scene graph
            if hasattr(scene, 'graph') and hasattr(scene.graph, 'nodes'):
                try:
                    for node_name, node in scene.graph.nodes.items():
                        if 'camera' in node_name.lower():
                            print(f"\n  Found camera node: {node_name}")
                            
                            # Check for transforms
                            if hasattr(scene.graph, 'transforms') and node_name in scene.graph.transforms:
                                transform = scene.graph.transforms[node_name]
                                print(f"    Has transform matrix: {transform.shape}")
                except AttributeError:
                    # Handle the dict_keys error gracefully
                    pass
            
            # Check metadata
            if hasattr(scene, 'metadata') and scene.metadata:
                try:
                    # Only try to serialize if metadata is serializable
                    if isinstance(scene.metadata, dict):
                        print(f"\n✓ Scene metadata keys: {list(scene.metadata.keys())}")
                        
                        # Look for camera parameters in metadata
                        if 'cameras' in scene.metadata:
                            report.has_cameras = True
                            cam_data = scene.metadata['cameras']
                            if isinstance(cam_data, list):
                                report.camera_count = len(cam_data)
                            print(f"  Found {report.camera_count} cameras in metadata")
                except Exception:
                    # Metadata might not be JSON serializable
                    pass
        
        elif isinstance(scene, trimesh.PointCloud):
            # Direct point cloud
            report.has_point_cloud = True
            report.point_count = len(scene.vertices)
            report.bounds = (scene.vertices.min(axis=0), scene.vertices.max(axis=0))
            
            if hasattr(scene, 'colors') and scene.colors is not None:
                report.has_colors = True
                report.color_format = f"shape={scene.colors.shape}, dtype={scene.colors.dtype}"
        
        else:
            report.errors.append(f"Unexpected type: {type(scene).__name__}")
    
    except Exception as e:
        report.errors.append(f"Failed to process file: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return report

def analyze_density(vertices: np.ndarray, k_neighbors: int = 10) -> Dict:
    """
    Analyze point cloud density statistics
    """
    from scipy.spatial import KDTree
    
    # Build KD-tree
    tree = KDTree(vertices)
    
    # Find k nearest neighbors for each point
    distances, _ = tree.query(vertices, k=k_neighbors+1)  # +1 because it includes self
    distances = distances[:, 1:]  # Remove self-distance
    
    # Compute statistics
    mean_distances = distances.mean(axis=1)
    
    return {
        'mean_nn_distance': float(mean_distances.mean()),
        'std_nn_distance': float(mean_distances.std()),
        'min_nn_distance': float(mean_distances.min()),
        'max_nn_distance': float(mean_distances.max()),
        'density_variation': float(mean_distances.std() / mean_distances.mean())  # Coefficient of variation
    }

def print_report(report: ValidationReport, filepath: str):
    """
    Pretty print the validation report
    """
    print("\n" + "="*60)
    print(f"VGG-T GLB VALIDATION REPORT")
    print(f"File: {Path(filepath).name}")
    print("="*60)
    
    # Essential components for HeatSDF
    print("\n📊 ESSENTIAL COMPONENTS FOR HeatSDF:")
    print(f"  Point Cloud: {'✅ YES' if report.has_point_cloud else '❌ NO'}")
    if report.has_point_cloud:
        print(f"    - Point count: {report.point_count:,}")
        if report.bounds:
            print(f"    - Bounds: [{report.bounds[0]}] to [{report.bounds[1]}]")
            size = report.bounds[1] - report.bounds[0]
            print(f"    - Size: {size}")
            print(f"    - Diagonal: {np.linalg.norm(size):.4f}")
    
    # Color information
    print(f"\n  Colors: {'✅ YES' if report.has_colors else '⚠️  NO (needed for textured output)'}")
    if report.has_colors:
        print(f"    - Format: {report.color_format}")
    
    # Density analysis
    if report.density_stats:
        print(f"\n  Density Statistics:")
        for key, value in report.density_stats.items():
            print(f"    - {key}: {value:.6f}")
        
        # Density warnings
        if report.density_stats['density_variation'] > 0.5:
            report.warnings.append("High density variation detected (>50%) - consider adaptive weighting")
    
    # Camera information (optional for HeatSDF)
    print(f"\n📸 CAMERA INFORMATION (optional for HeatSDF):")
    print(f"  Cameras: {'✅ YES' if report.has_cameras else '⚠️  NO (not required for HeatSDF)'}")
    if report.has_cameras:
        print(f"    - Camera count: {report.camera_count}")
    
    # Warnings and errors
    if report.warnings:
        print(f"\n⚠️  WARNINGS ({len(report.warnings)}):")
        for warning in report.warnings:
            print(f"  - {warning}")
    
    if report.errors:
        print(f"\n❌ ERRORS ({len(report.errors)}):")
        for error in report.errors:
            print(f"  - {error}")
    
    # Final verdict
    print(f"\n{'='*60}")
    can_use_heatsdf = report.has_point_cloud
    can_use_texture = can_use_heatsdf and report.has_colors
    
    print("✅ READY FOR:")
    if can_use_heatsdf:
        print("  - Basic HeatSDF (geometry only)")
    if can_use_texture:
        print("  - Textured HeatSDF (geometry + color)")
    if not can_use_heatsdf:
        print("  ❌ File is NOT ready for HeatSDF pipeline")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if report.has_point_cloud:
        if report.point_count < 10000:
            print(f"  - Point count ({report.point_count}) is low. Consider denser sampling.")
        elif report.point_count > 1000000:
            print(f"  - Point count ({report.point_count}) is very high. Consider downsampling for faster processing.")
        
        if report.density_stats and report.density_stats['density_variation'] > 0.5:
            print(f"  - Use adaptive weighting in HeatSDF due to density variation")
    
    if not report.has_colors:
        print(f"  - No color data found. Output will be geometry only.")
    
    print("="*60 + "\n")

def extract_data_for_heatsdf(filepath: str, output_dir: str = "./extracted"):
    """
    Extract and save the data needed for HeatSDF pipeline
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    scene = trimesh.load(filepath, force='scene')
    
    # Extract point cloud
    point_cloud = None
    colors = None
    camera_transforms = []
    
    if isinstance(scene, trimesh.Scene):
        # VGG-T specific: geometry_0 is the point cloud
        if 'geometry_0' in scene.geometry:
            geom = scene.geometry['geometry_0']
            if isinstance(geom, trimesh.PointCloud):
                point_cloud = geom.vertices
                if hasattr(geom, 'colors'):
                    colors = geom.colors
        
        # Extract camera transforms from other geometries
        for name, geom in scene.geometry.items():
            if name.startswith('geometry_') and name != 'geometry_0':
                if isinstance(geom, trimesh.Trimesh):
                    # Camera is represented as a mesh, get its center/transform
                    camera_pos = geom.vertices.mean(axis=0)
                    camera_transforms.append({
                        'name': name,
                        'position': camera_pos,
                        'mesh_bounds': (geom.vertices.min(axis=0), geom.vertices.max(axis=0))
                    })
    
    elif isinstance(scene, trimesh.PointCloud):
        point_cloud = scene.vertices
        if hasattr(scene, 'colors'):
            colors = scene.colors
    
    if point_cloud is not None:
        # Normalize colors if needed
        if colors is not None and colors.max() > 1.0:
            colors = colors.astype(np.float32) / 255.0
            print(f"✓ Normalized colors from 0-255 to 0-1 range")
        
        # Save as NPZ for easy loading
        save_data = {'positions': point_cloud}
        if colors is not None:
            save_data['colors'] = colors
        
        output_path = Path(output_dir) / f"{Path(filepath).stem}_extracted.npz"
        np.savez(output_path, **save_data)
        print(f"\n📁 Extracted data saved to: {output_path}")
        print(f"   - Points: {point_cloud.shape}")
        if colors is not None:
            print(f"   - Colors: {colors.shape} (normalized)")
        
        # Save camera info if found
        if camera_transforms:
            camera_path = Path(output_dir) / f"{Path(filepath).stem}_cameras.json"
            with open(camera_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                cameras_json = []
                for cam in camera_transforms:
                    cameras_json.append({
                        'name': cam['name'],
                        'position': cam['position'].tolist(),
                        'mesh_bounds': [cam['mesh_bounds'][0].tolist(), cam['mesh_bounds'][1].tolist()]
                    })
                json.dump(cameras_json, f, indent=2)
            print(f"   - Cameras: {len(camera_transforms)} saved to {camera_path}")
        
        # Also save as simple text files for debugging
        np.savetxt(Path(output_dir) / "positions.txt", point_cloud, fmt='%.6f')
        if colors is not None:
            np.savetxt(Path(output_dir) / "colors.txt", colors, fmt='%.6f')
    else:
        print("❌ No point cloud found to extract!")

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_vgg_t_glb.py <path_to_glb_file> [--extract]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    if not Path(filepath).exists():
        print(f"Error: File '{filepath}' not found")
        sys.exit(1)
    
    # Run validation
    report = analyze_glb(filepath)
    print_report(report, filepath)
    
    # Extract data if requested
    if '--extract' in sys.argv:
        extract_data_for_heatsdf(filepath)

if __name__ == "__main__":
    main()