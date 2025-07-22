import open3d as o3d
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Check if a PLY file was provided
if len(sys.argv) > 1:
    ply_path = sys.argv[1]
else:
    print("Usage: python viz_ply.py path/to/point_cloud.ply")
    sys.exit(1)

# Load the point cloud
print(f"Loading point cloud from: {ply_path}")
try:
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    print(f"Point cloud contains {len(points)} points")
    
    # Create output directory
    output_dir = "point_cloud_viz"
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.basename(ply_path).replace(".ply", "")
    
    # Save point cloud statistics
    info_path = os.path.join(output_dir, f"{base_filename}_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Point cloud statistics:\n")
        f.write(f"Number of points: {len(points)}\n")
        f.write(f"Bounding box min: {points.min(axis=0)}\n")
        f.write(f"Bounding box max: {points.max(axis=0)}\n")
        f.write(f"Centroid: {points.mean(axis=0)}\n")
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            f.write(f"Has color information: Yes\n")
            f.write(f"Color range: Min {colors.min(axis=0)}, Max {colors.max(axis=0)}\n")
        else:
            f.write(f"Has color information: No\n")
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
            f.write(f"Has normal information: Yes\n")
            f.write(f"Normal range: Min {normals.min(axis=0)}, Max {normals.max(axis=0)}\n")
        else:
            f.write(f"Has normal information: No\n")
    print(f"Saved point cloud information to: {info_path}")
    
    # Generate 2D projections using matplotlib (no OpenGL required)
    print("Generating 2D projections...")
    
    # Create color map based on height (Z coordinate)
    colors = plt.cm.viridis(Normalize(points[:, 2].min(), points[:, 2].max())(points[:, 2]))
    
    # Top view (XY plane)
    plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 0], points[:, 1], s=0.5, c=colors)
    plt.axis('equal')
    plt.title(f'Top View (XY) - {len(points)} points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(os.path.join(output_dir, f"{base_filename}_top_view.png"), dpi=150)
    plt.close()
    
    # Side view (XZ plane)
    plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 0], points[:, 2], s=0.5, c=colors)
    plt.axis('equal')
    plt.title(f'Side View (XZ) - {len(points)} points')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.savefig(os.path.join(output_dir, f"{base_filename}_side_view.png"), dpi=150)
    plt.close()
    
    # Front view (YZ plane)
    plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 1], points[:, 2], s=0.5, c=colors)
    plt.axis('equal')
    plt.title(f'Front View (YZ) - {len(points)} points')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.savefig(os.path.join(output_dir, f"{base_filename}_front_view.png"), dpi=150)
    plt.close()
    
    # If the point cloud has too many points, create a downsampled version for metadata display
    if len(points) > 10000:
        print("Creating downsampled visualization...")
        pcd_down = pcd.voxel_down_sample(voxel_size=0.05)
        down_points = np.asarray(pcd_down.points)
        print(f"Downsampled to {len(down_points)} points")
        
        # Save downsampled point coordinates
        np.savetxt(os.path.join(output_dir, f"{base_filename}_downsampled.xyz"), down_points, 
                  header=f"X Y Z - Downsampled from {len(points)} to {len(down_points)} points", 
                  comments='# ')
    
    print(f"All visualizations saved to {output_dir}/ directory")

except Exception as e:
    print(f"Error processing point cloud: {e}")

print("Done.")

