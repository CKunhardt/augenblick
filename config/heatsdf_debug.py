#!/usr/bin/env python3
"""
Debug script for VGG-T to HeatSDF processed data
Analyzes the output of vgg_t_loader.py to ensure data is ready for HeatSDF training
"""

import sys
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
import torch

def load_processed_data(npz_path: str) -> Dict:
    """Load the processed NPZ file"""
    data = np.load(npz_path, allow_pickle=True)
    return {key: data[key] for key in data.files}

def analyze_positions(positions: np.ndarray) -> Dict:
    """Analyze the spatial distribution of positions"""
    analysis = {}
    
    # Basic statistics
    analysis['count'] = len(positions)
    analysis['shape'] = positions.shape
    analysis['dtype'] = str(positions.dtype)
    
    # Bounds check (should be in [-1, 1])
    analysis['min_coords'] = positions.min(axis=0)
    analysis['max_coords'] = positions.max(axis=0)
    analysis['center'] = positions.mean(axis=0)
    
    # Check if properly normalized
    abs_max = np.abs(positions).max()
    analysis['abs_max'] = abs_max
    analysis['properly_normalized'] = abs_max <= 1.0
    
    # Density analysis
    from scipy.spatial import KDTree
    tree = KDTree(positions)
    
    # Sample subset for efficiency
    sample_size = min(10000, len(positions))
    sample_indices = np.random.choice(len(positions), sample_size, replace=False)
    sample_points = positions[sample_indices]
    
    # Find nearest neighbors
    distances, _ = tree.query(sample_points, k=11)  # k=11 to exclude self
    nn_distances = distances[:, 1:]  # Remove self-distance
    
    analysis['density'] = {
        'mean_nn_distance': float(nn_distances.mean()),
        'std_nn_distance': float(nn_distances.std()),
        'min_nn_distance': float(nn_distances.min()),
        'max_nn_distance': float(nn_distances.max()),
        'median_nn_distance': float(np.median(nn_distances)),
    }
    
    # Spatial distribution
    analysis['spatial_std'] = positions.std(axis=0)
    
    # Check for outliers
    distances_from_center = np.linalg.norm(positions - analysis['center'], axis=1)
    analysis['outliers'] = {
        'count': int((distances_from_center > 1.5).sum()),
        'percentage': float((distances_from_center > 1.5).sum() / len(positions) * 100)
    }
    
    return analysis

def analyze_colors(colors: np.ndarray) -> Dict:
    """Analyze color distribution"""
    analysis = {}
    
    analysis['shape'] = colors.shape
    analysis['dtype'] = str(colors.dtype)
    analysis['channels'] = colors.shape[1] if len(colors.shape) > 1 else 1
    
    # Range check (should be in [0, 1])
    analysis['min_values'] = colors.min(axis=0).tolist()
    analysis['max_values'] = colors.max(axis=0).tolist()
    analysis['mean_values'] = colors.mean(axis=0).tolist()
    
    # Check if properly normalized
    analysis['properly_normalized'] = colors.min() >= 0 and colors.max() <= 1.0
    
    # Color statistics per channel
    if len(colors.shape) > 1:
        for i in range(colors.shape[1]):
            channel_name = ['R', 'G', 'B', 'A'][i] if i < 4 else f'Channel_{i}'
            analysis[f'{channel_name}_stats'] = {
                'mean': float(colors[:, i].mean()),
                'std': float(colors[:, i].std()),
                'min': float(colors[:, i].min()),
                'max': float(colors[:, i].max())
            }
    
    # Check for monochrome
    if colors.shape[1] >= 3:
        color_variance = colors[:, :3].var(axis=1).mean()
        analysis['is_monochrome'] = color_variance < 0.01
    
    return analysis

def analyze_training_readiness(data: Dict) -> Dict:
    """Check if data is ready for HeatSDF training"""
    readiness = {
        'ready': True,
        'issues': [],
        'warnings': []
    }
    
    # Check positions
    if 'positions' not in data:
        readiness['ready'] = False
        readiness['issues'].append("No positions found in data")
    else:
        positions = data['positions']
        if positions.shape[1] != 3:
            readiness['ready'] = False
            readiness['issues'].append(f"Positions should be 3D, got shape {positions.shape}")
        
        if np.abs(positions).max() > 1.0:
            readiness['ready'] = False
            readiness['issues'].append("Positions not normalized to [-1, 1]")
        
        if len(positions) < 1000:
            readiness['warnings'].append(f"Very few points ({len(positions)}), may affect quality")
        elif len(positions) > 10000000:
            readiness['warnings'].append(f"Very many points ({len(positions)}), may be slow")
    
    # Check normalization data
    if 'normalization' not in data:
        readiness['warnings'].append("No normalization parameters saved")
    
    # Check colors (optional but recommended)
    if 'colors' not in data:
        readiness['warnings'].append("No color data found - will produce geometry only")
    else:
        colors = data['colors']
        if colors.min() < 0 or colors.max() > 1:
            readiness['warnings'].append("Colors not in [0, 1] range")
    
    return readiness

def create_visualization(data: Dict, output_path: str):
    """Create visualization plots of the processed data"""
    positions = data.get('positions', None)
    colors = data.get('colors', None)
    
    if positions is None:
        print("No positions to visualize")
        return
    
    # Sample for visualization
    sample_size = min(50000, len(positions))
    indices = np.random.choice(len(positions), sample_size, replace=False)
    sample_pos = positions[indices]
    sample_colors = colors[indices] if colors is not None else None
    
    fig = plt.figure(figsize=(15, 10))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    if sample_colors is not None and sample_colors.shape[1] >= 3:
        ax1.scatter(sample_pos[:, 0], sample_pos[:, 1], sample_pos[:, 2], 
                   c=sample_colors[:, :3], s=1, alpha=0.5)
    else:
        ax1.scatter(sample_pos[:, 0], sample_pos[:, 1], sample_pos[:, 2], 
                   s=1, alpha=0.5)
    ax1.set_title(f'3D Point Cloud ({sample_size} points)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_zlim(-1.1, 1.1)
    
    # XY projection
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(sample_pos[:, 0], sample_pos[:, 1], s=1, alpha=0.5)
    ax2.set_title('XY Projection')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # XZ projection
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(sample_pos[:, 0], sample_pos[:, 2], s=1, alpha=0.5)
    ax3.set_title('XZ Projection')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_xlim(-1.1, 1.1)
    ax3.set_ylim(-1.1, 1.1)
    ax3.grid(True, alpha=0.3)
    
    # Density histogram
    ax4 = fig.add_subplot(2, 3, 4)
    distances = np.linalg.norm(sample_pos, axis=1)
    ax4.hist(distances, bins=50, alpha=0.7, edgecolor='black')
    ax4.set_title('Distance from Origin Distribution')
    ax4.set_xlabel('Distance')
    ax4.set_ylabel('Count')
    ax4.axvline(x=1.0, color='r', linestyle='--', label='Unit sphere')
    ax4.legend()
    
    # Color distribution (if available)
    if colors is not None and colors.shape[1] >= 3:
        ax5 = fig.add_subplot(2, 3, 5)
        color_labels = ['Red', 'Green', 'Blue']
        for i in range(3):
            ax5.hist(colors[:, i], bins=50, alpha=0.5, label=color_labels[i], 
                    color=['red', 'green', 'blue'][i])
        ax5.set_title('Color Channel Distribution')
        ax5.set_xlabel('Value')
        ax5.set_ylabel('Count')
        ax5.set_xlim(0, 1)
        ax5.legend()
    
    # Coordinate distribution
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.boxplot([positions[:, 0], positions[:, 1], positions[:, 2]], 
                labels=['X', 'Y', 'Z'])
    ax6.set_title('Coordinate Distribution')
    ax6.set_ylabel('Value')
    ax6.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    ax6.axhline(y=-1.0, color='r', linestyle='--', alpha=0.5)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

def print_report(data: Dict, npz_path: str):
    """Print comprehensive analysis report"""
    print("\n" + "="*60)
    print("VGG-T TO HEATSDF PROCESSED DATA ANALYSIS")
    print(f"File: {Path(npz_path).name}")
    print("="*60)
    
    # Analyze positions
    if 'positions' in data:
        pos_analysis = analyze_positions(data['positions'])
        
        print("\n📍 POSITION DATA:")
        print(f"  Count: {pos_analysis['count']:,} points")
        print(f"  Shape: {pos_analysis['shape']}")
        print(f"  Data type: {pos_analysis['dtype']}")
        print(f"  Bounds: [{pos_analysis['min_coords']}] to [{pos_analysis['max_coords']}]")
        print(f"  Center: {pos_analysis['center']}")
        print(f"  Properly normalized: {'✅ YES' if pos_analysis['properly_normalized'] else '❌ NO'}")
        
        print(f"\n  Density Statistics:")
        for key, value in pos_analysis['density'].items():
            print(f"    {key}: {value:.6f}")
        
        print(f"\n  Spatial Distribution:")
        print(f"    Std dev per axis: {pos_analysis['spatial_std']}")
        print(f"    Outliers (>1.5 units): {pos_analysis['outliers']['count']} ({pos_analysis['outliers']['percentage']:.2f}%)")
    
    # Analyze colors
    if 'colors' in data:
        color_analysis = analyze_colors(data['colors'])
        
        print("\n🎨 COLOR DATA:")
        print(f"  Shape: {color_analysis['shape']}")
        print(f"  Channels: {color_analysis['channels']}")
        print(f"  Data type: {color_analysis['dtype']}")
        print(f"  Range: {color_analysis['min_values']} to {color_analysis['max_values']}")
        print(f"  Properly normalized: {'✅ YES' if color_analysis['properly_normalized'] else '❌ NO'}")
        
        if 'R_stats' in color_analysis:
            print(f"\n  Channel Statistics:")
            for channel in ['R', 'G', 'B', 'A']:
                key = f'{channel}_stats'
                if key in color_analysis:
                    stats = color_analysis[key]
                    print(f"    {channel}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, range=[{stats['min']:.3f}, {stats['max']:.3f}]")
        
        if 'is_monochrome' in color_analysis:
            print(f"  Monochrome: {'Yes' if color_analysis['is_monochrome'] else 'No'}")
    
    # Check normalization parameters
    if 'normalization' in data:
        norm = data['normalization'].item() if hasattr(data['normalization'], 'item') else data['normalization']
        print("\n📏 NORMALIZATION PARAMETERS:")
        print(f"  Center: {norm.get('center', 'Not found')}")
        print(f"  Scale: {norm.get('scale', 'Not found')}")
    
    # Training readiness
    readiness = analyze_training_readiness(data)
    
    print("\n🚀 HEATSDF TRAINING READINESS:")
    print(f"  Ready: {'✅ YES' if readiness['ready'] else '❌ NO'}")
    
    if readiness['issues']:
        print(f"\n  ❌ Issues ({len(readiness['issues'])}):")
        for issue in readiness['issues']:
            print(f"    - {issue}")
    
    if readiness['warnings']:
        print(f"\n  ⚠️  Warnings ({len(readiness['warnings'])}):")
        for warning in readiness['warnings']:
            print(f"    - {warning}")
    
    # Memory estimation
    if 'positions' in data:
        pos_mem = data['positions'].nbytes / (1024**2)  # MB
        color_mem = data['colors'].nbytes / (1024**2) if 'colors' in data else 0
        total_mem = pos_mem + color_mem
        
        print("\n💾 MEMORY USAGE:")
        print(f"  Positions: {pos_mem:.2f} MB")
        if 'colors' in data:
            print(f"  Colors: {color_mem:.2f} MB")
        print(f"  Total: {total_mem:.2f} MB")
        
        # Training memory estimate (rough)
        network_mem = 256 * 256 * 4 * 4 / (1024**2)  # Rough estimate for SIREN
        batch_mem = 10000 * 3 * 4 / (1024**2)  # 10k batch size
        est_gpu_mem = total_mem + network_mem * 2 + batch_mem * 10  # Very rough
        
        print(f"\n  Estimated GPU memory for training: ~{est_gpu_mem:.0f} MB")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if 'positions' in data:
        count = len(data['positions'])
        if count > 1000000:
            print(f"  - Consider downsampling from {count:,} points for initial experiments")
        if count < 10000:
            print(f"  - Point count ({count:,}) is low, may need to adjust batch size")
    
    if 'colors' not in data:
        print("  - No colors found - will produce untextured geometry only")
    
    if not readiness['ready']:
        print("  - Fix the issues above before training HeatSDF")
    else:
        print("  - Data is ready for HeatSDF training!")
    
    print("="*60 + "\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_processed_data.py <path_to_npz> [--visualize]")
        sys.exit(1)
    
    npz_path = sys.argv[1]
    
    if not Path(npz_path).exists():
        print(f"Error: File '{npz_path}' not found")
        sys.exit(1)
    
    # Load data
    try:
        data = load_processed_data(npz_path)
    except Exception as e:
        print(f"Error loading NPZ file: {e}")
        sys.exit(1)
    
    # Print analysis report
    print_report(data, npz_path)
    
    # Create visualization if requested
    if '--visualize' in sys.argv:
        vis_path = Path(npz_path).stem + "_analysis.png"
        create_visualization(data, vis_path)

if __name__ == "__main__":
    main()