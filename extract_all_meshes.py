#!/usr/bin/env python
"""
Extract meshes from all checkpoints in a directory
"""

import os
import sys
import glob
import argparse

# Fix CUDA environment before any torch imports
os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)

def extract_mesh_from_checkpoint(config_path, checkpoint_path, output_path, resolution, block_res, single_gpu=True, keep_lcc=False):
    """Extract mesh from a single checkpoint"""
    cmd = [
        sys.executable,
        os.path.expanduser("~/augenblick/extract_mesh_complete.py"),
        "--config", config_path,
        "--checkpoint", checkpoint_path,
        "--output_file", output_path,
        "--resolution", str(resolution),
        "--block_res", str(block_res)
    ]
    
    if single_gpu:
        cmd.append("--single_gpu")
    if keep_lcc:
        cmd.append("--keep_lcc")
    
    print(f"\nExtracting mesh from: {os.path.basename(checkpoint_path)}")
    print(f"Output: {output_path}")
    
    import subprocess
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Success")
        # Parse vertices/faces from output
        for line in result.stdout.split('\n'):
            if 'Vertices:' in line or 'Faces:' in line:
                print(f"  {line.strip()}")
    else:
        print("✗ Failed")
        print(result.stderr)
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Extract meshes from all checkpoints")
    parser.add_argument("--checkpoint_dir", required=True, help="Directory containing checkpoints")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--output_dir", help="Output directory (default: checkpoint_dir/meshes)")
    parser.add_argument("--resolution", default=512, type=int, help="Mesh resolution")
    parser.add_argument("--block_res", default=128, type=int, help="Block resolution")
    parser.add_argument("--single_gpu", action="store_true", default=True)
    parser.add_argument("--keep_lcc", action="store_true", help="Keep only largest component")
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoint_dir, "meshes")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all checkpoints
    checkpoint_pattern = os.path.join(args.checkpoint_dir, "*checkpoint.pt")
    checkpoints = glob.glob(checkpoint_pattern)
    checkpoints.sort()
    
    print(f"Found {len(checkpoints)} checkpoints:")
    for cp in checkpoints:
        print(f"  - {os.path.basename(cp)}")
    
    # Extract mesh from each checkpoint
    successful = 0
    failed = 0
    
    for checkpoint_path in checkpoints:
        # Create output filename based on checkpoint name
        cp_name = os.path.basename(checkpoint_path)
        if "iteration" in cp_name:
            # Extract iteration number
            parts = cp_name.split('_')
            iter_num = None
            for part in parts:
                if part.startswith('0'):
                    iter_num = part
                    break
            output_name = f"mesh_iter_{iter_num}.ply" if iter_num else cp_name.replace('.pt', '.ply')
        else:
            output_name = cp_name.replace('.pt', '.ply')
        
        output_path = os.path.join(args.output_dir, output_name)
        
        # Extract mesh
        success = extract_mesh_from_checkpoint(
            args.config,
            checkpoint_path,
            output_path,
            args.resolution,
            args.block_res,
            args.single_gpu,
            args.keep_lcc
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Meshes saved to: {args.output_dir}")
    
    # List output meshes
    meshes = glob.glob(os.path.join(args.output_dir, "*.ply"))
    if meshes:
        print(f"\nGenerated meshes:")
        for mesh in sorted(meshes):
            size = os.path.getsize(mesh) / (1024*1024)  # MB
            print(f"  - {os.path.basename(mesh)} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
