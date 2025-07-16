#!/usr/bin/env python
"""
Complete mesh extraction script for Neuralangelo
"""

import os
import sys
import argparse

# Fix CUDA environment before any torch imports
os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)

# Add current directory to path
sys.path.append(os.getcwd())

# Now import everything else
from imaginaire.config import Config, recursive_update_strict, parse_cmdline_arguments
from imaginaire.utils.distributed import init_dist, get_world_size, is_master, master_only_print as print
from imaginaire.utils.gpu_affinity import set_affinity
from imaginaire.trainers.utils.get_trainer import get_trainer
from projects.neuralangelo.utils.mesh import extract_mesh, extract_texture


def parse_args():
    parser = argparse.ArgumentParser(description="Extract mesh from Neuralangelo checkpoint")
    parser.add_argument("--config", required=True, help="Path to the training config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file.")
    parser.add_argument("--output_file", default="mesh.ply", type=str, help="Output mesh filename")
    parser.add_argument("--resolution", default=512, type=int, help="Marching cubes resolution")
    parser.add_argument("--block_res", default=128, type=int, help="Block resolution for marching cubes")
    parser.add_argument("--textured", action="store_true", help="Export mesh with texture")
    parser.add_argument("--keep_lcc", action="store_true", help="Keep only largest connected component")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--single_gpu', action='store_true')
    args, cfg_cmd = parser.parse_known_args()
    return args, cfg_cmd


def main():
    args, cfg_cmd = parse_args()
    
    # Set GPU affinity
    set_affinity(args.local_rank)
    
    # Load config
    cfg = Config(args.config)
    cfg_cmd = parse_cmdline_arguments(cfg_cmd)
    recursive_update_strict(cfg, cfg_cmd)
    
    # Initialize distributed training
    cfg.local_rank = args.local_rank
    init_dist(cfg.local_rank, rank=-1, world_size=-1)
    
    # Print info
    print(f"Extracting mesh from checkpoint: {args.checkpoint}")
    print(f"Resolution: {args.resolution}")
    print(f"Output file: {args.output_file}")
    
    # Setup trainer and load checkpoint
    trainer = get_trainer(cfg, is_inference=True, seed=0)
    trainer.checkpointer.load(args.checkpoint, load_sch=False, load_opt=False)
    print("Checkpoint loaded successfully")
    
    # Extract mesh
    mesh = extract_mesh(
        trainer.model,
        resolution=(args.resolution, args.resolution, args.resolution),
        block_res=args.block_res,
        keep_lcc=args.keep_lcc
    )
    
    if mesh is None:
        print("Error: Mesh extraction failed")
        return
    
    # Export mesh
    mesh.export(args.output_file)
    print(f"Mesh saved to: {args.output_file}")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")
    
    # Extract texture if requested
    if args.textured:
        print("Extracting texture...")
        texture_file = args.output_file.replace('.ply', '_textured.ply')
        textured_mesh = extract_texture(
            trainer.model,
            mesh,
            resolution=1024
        )
        textured_mesh.export(texture_file)
        print(f"Textured mesh saved to: {texture_file}")


if __name__ == "__main__":
    main()
