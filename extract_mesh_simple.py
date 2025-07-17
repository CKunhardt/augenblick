#!/usr/bin/env python
"""
Simple mesh extraction script without distributed training
"""

import os
import sys
import argparse
import torch

# Fix environment
os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

sys.path.append(os.getcwd())

from imaginaire.config import Config
from projects.neuralangelo.utils.mesh import extract_mesh

def load_model_from_checkpoint(config_path, checkpoint_path):
    """Load model from checkpoint without trainer"""
    # Load config
    cfg = Config(config_path)
    cfg.local_rank = 0
    
    # Import model
    from projects.neuralangelo.model import Model
    
    # Create model
    model = Model(cfg.model)
    model.cuda()
    model.eval()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    print("Model loaded successfully")
    
    return model, cfg

def extract_mesh_simple(config_path, checkpoint_path, output_path, resolution=512, block_res=128):
    """Extract mesh from checkpoint"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load model
    model, cfg = load_model_from_checkpoint(config_path, checkpoint_path)
    
    # Extract mesh
    print(f"Extracting mesh at resolution {resolution}...")
    mesh = extract_mesh(
        model,
        resolution=(resolution, resolution, resolution),
        block_res=block_res,
        keep_lcc=True
    )
    
    if mesh is None:
        print("Error: Mesh extraction failed")
        return False
    
    # Save mesh
    mesh.export(output_path)
    print(f"Mesh saved to: {output_path}")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")
    
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--block_res", type=int, default=128)
    
    args = parser.parse_args()
    
    extract_mesh_simple(
        args.config,
        args.checkpoint,
        args.output,
        args.resolution,
        args.block_res
    )

if __name__ == "__main__":
    main()
