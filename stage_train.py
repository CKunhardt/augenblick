#!/usr/bin/env python
"""
Unified staged training script for Neuralangelo
Usage: staged_train.py --stage 1|2|3 --config cfg.yaml [--checkpoint ckpt.pth]
"""
import argparse
import gc
import torch
import sys
import re

# Add Neuralangelo path
sys.path.insert(0, '/home/jhennessy7.gatech/augenblick/src/neuralangelo')

def clear_cache():
    """Clear GPU memory cache"""
    gc.collect()
    torch.cuda.empty_cache()
    # Note: torch.cuda.synchronize() removed as per ChatGPT's advice

# Import base trainer before patching
import imaginaire.trainers.base as base

# Store original methods
orig_init = base.BaseTrainer.__init__
orig_train_step = base.BaseTrainer._training_step
orig_log = base.BaseTrainer._log_loss
orig_validate = base.BaseTrainer._validate
orig_load_ckpt = base.BaseTrainer._load_checkpoint

def patch_for_stage(stage):
    """Apply stage-specific patches to the trainer"""
    
    # Stage-specific configurations
    stage_configs = {
        1: {
            'val_iter': 500,
            'cache_freq': 50,
            'log_every': 100,
            'target_iter': 2000,
            'base_iter': 0,
            'name': 'Coarse'
        },
        2: {
            'val_iter': 1000,
            'cache_freq': 75,
            'log_every': 200,
            'target_iter': 10000,
            'base_iter': 2000,
            'name': 'Mid-Resolution'
        },
        3: {
            'val_iter': 2000,
            'cache_freq': 100,
            'log_every': 200,
            'target_iter': 20000,
            'base_iter': 10000,
            'name': 'Fine Detail'
        }
    }
    
    cfg = stage_configs[stage]
    
    # Patch initialization
    def patched_init(self, config, *args, **kwargs):
        orig_init(self, config, *args, **kwargs)
        self.cfg.validation_iter = cfg['val_iter']
        self.cfg.image_save_iter = 999999999  # Disable image saving
        self._last_cache_clear = 0
        
        # Print stage info
        print("=" * 60)
        print(f"Stage {stage}: {cfg['name']} Training")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Resolution: {config.data.train.image_size}")
        print(f"  Target iterations: {cfg['target_iter']}")
        print(f"  Learning rate: {config.optim.params.lr}")
        print(f"  Batch size: {config.data.train.batch_size}")
        print(f"  Grad accumulation: {config.trainer.grad_accum_iter}")
        print(f"  Validation every: {cfg['val_iter']} iterations")
        print(f"  Cache clear every: {cfg['cache_freq']} iterations")
        print("")
    
    # Patch training step
    def patched_train_step(self, *args, **kwargs):
        result = orig_train_step(self, *args, **kwargs)
        
        # Clear cache periodically
        if self.current_iteration - self._last_cache_clear >= cfg['cache_freq']:
            clear_cache()
            self._last_cache_clear = self.current_iteration
        
        return result
    
    # Patch logging
    def patched_log(self, *args, **kwargs):
        orig_log(self, *args, **kwargs)
        
        # Progress logging
        if self.current_iteration % cfg['log_every'] == 0:
            progress = (self.current_iteration - cfg['base_iter']) / (cfg['target_iter'] - cfg['base_iter']) * 100
            print(f"\n[Stage {stage}] Iteration {self.current_iteration}/{cfg['target_iter']} - "
                  f"Stage Progress: {progress:.1f}%")
    
    # Patch validation
    def patched_validate(self, *args, **kwargs):
        # Log memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"\n[Memory] Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")
        
        result = orig_validate(self, *args, **kwargs)
        
        # Clear cache after validation
        clear_cache()
        
        return result
    
    # Apply patches
    base.BaseTrainer.__init__ = patched_init
    base.BaseTrainer._training_step = patched_train_step
    base.BaseTrainer._log_loss = patched_log
    base.BaseTrainer._validate = patched_validate
    
    # Special checkpoint handling for stages 2 and 3
    if stage > 1:
        def patched_load_checkpoint(self, config, checkpoint_path, *args, **kwargs):
            print(f"\n[CHECKPOINT] Loading from: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract iteration number
            if 'iteration' in checkpoint:
                resume_iter = checkpoint['iteration']
            else:
                # Try to extract from filename
                match = re.search(r'iteration_(\d+)', checkpoint_path)
                if match:
                    resume_iter = int(match.group(1))
                else:
                    resume_iter = cfg['base_iter']
            
            # Call original load function
            orig_load_ckpt(self, config, checkpoint_path, *args, **kwargs)
            
            # Force set the current iteration
            self.current_iteration = resume_iter
            self.current_epoch = resume_iter // len(self.train_data_loader) if hasattr(self, 'train_data_loader') else 0
            
            print(f"[CHECKPOINT] Resuming from iteration {resume_iter}")
            print(f"[CHECKPOINT] Training will continue to iteration {cfg['target_iter']}")
            
            # Update scheduler
            if hasattr(self, 'opt') and hasattr(self.opt, 'scheduler'):
                # Reset and step to correct iteration
                for _ in range(resume_iter):
                    self.opt.step_lr()
        
        base.BaseTrainer._load_checkpoint = patched_load_checkpoint

def main():
    parser = argparse.ArgumentParser(description='Unified staged training for Neuralangelo')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3], required=True,
                        help='Training stage (1=coarse, 2=mid, 3=fine)')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', help='Path to checkpoint for resuming')
    parser.add_argument('--logdir', help='Override log directory')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--show_pbar', action='store_true', help='Show progress bar')
    
    args, unknown_args = parser.parse_known_args()
    
    # Apply stage-specific patches
    patch_for_stage(args.stage)
    
    # Build command line for Neuralangelo's train.py
    train_args = ['train.py', '--config', args.config]
    
    if args.checkpoint:
        train_args.extend(['--checkpoint', args.checkpoint])
    
    if args.logdir:
        train_args.extend(['--logdir', args.logdir])
    
    train_args.extend(['--local_rank', str(args.local_rank)])
    
    if args.show_pbar:
        train_args.append('--show_pbar')
    
    # Add any unknown args
    train_args.extend(unknown_args)
    
    # Update sys.argv for train.py
    sys.argv = train_args
    
    # Import and run training
    from train import main as train_main
    train_main()

if __name__ == '__main__':
    main()
