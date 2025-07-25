#!/usr/bin/env python3
"""
Monitor Neuralangelo training progress and validate it's actually learning
"""

import json
import time
import sys
from pathlib import Path
import numpy as np
import re
from datetime import datetime

def parse_log_file(log_path):
    """Parse Neuralangelo training log for loss values"""
    losses = {
        'total': [],
        'rgb': [],
        'depth': [],
        'eikonal': [],
        'normal': [],
        'iterations': [],
        'epochs': [],
        'learning_rate': []
    }
    
    if not log_path.exists():
        return losses
    
    with open(log_path, 'r') as f:
        for line in f:
            # Neuralangelo uses specific logging format
            # Look for patterns like:
            # [Epoch 1/100, Iter 50/1000] loss: 0.1234, rgb: 0.0123, eikonal: 0.0012
            # [INFO] Iter 1000: loss=0.5432 rgb=0.4321 depth=0.1234
            
            # Extract iteration/epoch
            iter_match = re.search(r'(?:Iter|iteration)[:\s]+(\d+)', line, re.IGNORECASE)
            epoch_match = re.search(r'Epoch[:\s]+(\d+)', line, re.IGNORECASE)
            
            if iter_match:
                iteration = int(iter_match.group(1))
                losses['iterations'].append(iteration)
                
                # Look for loss values in the same line
                # Total loss
                loss_patterns = [
                    r'loss[:\s=]+([\d.]+)',
                    r'total_loss[:\s=]+([\d.]+)',
                    r'train/loss[:\s=]+([\d.]+)'
                ]
                for pattern in loss_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        losses['total'].append(float(match.group(1)))
                        break
                
                # RGB loss
                rgb_match = re.search(r'(?:rgb|color)[:\s=]+([\d.]+)', line, re.IGNORECASE)
                if rgb_match:
                    losses['rgb'].append(float(rgb_match.group(1)))
                
                # Depth loss (if using depth supervision)
                depth_match = re.search(r'depth[:\s=]+([\d.]+)', line, re.IGNORECASE)
                if depth_match:
                    losses['depth'].append(float(depth_match.group(1)))
                
                # Eikonal loss (SDF regularization)
                eikonal_match = re.search(r'eikonal[:\s=]+([\d.]+)', line, re.IGNORECASE)
                if eikonal_match:
                    losses['eikonal'].append(float(eikonal_match.group(1)))
                
                # Normal loss
                normal_match = re.search(r'normal[:\s=]+([\d.]+)', line, re.IGNORECASE)
                if normal_match:
                    losses['normal'].append(float(normal_match.group(1)))
                
                # Learning rate
                lr_match = re.search(r'lr[:\s=]+([\d.e-]+)', line, re.IGNORECASE)
                if lr_match:
                    losses['learning_rate'].append(float(lr_match.group(1)))
            
            if epoch_match:
                losses['epochs'].append(int(epoch_match.group(1)))
            
            # Also check for validation metrics
            if 'val' in line.lower() and 'psnr' in line.lower():
                psnr_match = re.search(r'psnr[:\s=]+([\d.]+)', line, re.IGNORECASE)
                if psnr_match and 'val_psnr' not in losses:
                    losses['val_psnr'] = []
                if psnr_match:
                    losses['val_psnr'].append(float(psnr_match.group(1)))
    
    return losses

def check_training_progress(work_dir, min_iterations=100):
    """Check if training is making progress"""
    
    logs_dir = Path(work_dir) / "logs"
    
    # Find the latest log file - Neuralangelo uses different log structure
    log_files = []
    
    # Check for rank-specific logs (distributed training)
    rank_dirs = list(logs_dir.glob("rank*"))
    for rank_dir in rank_dirs:
        log_files.extend(list(rank_dir.glob("log.txt")))
    
    # Also check main logs directory
    log_files.extend(list(logs_dir.glob("*.log")))
    log_files.extend(list(logs_dir.glob("log.txt")))
    
    if not log_files:
        return False, "No log files found"
    
    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
    
    # Parse losses
    losses = parse_log_file(latest_log)
    
    # Check if we have enough iterations
    if not losses['iterations']:
        return False, "No iterations found in log"
    
    current_iter = losses['iterations'][-1] if losses['iterations'] else 0
    
    if current_iter < min_iterations:
        return False, f"Only {current_iter} iterations completed (minimum: {min_iterations})"
    
    # Check if training is stuck at epoch boundaries
    if len(losses['iterations']) > 20:
        recent_iters = losses['iterations'][-20:]
        if len(set(recent_iters)) == 1:
            return False, f"Training appears stuck at iteration {recent_iters[0]}"
    
    # Check if loss is decreasing
    if losses['total'] and len(losses['total']) > 20:
        early_loss = np.mean(losses['total'][:20])
        recent_loss = np.mean(losses['total'][-20:])
        
        # More lenient check - loss should decrease or at least not increase much
        if recent_loss > early_loss * 1.1:  # Allow 10% increase due to regularization changes
            return False, f"Loss increasing: early={early_loss:.4f}, recent={recent_loss:.4f}"
        
        # Check if loss is changing at all
        recent_std = np.std(losses['total'][-50:]) if len(losses['total']) > 50 else 1.0
        if recent_std < 1e-6:
            return False, f"Loss not changing (std={recent_std:.2e}) - possible convergence or stuck training"
        
        return True, f"Training progressing. Iter {current_iter}, loss decreased from {early_loss:.4f} to {recent_loss:.4f}"
    
    # If we can't determine from losses, check if iterations are increasing
    if len(losses['iterations']) > 1:
        if losses['iterations'][-1] > losses['iterations'][0]:
            return True, f"Training running at iteration {current_iter}"
    
    return False, "Unable to determine training progress"

def check_neuralangelo_config(work_dir):
    """Check if Neuralangelo config is set up correctly"""
    config_path = Path(work_dir) / "logs" / "config.yaml"
    issues = []
    
    if not config_path.exists():
        # Try alternative location
        config_path = Path(work_dir) / "config.yaml"
    
    if config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check critical settings
        if 'data' in config:
            data_config = config['data']
            if 'train' in data_config:
                batch_size = data_config['train'].get('batch_size', 0)
                if batch_size == 0:
                    issues.append("Batch size is 0 - no data will be loaded!")
                
                # Check data root
                data_root = data_config.get('root', '')
                if not Path(data_root).exists():
                    issues.append(f"Data root does not exist: {data_root}")
        
        if 'model' in config:
            model_config = config['model']
            # Check if depth supervision is properly configured
            if 'depth_supervision' in model_config:
                depth_enabled = model_config['depth_supervision'].get('enabled', False)
                if depth_enabled:
                    # Check if depth maps actually exist
                    depth_dir = Path(data_root) / "depth_maps" if 'data_root' in locals() else None
                    if depth_dir and not depth_dir.exists():
                        issues.append("Depth supervision enabled but no depth_maps directory found")
        
        # Check training settings
        if 'training' in config:
            max_iter = config['training'].get('max_iter', 0)
            if max_iter == 0:
                issues.append("max_iter is 0 - training will not run!")
    else:
        issues.append("Config file not found")
    
    return issues

def check_data_loading(work_dir):
    """Check if data is being loaded correctly"""
    logs_dir = Path(work_dir) / "logs"
    
    # Look for data loading messages
    log_files = list(logs_dir.glob("**/*.log")) + list(logs_dir.glob("**/log.txt"))
    
    data_loading_ok = False
    loading_errors = []
    
    for log_file in log_files:
        with open(log_file, 'r') as f:
            content = f.read()
            
            # Check for successful data loading
            if "train data loader" in content.lower() or "dataset size" in content.lower():
                data_loading_ok = True
            
            # Check for common errors
            if "no such file or directory" in content.lower():
                loading_errors.append("File not found errors detected")
            if "cuda out of memory" in content.lower():
                loading_errors.append("CUDA out of memory errors")
            if "dataset is empty" in content.lower():
                loading_errors.append("Dataset is empty")
    
    return data_loading_ok, loading_errors

def monitor_training(work_dir, check_interval=60, max_stall_time=600):
    """
    Monitor training progress in real-time
    
    Args:
        work_dir: Training directory
        check_interval: Seconds between checks
        max_stall_time: Maximum seconds without progress before warning
    """
    
    work_dir = Path(work_dir)
    last_progress_time = time.time()
    last_iteration = 0
    
    print(f"Monitoring training in: {work_dir}")
    print(f"Check interval: {check_interval}s")
    print(f"Max stall time: {max_stall_time}s")
    print("-" * 60)
    
    while True:
        try:
            # Check logs
            is_progressing, message = check_training_progress(work_dir, min_iterations=10)
            
            # Check latest checkpoint
            checkpoint_dir = work_dir / "logs" / "checkpoints"
            if checkpoint_dir.exists():
                checkpoints = sorted(checkpoint_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
                if checkpoints:
                    latest_checkpoint = checkpoints[-1]
                    ckpt_valid, ckpt_msg = check_checkpoint_validity(latest_checkpoint)
                    
                    # Extract iteration from checkpoint name
                    iter_match = re.search(r'iter_(\d+)', latest_checkpoint.name)
                    if iter_match:
                        current_iteration = int(iter_match.group(1))
                        if current_iteration > last_iteration:
                            last_progress_time = time.time()
                            last_iteration = current_iteration
            
            # Print status with Neuralangelo-specific metrics
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}] Status Update:")
            print(f"  Progress: {message}")
            print(f"  Current iteration: {last_iteration}")
            
            # Parse latest losses if available
            logs_dir = work_dir / "logs"
            log_files = list(logs_dir.glob("**/*.log")) + list(logs_dir.glob("**/log.txt"))
            if log_files:
                latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
                losses = parse_log_file(latest_log)
                
                if losses['total']:
                    print(f"  Latest loss: {losses['total'][-1]:.4f}")
                    if len(losses['total']) > 10:
                        recent_avg = np.mean(losses['total'][-10:])
                        print(f"  Recent avg loss: {recent_avg:.4f}")
                
                if losses['learning_rate']:
                    print(f"  Learning rate: {losses['learning_rate'][-1]:.2e}")
                
                if 'val_psnr' in losses and losses['val_psnr']:
                    print(f"  Val PSNR: {losses['val_psnr'][-1]:.2f} dB")
            
            # Check for Neuralangelo-specific issues
            config_issues = check_neuralangelo_config(work_dir)
            if config_issues:
                print("  ⚠️  Config issues:")
                for issue in config_issues:
                    print(f"    - {issue}")
            
            data_ok, data_errors = check_data_loading(work_dir)
            if not data_ok and last_iteration == 0:
                print("  ⚠️  Data loading may have issues")
            if data_errors:
                for error in data_errors:
                    print(f"    - {error}")
            
            if is_progressing:
                print(f"  ✓ Training is progressing normally")
            else:
                time_since_progress = time.time() - last_progress_time
                if time_since_progress > max_stall_time:
                    print(f"  ⚠️  WARNING: No progress for {time_since_progress:.0f}s")
                    print("  Possible issues:")
                    print("    - Training might be stuck")
                    print("    - Loss might have converged")
                    print("    - Check GPU utilization with nvidia-smi")
            
            # Check GPU utilization
            import subprocess
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    gpu_util = int(result.stdout.strip())
                    print(f"  GPU Utilization: {gpu_util}%")
                    if gpu_util < 50:
                        print("  ⚠️  Low GPU utilization detected")
            except:
                pass
            
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"Error during monitoring: {e}")
            time.sleep(check_interval)

def generate_validation_report(work_dir):
    """Generate a comprehensive validation report"""
    
    work_dir = Path(work_dir)
    report = {
        'timestamp': datetime.now().isoformat(),
        'work_dir': str(work_dir),
        'training_valid': False,
        'issues': [],
        'metrics': {}
    }
    
    # Check training progress
    is_progressing, message = check_training_progress(work_dir)
    report['training_valid'] = is_progressing
    if not is_progressing:
        report['issues'].append(f"Training progress: {message}")
    
    # Check data loading
    transforms_path = work_dir / "transforms.json"
    if transforms_path.exists():
        with open(transforms_path, 'r') as f:
            transforms = json.load(f)
        
        num_frames = len(transforms.get('frames', []))
        report['metrics']['num_frames'] = num_frames
        
        if num_frames == 0:
            report['issues'].append("No frames in transforms.json")
        
        # Check if images exist
        missing_images = 0
        for frame in transforms.get('frames', [])[:5]:  # Check first 5
            img_path = work_dir / frame['file_path']
            if not img_path.exists():
                missing_images += 1
        
        if missing_images > 0:
            report['issues'].append(f"{missing_images} missing images detected")
    else:
        report['issues'].append("transforms.json not found")
    
    # Save report
    report_path = work_dir / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor Neuralangelo training")
    parser.add_argument("work_dir", help="Training work directory")
    parser.add_argument("--mode", choices=["monitor", "validate"], default="monitor",
                       help="Mode: continuous monitoring or one-time validation")
    parser.add_argument("--interval", type=int, default=60,
                       help="Check interval in seconds (for monitor mode)")
    
    args = parser.parse_args()
    
    if args.mode == "monitor":
        monitor_training(args.work_dir, check_interval=args.interval)
    else:
        report = generate_validation_report(args.work_dir)
        print("\nValidation Report:")
        print(json.dumps(report, indent=2))
        
        if report['training_valid']:
            print("\n✓ Training appears to be working correctly")
            sys.exit(0)
        else:
            print("\n✗ Training issues detected")
            sys.exit(1)
