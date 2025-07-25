#!/usr/bin/env python3
"""
Check VGG-T installation and find the correct entry point
"""
import os
import sys
from pathlib import Path
import json

def check_vggt_installation(vggt_path):
    """Check VGG-T installation and find available scripts"""
    vggt_dir = Path(vggt_path)
    
    if not vggt_dir.exists():
        print(f"❌ VGG-T directory not found: {vggt_path}")
        return False
    
    print(f"✓ VGG-T directory found: {vggt_path}")
    print("\nSearching for VGG-T components...")
    
    # Look for common entry points
    possible_scripts = [
        "run_vggt.py",
        "main.py",
        "reconstruct.py",
        "run_reconstruction.py",
        "vggt_reconstruction.py",
        "scripts/run_vggt.py",
        "scripts/reconstruct.py",
        "demo.py",
        "inference.py"
    ]
    
    found_scripts = []
    for script in possible_scripts:
        script_path = vggt_dir / script
        if script_path.exists():
            found_scripts.append(script_path)
            print(f"  ✓ Found: {script}")
    
    # Look for any Python files that might be entry points
    print("\nAll Python files in root directory:")
    for py_file in vggt_dir.glob("*.py"):
        print(f"  - {py_file.name}")
        # Check if it has main
        with open(py_file, 'r') as f:
            content = f.read()
            if "if __name__ == '__main__':" in content or "def main(" in content:
                print(f"    → Has main function!")
    
    # Check for scripts directory
    scripts_dir = vggt_dir / "scripts"
    if scripts_dir.exists():
        print("\nPython files in scripts/ directory:")
        for py_file in scripts_dir.glob("*.py"):
            print(f"  - scripts/{py_file.name}")
    
    # Check for config examples
    print("\nConfiguration files:")
    for config_pattern in ["*.json", "*.yaml", "*.yml", "config/*", "configs/*"]:
        for config_file in vggt_dir.glob(config_pattern):
            print(f"  - {config_file.relative_to(vggt_dir)}")
    
    # Check for README
    print("\nDocumentation:")
    for readme_pattern in ["README*", "readme*", "docs/*"]:
        for readme in vggt_dir.glob(readme_pattern):
            print(f"  - {readme.relative_to(vggt_dir)}")
    
    # Check for requirements
    print("\nDependencies:")
    for req_file in ["requirements.txt", "setup.py", "environment.yml"]:
        if (vggt_dir / req_file).exists():
            print(f"  ✓ Found: {req_file}")
    
    # Try to find model files or checkpoints
    print("\nModel/checkpoint files:")
    model_patterns = ["*.pth", "*.ckpt", "*.pt", "checkpoints/*", "models/*", "pretrained/*"]
    model_found = False
    for pattern in model_patterns:
        for model_file in vggt_dir.glob(pattern):
            print(f"  - {model_file.relative_to(vggt_dir)}")
            model_found = True
    if not model_found:
        print("  ⚠ No model files found - may need to download separately")
    
    # Suggest next steps
    print("\n" + "="*50)
    print("NEXT STEPS:")
    print("="*50)
    
    if found_scripts:
        print("\n1. Check the main script for usage:")
        main_script = found_scripts[0]
        print(f"   python {main_script.relative_to(vggt_dir)} --help")
    
    print("\n2. Look for example usage in README:")
    print(f"   cat {vggt_dir}/README* | grep -A 10 -i 'usage\\|example\\|run'")
    
    print("\n3. Check if there's a demo or test dataset:")
    print(f"   find {vggt_dir} -name '*demo*' -o -name '*example*' -o -name '*test*'")
    
    print("\n4. Set the environment variable for the pipeline:")
    print(f"   export VGGT_PATH={vggt_dir}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        vggt_path = sys.argv[1]
    else:
        # Default path from your message
        vggt_path = "/home/jhennessy7.gatech/augenblick/src/vggt"
    
    check_vggt_installation(vggt_path)
