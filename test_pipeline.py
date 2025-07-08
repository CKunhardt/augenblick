import sys
import os

# Add NeuS2 directory to path
neus2_path = '/home/jhennessy7.gatech/augenblick/src/NeuS2'
sys.path.insert(0, neus2_path)

print("Testing pipeline components...")
try:
    # Save current directory
    original_dir = os.getcwd()
    
    # Change to NeuS2 directory for import
    os.chdir(neus2_path)
    
    import torch
    print(f"✓ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
    
    import pyngp as ngp
    print("✓ NeuS2 imported successfully")
    
    # Return to original directory
    os.chdir(original_dir)
    
    # Check what's in src directory
    print("\nContents of src/:")
    src_dir = '/home/jhennessy7.gatech/augenblick/src'
    for item in os.listdir(src_dir):
        print(f"  - {item}")
    
    print("\n✓ All components ready!")
    print("\nNow you can run your reconstruction pipeline!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
