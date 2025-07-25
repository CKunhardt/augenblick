import subprocess
import os

# Test that torchrun can see the GPU
env = os.environ.copy()
cmd = ["torchrun", "--nproc_per_node=1", "--help"]
result = subprocess.run(cmd, env=env, capture_output=True, text=True)
print("Torchrun help executed successfully" if result.returncode == 0 else f"Failed: {result.stderr}")

# Test with a simple GPU check
test_script = """
import torch
print(f"GPU available in torchrun: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
"""

with open("test_torch_gpu.py", "w") as f:
    f.write(test_script)

cmd = ["torchrun", "--nproc_per_node=1", "test_torch_gpu.py"]
result = subprocess.run(cmd, env=env, capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print(f"Stderr: {result.stderr}")

# Clean up
os.remove("test_torch_gpu.py")
