import torch

# Check if CUDA (NVIDIA GPU support) is available
if torch.cuda.is_available():
    print("✅ Success! PyTorch can see your GPU.")
    
    # Get the number of available GPUs
    gpu_count = torch.cuda.device_count()
    print(f"   - GPUs Found: {gpu_count}")
    
    # Get the name of the current GPU (device 0 is the default)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"   - GPU Name: {gpu_name}")
    
else:
    print("❌ Failure! PyTorch cannot see your GPU.")
    print("   - Training will fall back to the much slower CPU.")
    print("   - Make sure you have the NVIDIA CUDA Toolkit and cuDNN installed correctly.")