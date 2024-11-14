#!/bin/bash

# Set the GPU memory threshold in GB
GPU_MEMORY_THRESHOLD=20480 # infinite threshold

# Check if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    # Get the total memory of all GPUs in MiB
    gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{sum += $1} END {print sum}')
    
    # Convert MiB to GiB
    gpu_memory_gb=$(echo "scale=1; $gpu_memory / 1024" | bc)
    
    # Check if total GPU memory is smaller than the threshold
    if (( $(echo "$gpu_memory_gb < $GPU_MEMORY_THRESHOLD" | bc -l) )); then
        echo "Total GPU memory ($gpu_memory_gb GB) is less than ${GPU_MEMORY_THRESHOLD}GB. Setting PYTORCH_CUDA_ALLOC_CONF."
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    else
        echo "Total GPU memory ($gpu_memory_gb GB) is ${GPU_MEMORY_THRESHOLD}GB or larger. No need to set PYTORCH_CUDA_ALLOC_CONF."
    fi
else
    echo "nvidia-smi not found. Unable to check GPU memory."
fi
