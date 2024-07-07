import os
from pynvml import *
import subprocess
import re

def evaluate_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)

    # Calculate memory usage in MB
    memory_used = info.used // 1024**2
    unit = "MB"
    if memory_used > 1024:
        memory_used = memory_used / 1024  # Convert to GB
        unit = "GB"

    # print(f"GPU memory occupied: {memory_used:.2f} {unit}.")
    return f"{memory_used} {unit}"

def evaluate_model_size(model_path):
    # Loop through files in the model directory
    if not os.path.exists(model_path):
        return "nan"
    total_size = 0
    for filename in os.listdir(model_path):
        file_path = os.path.join(model_path, filename)
        # Check if it's a file (not a directory)
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)
            total_size += file_size

    # Convert to human-readable format
    if total_size > 1024**3:
        total_size = total_size / (1024**3)
        unit = "GB"
    elif total_size > 1024**2:
        total_size = total_size / (1024**2)
        unit = "MB"
    elif total_size > 1024:
        total_size = total_size / 1024
        unit = "KB"
    else:
        unit = "B"

    return f"{total_size} {unit}"

def evaluate_quantize_runtime(model):
    return model.QUANT_TIME

def get_gpu_memory():
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'])
    return float(result.decode('utf-8').strip()) / 1024  # Convert to GB

def record_gpu_memory(gpu_memory_usage, context):
    memory = get_gpu_memory()
    gpu_memory_usage[context] = memory
    print(f"Free GPU Memory (GB): {memory:.4f}. Context: {context}.")