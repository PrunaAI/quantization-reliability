import os
from pynvml import *

def evaluate_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)

    # Calculate memory usage in MB
    memory_used = info.used // 1024**2

    # Determine unit based on usage
    unit = "MB"
    if memory_used > 1024:
      memory_used = memory_used / 1024  # Convert to GB
      unit = "GB"

    # print(f"GPU memory occupied: {memory_used:.2f} {unit}.")
    return memory_used, unit
      

def evaluate_model_size(model_path):
    """
    This function calculates the total size of a model directory containing saved model files.

    Args:
        model_path (str): The path to the directory containing the model files.

    Returns:
        None: The function directly prints the total model size in human-readable format.
    """

    # Initialize total size variable
    total_size = 0

    # Loop through files in the model directory
    for filename in os.listdir(model_path):
        file_path = os.path.join(model_path, filename)
        # Check if it's a file (not a directory)
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path)
            total_size += file_size

    # Convert to human-readable format
    if total_size > 1024**3:
        total_size_gb = total_size / (1024**3)
        return total_size_gb, "GB"
        # print(f"Total Model Size for {model_path}: {total_size_gb:.2f} GB")
    elif total_size > 1024**2:
        total_size_mb = total_size / (1024**2)
        return total_size_mb, "MB"
        # print(f"Total Model Size for {model_path}: {total_size_mb:.2f} MB")
    elif total_size > 1024:
        total_size_kb = total_size / 1024
        return total_size_kb, "KB"
        # print(f"Total Model Size for {model_path}: {total_size_kb:.2f} KB")
    
    return total_size, "B"
    # print(f"Total Model Size for {model_path}: {total_size} bytes")  # Keep as bytes for small sizes
