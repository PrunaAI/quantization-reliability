import os
from pynvml import *
import subprocess
import re

import logging
logger = logging.getLogger("quant_logger")

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

def evaluate_disk_space_usage(model):
    try:
        # Check if the model has a PATH attribute
        if not hasattr(model, 'PATH'):
            raise AttributeError("The model object does not have a 'PATH' attribute.")
        
        model_path = model.PATH
        # Check if the provided path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The path '{model_path}' does not exist.")
        
        total_size = 0
        
        if os.path.isdir(model_path):
            for filename in os.listdir(model_path):
                file_path = os.path.join(model_path, filename)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
        elif os.path.isfile(model_path):
            total_size = os.path.getsize(model_path)
        else:
            raise ValueError(f"The path '{model_path}' is neither a file nor a directory.")
        
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

        return f"{total_size:.2f} {unit}"
    
    except Exception as e:
        logger.error(e)
        return "nan"

def evaluate_quantize_runtime(model):
    if not hasattr(model, 'QUANT_TIME'):
        return "nan"
    
    return model.QUANT_TIME

def get_gpu_memory():
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'])
    memory_values = result.decode('utf-8').strip().split('\n')
    memory_values_gb = [float(mem) / 1024 for mem in memory_values]  # Convert to GB
    return memory_values_gb

def record_gpu_memory(gpu_memory_usage, context):
    memory_values = get_gpu_memory()
    gpu_memory_usage[context] = memory_values
    logger.info(f"Free GPU Memory (GB) per GPU: {memory_values}. Context: {context}.")