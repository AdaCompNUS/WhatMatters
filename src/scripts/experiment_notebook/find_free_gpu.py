import torch
import os
import subprocess
import time

def select_free_gpu():
    """
    Selects the GPU with the most free memory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if str(device) == "cuda":
        # Get the memory usage of each GPU
        memory_usage = []
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                free_memory = torch.cuda.memory_stats(i)["free"]
                memory_usage.append((free_memory, i))

        # Select the GPU with the most free memory
        print(memory_usage)
        best_memory, best_device = max(memory_usage)
        return best_device
    else:
        return None
    
def check():
    # Get the list of GPUs via nvidia-smi
    smi_query_result = subprocess.check_output(
        "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
    )
    # Extract the usage information
    gpu_info = smi_query_result.decode("utf-8").split("\n")
    used_gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
    used_gpu_info = [
        int(x.split(":")[1].replace("MiB", "").strip()) for x in used_gpu_info
    ]  # Remove garbage
    # Keep gpus under threshold only
    total_gpu_info = list(filter(lambda info: "Total" in info, gpu_info))
    total_gpu_info =  [
        int(x.split(":")[1].replace("MiB", "").strip()) for x in total_gpu_info
    ]

    free_gpus = [
        (total-used, i) for i, (total, used) in enumerate(zip(total_gpu_info, used_gpu_info))
    ]
    print("Total GPU info: ", free_gpus)

    best_memory, best_device = max(free_gpus)
    
    return best_device

# Select the most free GPU
# best_device = select_free_gpu()
# if best_device is not None:
#     # Set torch to use this GPU
#     torch.cuda.set_device(best_device)

# # Example usage in a function
# def my_function():
#     x = torch.randn(10, 10)
#     if torch.cuda.is_available():
#         x = x.cuda()

print(check())