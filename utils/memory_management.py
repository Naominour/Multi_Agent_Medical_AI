import torch
import gc

def clear_gpu_memory():
    # Clears GPU memory to prevent OOM errors.
    gc.collect()
    torch.cuda.empty_cache()
