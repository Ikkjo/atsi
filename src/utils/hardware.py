"""Hardware and environment configuration."""

import torch


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_hardware_config() -> dict:
    """Return hardware-specific configuration for batch sizes and workers."""
    device = get_device()

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

        if total_memory_gb > 20:
            whisper_batch_size = 4
            embedding_batch_size = 32
            num_workers = 4
        elif total_memory_gb > 10:
            whisper_batch_size = 2
            embedding_batch_size = 16
            num_workers = 2
        else:
            whisper_batch_size = 1
            embedding_batch_size = 8
            num_workers = 2
    else:
        gpu_name = "N/A"
        total_memory_gb = 0
        whisper_batch_size = 1
        embedding_batch_size = 4
        num_workers = 0

    return {
        "device": device.type,
        "device_name": gpu_name,
        "total_memory_gb": round(total_memory_gb, 1),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "whisper_batch_size": whisper_batch_size,
        "embedding_batch_size": embedding_batch_size,
        "num_workers": num_workers,
    }


def print_hardware_info() -> None:
    """Print hardware configuration summary."""
    config = get_hardware_config()
    print("=" * 60)
    print("Hardware Configuration")
    print("=" * 60)
    print(f"  Device:              {config['device']}")
    print(f"  Device Name:         {config['device_name']}")
    print(f"  Total Memory (GB):   {config['total_memory_gb']}")
    print(f"  CUDA Available:      {config['cuda_available']}")
    print(f"  CUDA Version:        {config['cuda_version']}")
    print(f"  Whisper Batch Size:  {config['whisper_batch_size']}")
    print(f"  Embedding Batch Sz:  {config['embedding_batch_size']}")
    print(f"  Num Workers:         {config['num_workers']}")
    print("=" * 60)
