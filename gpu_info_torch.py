import json
import logging
import torch


def get_vram(device_id: int = 0) -> int:
    if not torch.cuda.is_available():
        logging.error('CUDA not available, memory checks may be ignored!')
        return 0

    try:
        vram_bytes = torch.cuda.get_device_properties(device_id).total_memory
        total_vram_mb = vram_bytes // (1024 * 1024)
        return total_vram_mb
    except Exception as e:
        logging.error(e)
        return 0


def get_compute_cap(device_id: int = 0) -> float:
    if not torch.cuda.is_available():
        logging.error('CUDA not available, optimizations checks may be ignored!')
        return -1.0

    try:
        major, minor = torch.cuda.get_device_capability(device_id)
        compute_cap = float(f"{major}.{minor}")
        return compute_cap
    except Exception as e:
        logging.error(e)
        return -1.0


def get_gpu_name(device_id: int = 0) -> str | None:
    if not torch.cuda.is_available():
        logging.error('CUDA not available!')
        return None

    try:
        gpu_name = torch.cuda.get_device_name(device_id)
        return gpu_name
    except Exception as e:
        logging.error(e)
        return None


if __name__ == '__main__':
    gpu_info = {
        "vram": get_vram(0),
        "compute_cap": get_compute_cap(0),
        "gpu_name": get_gpu_name(0),
    }

    json.dump(gpu_info, open('.gpuinfo', 'w'))
