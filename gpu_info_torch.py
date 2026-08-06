import json
import logging
import sys
from pathlib import Path

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


def device_id_poll() -> int:
    num_devices = torch.cuda.device_count()
    print(f"Found {num_devices} CUDA device{'s' if num_devices > 1 else ''}")
    print("-" * 40)

    for device_id in range(num_devices):
        name = torch.cuda.get_device_name(device_id)
        print(f"{device_id}. {name}")

    device_id = input(f"Please enter device id (0~{num_devices - 1}): ")
    try:
        device_id = int(device_id)
    except:
        return device_id_poll()

    if device_id < 0 or device_id > num_devices - 1:
        return device_id_poll()

    return device_id


if __name__ == '__main__':
    if not torch.cuda.is_available(): sys.exit(0)

    info_file = Path('.gpuinfo')

    if info_file.exists():
        gpu_pre_info = json.load(info_file.open('r'))
        if 'device_id' in gpu_pre_info.keys() and gpu_pre_info['device_id'] is not None:
            did = gpu_pre_info['device_id']
            print(f"Found old data from device {did}. Old data was deleted now, re-start app if something went wrong with this data.")
            info_file.unlink(missing_ok=True)

        else: did = device_id_poll()
    else:
        did = device_id_poll()

    gpu_info = {
        "vram": get_vram(did),
        "compute_cap": get_compute_cap(did),
        "gpu_name": get_gpu_name(did),
        "device_id": did
    }

    json.dump(gpu_info, info_file.open('w'))
