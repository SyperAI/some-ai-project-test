import logging
import subprocess
import typing


class GPUInfoStorage:
    gpu_info: typing.Optional[dict] = None


def is_nvidia_available():
    try:
        subprocess.check_output(['nvidia-smi'])
    except Exception as e:
        logging.error(e)
        return False
    else:
        return True


def get_vram() -> int:
    try:
        result = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=memory.total',
                '--format=csv,noheader,nounits'
            ],
            encoding='utf-8'
        )
        total_vram_mb = int(result.strip())
        return total_vram_mb
    except FileNotFoundError:
        if GPUInfoStorage.gpu_info is not None: return GPUInfoStorage.gpu_info['vram']

        logging.error('nvidia-smi not found, memory checks may be ignored!')
    except Exception as e:
        logging.error(e)

    return 0


def get_compute_cap() -> float:
    try:
        result = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=compute_cap',
                '--format=csv,noheader,nounits'
            ],
            encoding='utf-8'
        )
        compute_cap = float(result.strip())
        return compute_cap
    except FileNotFoundError:
        if GPUInfoStorage.gpu_info is not None: return GPUInfoStorage.gpu_info['compute_cap']

        logging.error('nvidia-smi not found, optimizations checks may be ignored!')
    except Exception as e:
        logging.error(e)

    return -1


def get_gpu_name() -> str | None:
    try:
        result = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=gpu_name',
                '--format=csv,noheader,nounits'
            ],
            encoding='utf-8'
        )
        gpu_name = result.strip()
        return gpu_name
    except FileNotFoundError:
        if GPUInfoStorage.gpu_info is not None: return GPUInfoStorage.gpu_info['gpu_name']

        logging.error('nvidia-smi not found!')
    except Exception as e:
        logging.error(e)

    return None