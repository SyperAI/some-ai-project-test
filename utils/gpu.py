import logging
import subprocess


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
        logging.error('nvidia-smi not found!')
    except Exception as e:
        logging.error(e)

    return None