import hashlib
import json
import os
from functools import lru_cache
from json import JSONDecodeError
from pathlib import Path

import httpx

from utils import logger

models_cache_file = Path(".sd-models-cache")
models_cache_file.touch(exist_ok=True)

try:
    models_cache = json.load(models_cache_file.open("r"))
except (JSONDecodeError, FileNotFoundError):
    models_cache = {}


def _file_sha256(file_path):
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        # with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        #     sha256_hash.update(mm)

        for chunk in iter(lambda: f.read(1048576), b""):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


@lru_cache(maxsize=1024)
def _cached_sha256(file_path: str, mtime: float, size: int):
    if file_path in models_cache.keys():
        if mtime == models_cache[file_path]["mtime"] and size == models_cache[file_path]["size"]:
            return models_cache[file_path]["sha256"]

    file_sha256 = _file_sha256(file_path)
    models_cache[file_path] = {
        "mtime": mtime,
        "size": size,
        "sha256": file_sha256
    }
    json.dump(models_cache, models_cache_file.open("w"))

    return file_sha256


def get_file_sha256(file_path: str) -> str:
    if not os.path.exists(file_path): return None

    stat = os.stat(file_path)
    return _cached_sha256(file_path, stat.st_mtime, stat.st_size)


def download(file_url: str, save_path: str | Path) -> Path | None:
    if isinstance(save_path, str): save_path = Path(save_path)

    save_path.mkdir(parents=True, exist_ok=True)
    save_path = save_path / os.path.basename(file_url)

    logger.info(f"Downloading {file_url} to {save_path}")
    try:
        with httpx.Client(http2=True) as client:
            with client.stream("GET", file_url) as response:
                response.raise_for_status()

                with open(save_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024 * 5):
                        f.write(chunk)
    except Exception as e:
        logger.error(f"Failed to download {file_url}: {e}")
        return None

    return save_path
