import hashlib
import os
from functools import lru_cache
from pathlib import Path

import httpx

from utils import logger


@lru_cache
def get_file_md5(file_path: str) -> str:
    md5_hash = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(32768), b""):
            md5_hash.update(chunk)

    return md5_hash.hexdigest()


def download(file_url: str, save_path: str | Path) -> Path | None:
    if isinstance(save_path, str): save_path = Path(save_path)

    save_path.mkdir(parents=True, exist_ok=True)
    save_path = save_path / os.path.basename(file_url)

    logger.debug(f"Downloading {file_url} to {save_path}")
    try:
        with httpx.Client() as client:
            with client.stream("GET", file_url) as response:
                response.raise_for_status()

                with open(save_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)
    except Exception as e:
        logger.error(f"Failed to download {file_url}: {e}")
        return None

    return save_path
