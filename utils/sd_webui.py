import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from subprocess import Popen

import httpx
import requests.exceptions
from webuiapi import WebUIApi

from utils import config, logger
from utils.gpu import get_vram, get_compute_cap

if sys.platform == "win32":
    python_exe = os.path.join(config.SD_CONFIG.CONFIG.path, "venv", "Scripts", "python.exe")
else:
    # Linux / macOS
    python_exe = os.path.join(config.SD_CONFIG.CONFIG.path, "venv", "bin", "python")

vram_gb = get_vram() / 1024
opti_flags = []
if vram_gb <= 0:
    pass
elif vram_gb <= 4:
    opti_flags.append("--lowvram")
    logger.info("Used lowvram optimization")
elif vram_gb <= 6:
    opti_flags.append("--medvram")
    logger.info("Used medvram optimization")
elif vram_gb <= 12:
    opti_flags.append("--medvram-sdxl")
    logger.info("Used medvram optimization for SDXL")
else:
    logger.info("Used no vram optimization")

compute_cap = get_compute_cap()
if compute_cap >= 7.0:
    opti_flags.append("--xformers")
    logger.info("Used xformers optimization")
else:
    opti_flags.append("--opt-sdp-attention")
    logger.warning("Your currnet GPU does not support xformers, opt-sdp-attention will be used instead")

LAUNCH_SCRIPT = "launch.py"
START_FLAGS = config.SD_CONFIG.CONFIG.start_flags.split(" ")

command = [python_exe, LAUNCH_SCRIPT] + START_FLAGS + opti_flags

def start_a1111():
    if sys.platform == "win32":
        process = subprocess.Popen(command, cwd=config.SD_CONFIG.CONFIG.path,
                                   creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        process = subprocess.Popen(command, cwd=config.SD_CONFIG.CONFIG.path)

    return process


class SDWebUI(WebUIApi):
    def refresh_loras(self):
        response = self.session.post(url=f"{self.baseurl}/refresh-loras")
        return response.json()


def get_webui() -> tuple[SDWebUI, Popen | None] | None:
    if not config.SD_CONFIG.CONFIG.enable:
        logging.warning("SD WebUI is disabled which will result in not recieving SD tasks!")
        return None
    sd_webui_process = None
    # Auto start
    if config.SD_CONFIG.CONFIG.auto_start and config.SD_CONFIG.CONFIG.path == "":
        raise ValueError("Can't start SD Web UI without SD path specified! Check config and try again.")
    elif config.SD_CONFIG.CONFIG.auto_start:
        logger.info("Starting SD Web UI.")
        sd_webui_process = start_a1111()

    webui_api = SDWebUI(
        host=config.SD_CONFIG.PARAMS.host,
        port=config.SD_CONFIG.PARAMS.port,
        username=config.SD_CONFIG.PARAMS.username,
        password=config.SD_CONFIG.PARAMS.password,
    )
    if config.SD_CONFIG.PARAMS.default_model != "":
        for x in range(30):
            try:
                webui_api.set_options({"sd_model_checkpoint": config.SD_CONFIG.PARAMS.default_model})
                break
            except requests.exceptions.ConnectionError:
                logger.warning("SD Web UI connection failed, retrying in 1s...")
                time.sleep(1)

    return webui_api, sd_webui_process
