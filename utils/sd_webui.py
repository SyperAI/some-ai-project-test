import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from subprocess import Popen

import requests.exceptions
from webuiapi import WebUIApi

from utils import config, logger, get_file_sha256
from utils.gpu import get_vram, get_compute_cap, is_nvidia_available, GPUInfoStorage

if sys.platform == "win32":
    python_exe = os.path.join(config.SD_CONFIG.CONFIG.path, "venv", "Scripts", "python.exe")
else:
    # Linux / macOS
    python_exe = os.path.join(config.SD_CONFIG.CONFIG.path, "venv", "bin", "python")

if not is_nvidia_available() and config.SD_CONFIG.CONFIG.enable:
    logger.warning("nvidia-smi was not found, trying to obtain gpu info via torch...")

    subprocess.run([python_exe, "gpu_info_torch.py"], check=True)

    info_file = Path(".gpuinfo")
    if not info_file.exists():
        logger.error("Can't load GPU info.")
        sys.exit(1)

    GPUInfoStorage.gpu_info = json.load(info_file.open('r'))
    logger.info(f"GPU info loaded via torch: {GPUInfoStorage.gpu_info}")


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
# Dont download default junk sd1.5 model on start
if "--no-download-sd-model" not in START_FLAGS: START_FLAGS.append("--no-download-sd-model")

command = [python_exe, LAUNCH_SCRIPT] + START_FLAGS + opti_flags

os.environ["STABLE_DIFFUSION_REPO"] = "https://github.com/w-e-w/stablediffusion.git"


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


def get_webui() -> tuple[None, None] | tuple[SDWebUI, Popen]:
    if not config.SD_CONFIG.CONFIG.enable:
        logging.warning("SD WebUI is disabled which will result in not recieving SD tasks!")
        return None, None
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

    for x in range(30):
        try:
            models = webui_api.get_sd_models()
            logger.info(f"Connected to SD and found {len(models)} checkpoints")

            # Cache models hash
            for model in models:
                model_sha256 = get_file_sha256(model['filename'])
                logger.info(f"Found SD model {model['title']}[{model_sha256}]")

            # Applying default model if exists in config
            if config.SD_CONFIG.PARAMS.default_model != "": webui_api.set_options(
                {"sd_model_checkpoint": config.SD_CONFIG.PARAMS.default_model})
            break
        except requests.exceptions.ConnectionError:
            logger.warning("SD Web UI connection failed, retrying in 1s...")
            time.sleep(1)

    return webui_api, sd_webui_process
