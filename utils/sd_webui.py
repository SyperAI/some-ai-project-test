import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from subprocess import Popen
from threading import Thread
from typing import List

import requests.exceptions
from webuiapi import WebUIApi, HiResUpscaler, ControlNetUnit, AnimateDiff, Roop, ReActor, Sag

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

    def txt2img(
        self,
        enable_hr=False,
        denoising_strength=0.7,
        firstphase_width=0,
        firstphase_height=0,
        hr_scale=2,
        hr_upscaler=HiResUpscaler.Latent,
        hr_second_pass_steps=0,
        hr_resize_x=0,
        hr_resize_y=0,
        hr_checkpoint_name=None,
        hr_sampler_name=None,
        hr_scheduler=None,
        hr_prompt="",
        hr_negative_prompt="",
        prompt="",
        styles=[],
        seed=-1,
        subseed=-1,
        subseed_strength=0.0,
        seed_resize_from_h=0,
        seed_resize_from_w=0,
        sampler_name=None,  # use this instead of sampler_index
        scheduler=None,
        batch_size=1,
        n_iter=1,
        steps=None,
        cfg_scale=7.0,
        width=512,
        height=512,
        restore_faces=False,
        tiling=False,
        do_not_save_samples=False,
        do_not_save_grid=False,
        negative_prompt="",
        eta=1.0,
        s_churn=0,
        s_tmax=0,
        s_tmin=0,
        s_noise=1,
        override_settings={},
        override_settings_restore_afterwards=True,
        script_args=None,  # List of arguments for the script "script_name"
        script_name=None,
        send_images=True,
        save_images=False,
        alwayson_scripts={},
        controlnet_units: List[ControlNetUnit] = [],
        animatediff: AnimateDiff = None,
        roop: Roop = None,
        reactor: ReActor = None,
        sag: Sag = None,
        sampler_index=None,  # deprecated: use sampler_name
        use_deprecated_controlnet=False,
        use_async=False,
    ):
        if sampler_index is None:
            sampler_index = self.default_sampler
        if sampler_name is None:
            sampler_name = self.default_sampler

        if scheduler is None:
            scheduler = self.default_scheduler

        if steps is None:
            steps = self.default_steps
        if script_args is None:
            script_args = []
        payload = {
            "enable_hr": enable_hr,
            "hr_scale": hr_scale,
            "hr_upscaler": hr_upscaler,
            "hr_second_pass_steps": hr_second_pass_steps,
            "hr_resize_x": hr_resize_x,
            "hr_resize_y": hr_resize_y,
            "hr_checkpoint_name": hr_checkpoint_name,
            "hr_sampler_name": hr_sampler_name,
            "hr_scheduler": hr_scheduler,
            "hr_prompt": hr_prompt,
            "hr_negative_prompt": hr_negative_prompt,
            "denoising_strength": denoising_strength,
            "firstphase_width": firstphase_width,
            "firstphase_height": firstphase_height,
            "prompt": prompt,
            "styles": styles,
            "seed": seed,
            "subseed": subseed,
            "subseed_strength": subseed_strength,
            "seed_resize_from_h": seed_resize_from_h,
            "seed_resize_from_w": seed_resize_from_w,
            "batch_size": batch_size,
            "n_iter": n_iter,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "restore_faces": restore_faces,
            "tiling": tiling,
            "do_not_save_samples": do_not_save_samples,
            "do_not_save_grid": do_not_save_grid,
            "negative_prompt": negative_prompt,
            "eta": eta,
            "s_churn": s_churn,
            "s_tmax": s_tmax,
            "s_tmin": s_tmin,
            "s_noise": s_noise,
            "override_settings": override_settings,
            "override_settings_restore_afterwards": override_settings_restore_afterwards,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "sampler_index": sampler_index,
            "script_name": script_name,
            "script_args": script_args,
            "send_images": send_images,
            "save_images": save_images,
            "alwayson_scripts": alwayson_scripts,
        }

        if use_deprecated_controlnet and controlnet_units and len(controlnet_units) > 0:
            payload["controlnet_units"] = [x.to_dict() for x in controlnet_units]
            return self.custom_post(
                "controlnet/txt2img", payload=payload, use_async=use_async
            )

        if self.has_adetailer and "ADetailer" not in alwayson_scripts.keys():
            payload["alwayson_scripts"]["ADetailer"] = {
                "args": [False]
            }

        if animatediff:
            payload["alwayson_scripts"]["animatediff"] = {
                "args": [animatediff.to_dict(False)]
            }
        elif self.has_animatediff:
            payload["alwayson_scripts"]["animatediff"] = {
                "args": [False],
            }

        if roop :
            payload["alwayson_scripts"]["roop"] = {
                "args": roop.to_dict()
            }

        if reactor :
            payload["alwayson_scripts"]["reactor"] = {
                "args": reactor.to_dict()
            }

        if sag :
            payload["alwayson_scripts"]["Self Attention Guidance"] = {
                "args": sag.to_dict()
            }


        if controlnet_units and len(controlnet_units) > 0:
            payload["alwayson_scripts"]["ControlNet"] = {
                "args": [x.to_dict() for x in controlnet_units]
            }
        elif self.has_controlnet:
            # workaround : if not passed, webui will use previous args!
            payload["alwayson_scripts"]["ControlNet"] = {"args": []}

        return self.post_and_get_api_result(
            f"{self.baseurl}/txt2img", payload, use_async
        )


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
            webui_api.refresh_checkpoints()

            models = webui_api.get_sd_models()
            logger.info(f"Connected to SD and found {len(models)} checkpoints")

            def cache_models():
                for model in models:
                    model_sha256 = get_file_sha256(model['filename'])
                    logger.info(f"Found SD model {model['title']}[{model_sha256}]")

            # Cache models hash
            Thread(target=cache_models).start()

            # Applying default model if exists in config
            if config.SD_CONFIG.PARAMS.default_model != "": webui_api.set_options(
                {"sd_model_checkpoint": config.SD_CONFIG.PARAMS.default_model})
            break
        except requests.exceptions.ConnectionError:
            logger.warning("SD Web UI connection failed, retrying in 1s...")
            time.sleep(1)

    return webui_api, sd_webui_process
