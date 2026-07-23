import logging
import os.path
from pathlib import Path

import requests.exceptions

from app.worker_app import WorkerApp
from utils import config, get_file_sha256, download
from utils.llm import get_ollama
from utils.sd_webui import get_webui
from utils.types import TaskType, SDModelsResponse, T2IRequest, T2IResponse, SDLora, LLMRequest

app = WorkerApp(base_url=config.MOTHER_NODE.url, api_key=config.MOTHER_NODE.key)

webui_api, sd_webui_process = get_webui()
llm_client = get_ollama()


def split_sampler_and_scheduler(combined_sampler_name: str) -> tuple[str, str]:
    schedulers = ["Karras", "Exponential", "SGM Uniform"]

    for sched in schedulers:
        if combined_sampler_name.endswith(f" {sched}"):
            clean_sampler = combined_sampler_name.replace(f" {sched}", "")
            return clean_sampler, sched

    return combined_sampler_name, "Automatic"


@app.task_handler(task_type=TaskType.MODELS_LIST)
def models_list():
    try:
        models = SDModelsResponse.model_validate(webui_api.get_sd_models())
    except requests.exceptions.ConnectionError:
        logging.error("Can't connect to webui!")
        return None

    return models.model_dump(include={'__all__': {'model_name', 'sha256'}})


def check_model(model_name: str) -> bool:
    models = webui_api.get_sd_models()

    for model in models:
        if model_name in model['sha256']:
            return True

    if download(
            file_url=os.path.join("https://d7h1.c16.e2-4.dev/test/", model_name),
            save_path=Path(config.SD_CONFIG.CONFIG.path, "models", "Stable-diffusion")
    ) is not None: return check_model(model_name)

    return False


def check_lora(loras_info: list[SDLora]) -> bool:
    webui_api.refresh_loras()
    loras = webui_api.get_loras()

    for lora_info in loras_info:
        for lora in loras:
            if lora_info.hash == get_file_sha256(lora['path']): break
        else:
            logging.warning(f"Lora {os.path.basename(lora_info.path)} not found will try to download.")
            if download(file_url=lora_info.path,
                        save_path=Path(config.SD_CONFIG.CONFIG.path, "models", "Lora")) is not None: return check_lora(
                loras_info)

    return True


@app.task_handler(task_type=TaskType.TXT2IMG)
def txt2img(task: T2IRequest):
    alwayson_scripts = {}

    # Check and use model
    if task.model != "":
        if not check_model(task.model):
            return None
        webui_api.set_options({"sd_model_checkpoint": task.model})

    # Check LoRa and download if not exists
    if len(task.lora_info) > 0:
        if not check_lora(task.lora_info): return None

    # Apply adetailer if given
    if task.adetailer is not None:
        alwayson_scripts["ADetailer"] = {"args": [task.adetailer.to_args(), False, False, False]}

    try:
        actual_sampler, actual_scheduler = split_sampler_and_scheduler(str(task.sampler_name))
        result = webui_api.txt2img(
            **task.model_dump(exclude_none=True, exclude={"model", "sampler_name", "lora_info", "adetailer"}),
            sampler_name=actual_sampler, scheduler=actual_scheduler, alwayson_scripts=alwayson_scripts)
        if result is None or result.image is None:
            return None
    except Exception as e:
        logging.error(e)
        return None

    response = T2IResponse(
        files=[(image, result.info['prompt'], result.info['negative_prompt']) for image in result.images],
        prompt=result.info['prompt'],
        negative_prompt=result.info['negative_prompt']
    )

    return response


@app.task_handler(task_type=TaskType.LLM)
def llm_task(task: LLMRequest):
    answer = llm_client.generate(
        model=task.model,
        system=task.system_prompt,
        prompt=task.prompt,
        options=task.options,
    )

    return answer['response'].strip()


if __name__ == "__main__":
    try:
        app.run()
    except KeyboardInterrupt:
        if sd_webui_process is not None:
            sd_webui_process.kill()
