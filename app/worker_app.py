import logging
import queue
import sys
import time
import traceback
from threading import Thread
from typing import Dict, Callable, Any

import httpx
from pydantic import BaseModel

from utils import config, logger
from utils.gpu import get_gpu_name
from utils.sd_webui import vram_gb
from utils.timer import Timer
from utils.types import TaskType, T2IRequest, T2IResponse, AIType, LLMRequest


def get_supported() -> list[str]:
    supports = []

    if config.SD_CONFIG.CONFIG.enable:
        if 4.0 <= vram_gb: supports.append(AIType.SD15.value)
        if 7.5 <= vram_gb: supports.append(AIType.SDXL.value)

    if config.OLLAMA.enabled:
        supports.append(AIType.LLM.value)

    return supports


def is_supported(task_type: str) -> bool:
    supported = get_supported()

    if task_type == TaskType.TXT2IMG.value and AIType.SDXL.value in supported or AIType.SD15.value in supported:
        return True
    elif task_type == TaskType.LLM.value and AIType.LLM.value in supported:
        return True

    return False


class WorkerApp:
    def __init__(self, base_url: str, api_key: str) -> None:
        logging.getLogger("httpx").setLevel(logging.CRITICAL)

        self.base_url = base_url
        self.api_key = api_key

        self.task_queue = queue.Queue()
        self.task_processing_ids = set()
        self.polling_thread = Thread(target=self._polling)

        self._headers = {
            "X-Api-Key": self.api_key,
        }

        self._handlers: Dict[TaskType, Callable] = {}

    # TODO: Change to different handlers?
    def task_handler(self, task_type: TaskType):
        def decorator(func: Callable):
            if task_type in self._handlers:
                raise RuntimeError(f"Task {task_type} already registered")

            self._handlers[task_type] = func
            return func

        return decorator

    def _dispatch_and_execute(self, task_data: dict) -> Any:
        task_type = task_data.get("type")

        if not is_supported(task_type):
            logging.warning("Recieved unsupported task of type %s!", task_type)
            return None

        logger.info(f"Processing task id={task_data['id']} of type {task_type}...")

        if not task_type:
            logging.warning(f"Current task has no task type!")
            return
        try:
            task_type = TaskType(task_type)
        except ValueError:
            logging.warning(f"Invalid task type: {task_type}")
            return

        handler_func = self._handlers.get(task_type)
        if not handler_func:
            logging.warning(f"Task {task_type.value} has no handler function!")
            return

        if task_type == TaskType.TXT2IMG:
            task_class = T2IRequest
        elif task_type == TaskType.LLM:
            task_class = LLMRequest
        else:
            task_class = None

        if task_class is None:
            return handler_func()

        return handler_func(task=task_class(**task_data))

    def send_multipart_result(self, client, task_data, status, result) -> None:
        headers = {
            "X-Api-Key": self.api_key,
        }

        form_data, form_files = result.to_form()
        form_data['task_id'] = task_data['id']
        form_data['status'] = status

        r = client.post(f"{self.base_url}/node/form-result", headers=headers, data=form_data, files=form_files)

        if not r.is_success:
            logger.error(r.text)
            sys.exit(1)

        if not r.json()['status']: sys.exit(1)

    def _polling(self):
        with httpx.Client(timeout=30) as client:
            while True:
                try:
                    fetch_url = f"{self.base_url}/node/fetch"

                    response = client.get(fetch_url, headers=self._headers)
                    if response.status_code != httpx.codes.OK:
                        logging.error(
                            f"Request failed with status code {response.status_code}, sleeping for 1s before retrying...")
                        time.sleep(1)
                        continue

                    task_data = response.json()
                    if task_data is None:
                        continue

                    if len(task_data) < 1:
                        time.sleep(1)
                        continue

                    for task in task_data:
                        if task['id'] not in self.task_processing_ids:
                            logger.info(f"Task id={task['id']} queued")
                            self.task_processing_ids.add(task['id'])
                            self.task_queue.put(task)
                    else:
                        time.sleep(1)

                except httpx.RequestError as e:
                    logging.error(f"Network error: {e}, retry in 5s...")
                    time.sleep(5)
                except Exception as e:
                    logging.critical(f"Polling error: {e}")
                    traceback.print_exc()
                    time.sleep(5)

    def run(self):
        logging.info("Starting node")

        with httpx.Client(timeout=30) as client:
            response = client.put(
                f"{self.base_url}/node/info",
                headers={
                    "X-Api-Key": self.api_key,
                },
                json={
                    'supports': get_supported(),
                    'gpu_model': get_gpu_name(),
                }
            )

            if response.status_code != 200:
                logger.critical(response.text)
                raise RuntimeError("Node login failed!")

            self.polling_thread.start()

            while True:
                try:
                    task_data = self.task_queue.get()

                    with Timer(name=f"Task id={task_data['id']}", print_func=logging.info):
                        try:
                            client.put(f"{self.base_url}/node/task-info", headers=self._headers, json={
                                "task_id": task_data['id'],
                                "status": "processing",
                            })

                            result = self._dispatch_and_execute(task_data)
                            status = True if result else False
                        except Exception as e:
                            logging.error(e)
                            traceback.print_exc()
                            result = None
                            status = False

                    try:
                        if type(result) in (T2IResponse,):
                            self.send_multipart_result(client=client, task_data=task_data, status=status, result=result)
                            continue

                        client.post(f"{self.base_url}/node/result", headers=self._headers, json={
                            "status": status,
                            "task_id": task_data["id"],
                            "result": result.model_dump(mode='json') if isinstance(result, BaseModel) else result,
                        })
                    finally:
                        self.task_processing_ids.remove(task_data["id"])

                except httpx.RequestError as e:
                    logging.error(f"Network error: {e}, retry in 5s...")
                    time.sleep(5)
