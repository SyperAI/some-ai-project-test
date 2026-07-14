import logging
import sys

from ollama import Client

from utils import config


def get_ollama() -> Client | None:
    if not config.OLLAMA.enabled:
        logging.warning("Ollama is disabled which will result in not recieving LLM tasks!")
        return None

    client = Client(host=config.OLLAMA.url)

    try:
        print(client.list())
    except Exception as e:
        logging.critical(f"Can't connect to Ollama. Check config and try again. Error: {e}")
        sys.exit(1)

    return client

