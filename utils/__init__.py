import logging

from utils.config import Config

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s][%(levelname)s][%(module)s][%(lineno)d] - %(message)s")
logger = logging.getLogger()

config = Config(allow_missing=True).load()

from .sd_webui import vram_gb
from .utils import *