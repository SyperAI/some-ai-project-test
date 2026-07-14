from auto_configparser import AutoConfig
from pydantic import BaseModel


class MotherNode(BaseModel):
    url: str = None
    key: str = None


class SDConfig(BaseModel):
    enable: bool = True
    auto_start: bool = True
    path: str = None
    start_flags: str = "--nowebui --skip-python-version-check"


class SDParams(BaseModel):
    host: str = "localhost"
    port: int = 7861
    username: str | None = None
    password: str | None = None
    default_model: str | None = None


class OllamaConfig(BaseModel):
    enabled: bool = False
    host: str = "localhost"
    port: int = 11434

    @property
    def url(self):
        return f"http://{self.host}:{self.port}"


class SDWebUiConfig(BaseModel):
    CONFIG: SDConfig = SDConfig()
    PARAMS: SDParams = SDParams()


class Config(AutoConfig):
    MOTHER_NODE: MotherNode = MotherNode()
    SD_CONFIG: SDWebUiConfig = SDWebUiConfig()
    OLLAMA: OllamaConfig = OllamaConfig()