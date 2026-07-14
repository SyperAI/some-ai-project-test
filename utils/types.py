import hashlib
import io
from enum import Enum
from typing import Optional, List, Any

from PIL.Image import Image
from PIL.PngImagePlugin import PngImageFile
from pydantic import BaseModel, RootModel, Field, model_validator, field_validator
from typing_extensions import deprecated
from webuiapi import ADetailer


class AIType(Enum):
    LLM = "llm"
    SD15 = "sd15"
    SDXL = "sdxl"


class TaskType(Enum):
    MODELS_LIST = "models_list"
    API_CALL = "api_call"
    TXT2IMG = "txt2img"
    LLM = "llm"


class Task(BaseModel):
    id: str
    type: TaskType


class SDModelItem(BaseModel):
    title: str
    model_name: str
    hash: Optional[str] = None
    sha256: Optional[str] = None
    filename: str
    config: Optional[str] = None


class SDModelsResponse(RootModel[List[SDModelItem]]):
    pass


class SDLora(BaseModel):
    path: str
    hash: str


class SDADetailer(BaseModel):
    prompt: str = Field(serialization_alias="ad_prompt")
    confidence: float = Field(serialization_alias="ad_confidence")
    denoising_strength: float = Field(serialization_alias="ad_denoising_strength")
    mask_blur: int = Field(serialization_alias="ad_mask_blur")
    model: str = Field(serialization_alias="ad_model")

    def to_args(self) -> dict:
        return self.model_dump(exclude_none=True, by_alias=True)


class T2IRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    seed: int = -1

    steps: int = Field(default=20, ge=1, le=150)
    width: int = Field(default=512, multiple_of=8)
    height: int = Field(default=512, multiple_of=8)
    cfg_scale: float = Field(default=7.0, ge=1.0, le=30.0)

    sampler_name: str = "Euler a"
    batch_size: int = Field(default=1, ge=1)
    n_iter: int = Field(default=1, ge=1)

    enable_hr: bool = False
    hr_scale: float = Field(default=2.0, ge=1.0)
    hr_upscaler: str = "Latent"
    hr_second_pass_steps: int = 0

    model: str = None
    lora_info: Optional[list[SDLora]] = []
    adetailer: SDADetailer | None = None

    def get_adetailer(self) -> list[ADetailer]:
        if self.adetailer is None: return []

        return [ADetailer(**self.adetailer.model_dump(by_alias=True, exclude_none=True))]


class File(BaseModel):
    image: bytes
    name: str = None
    hash: str = None
    mime_type: str = None

    @model_validator(mode="before")
    @classmethod
    def pre_validate(cls, data: Any) -> Any:
        if not isinstance(data, dict): return data

        if isinstance(data['image'], PngImageFile):
            buffer = io.BytesIO()
            data['image'].save(buffer, format='PNG')
            buffer.seek(0)

            data['image'] = buffer.getvalue()
        elif isinstance(data['image'], io.BytesIO):
            data['image'].seek(0)

            data['image'] = data['image'].getvalue()

        file_hash = hashlib.md5(data['image']).hexdigest()
        data['hash'] = file_hash

        data['mime_type'] = "image/png"

        data['name'] = file_hash + ".png"

        return data


class SDFile(File):
    prompt: str = ""
    negative_prompt: str = ""


class T2IResponse(BaseModel):
    files: List[SDFile] | Any
    prompt: str = ""
    negative_prompt: str = ""

    @field_validator("files", mode="before")
    @classmethod
    def pre_validate(cls, value: Any) -> Any:
        if type(value) is not list: return value

        files = [SDFile(image=file_data[0], prompt=file_data[1], negative_prompt=file_data[2]) for file_data in value]

        return files

    def to_form(self):
        files_list = []
        data = self.model_dump(mode="json", exclude={"files"})

        for file in self.files:
            files_list.append(("files", (file.name, file.image, file.mime_type)))
            data[file.name] = file.model_dump_json(exclude={"image"})

        return data, files_list


class LLMRequest(BaseModel):
    model: str
    system_prompt: str
    prompt: str

    max_tokens: int | None = None
    temperature: float | None = None

    @property
    def options(self):
        options = {}

        if self.max_tokens: options["num_predict"] = self.max_tokens
        if self.temperature: options["temperature"] = self.temperature

        return options if len(options) > 1 else None


class LLMResponse(BaseModel):
    answer: str
