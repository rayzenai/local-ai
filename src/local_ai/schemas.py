from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ImageRequest(Request):
    model: Literal["z-image-turbo"] = "z-image-turbo"
    prompt: str = Field(min_length=1, max_length=4000)
    size: Literal["512x512", "768x768", "1024x1024"] = "1024x1024"
    n: Literal[1] = 1
    steps: int = Field(8, ge=4, le=12)
    seed: int = Field(42, ge=0, le=2**63 - 1)


class SpeechRequest(Request):
    model: Literal["voxcpm-1.5", "voxcpm-2"] = "voxcpm-1.5"
    input: str = Field(min_length=1, max_length=8000)
    voice: str | None = None
    adapter: str | None = None
    response_format: Literal["wav"] = "wav"
    steps: int = Field(10, ge=4, le=20)
    seed: int = Field(42, ge=0, le=2**32 - 1)
    chunk_chars: int = Field(180, ge=40, le=300)


class VoiceCreate(Request):
    name: str = Field(min_length=1, max_length=100)
    file_id: str
    transcript: str = Field(min_length=1, max_length=2000)


class DatasetCreate(Request):
    name: str = Field(min_length=1, max_length=100)


class SampleCreate(Request):
    file_id: str
    text: str = Field(min_length=1, max_length=1000)


class SamplesCreate(Request):
    samples: list[SampleCreate] = Field(min_length=1, max_length=100)


class TrainingRequest(Request):
    dataset_id: str
    name: str = Field(min_length=1, max_length=100)
    model: Literal["voxcpm-1.5"] = "voxcpm-1.5"
    steps: int = Field(500, ge=1, le=10000)
    learning_rate: float = Field(0.0001, gt=0, le=0.001)
    rank: Literal[4, 8] = 8
    gradient_accumulation: int = Field(8, ge=1, le=32)
    save_every: int = Field(100, ge=1, le=1000)

    @field_validator("learning_rate")
    @classmethod
    def finite_rate(cls, value: float) -> float:
        import math

        if not math.isfinite(value):
            raise ValueError("learning_rate must be finite")
        return value
