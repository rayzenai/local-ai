from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from local_ai.errors import AppError


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    task: str
    runtime: str
    repo: str
    revision: str
    files: tuple[str, ...]
    license: str = "Apache-2.0"
    training: bool = False
    notes: str = ""


VOICE_FILES = (
    "config.json", "model.safetensors", "audiovae.pth", "tokenizer.json",
    "special_tokens_map.json", "tokenizer_config.json", "README.md",
)
MODELS = {
    spec.id: spec
    for spec in (
        ModelSpec(
            "z-image-turbo", "Z-Image Turbo NVFP4", "image", "images",
            "Comfy-Org/z_image_turbo", "6fc90a3b1b653e935a0d175e260736de25b84df5",
            ("split_files/diffusion_models/z_image_turbo_nvfp4.safetensors",
             "split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors",
             "split_files/vae/ae.safetensors", "README.md"),
            notes="RTX 50-series, native NVFP4, CUDA 13.0+. One 1024-square image at a time; staged offload.",
        ),
        ModelSpec(
            "voxcpm-1.5", "VoxCPM 1.5", "speech", "voice", "openbmb/VoxCPM1.5",
            "8cdc403854cda0e3af12252d27da038fda5982ac", VOICE_FILES, training=True,
            notes="Default smaller voice model. English and Chinese; reference cloning and LoRA adaptation.",
        ),
        ModelSpec(
            "voxcpm-2", "VoxCPM 2", "speech", "voice", "openbmb/VoxCPM2",
            "32279effe8c19989596f05d353d1447f51d9e915", VOICE_FILES + ("tokenization_voxcpm2.py",),
            notes="Optional multilingual quality model. Higher peak VRAM; 8GB fit is not guaranteed. Training disabled in the 8GB profile.",
        ),
    )
}


def get_model(model_id: str) -> ModelSpec:
    try:
        return MODELS[model_id]
    except KeyError as exc:
        raise AppError("unknown_model", f"Unknown model '{model_id}'. See GET /v1/models.", 404) from exc


def model_directory(data_dir: Path, model_id: str) -> Path:
    return data_dir / "models" / get_model(model_id).id


def model_installed(data_dir: Path, spec: ModelSpec) -> bool:
    root = model_directory(data_dir, spec.id)
    try:
        receipt = json.loads((root / "installed.json").read_text("utf-8"))
        return receipt["revision"] == spec.revision and all(
            (root / filename).is_file() and (root / filename).stat().st_size == receipt["files"][filename]
            for filename in spec.files
        )
    except (OSError, ValueError, KeyError):
        return False


def catalog(data_dir: Path) -> list[dict]:
    return [{**asdict(spec), "installed": model_installed(data_dir, spec)} for spec in MODELS.values()]
