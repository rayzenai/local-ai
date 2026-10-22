from __future__ import annotations

import json
from pathlib import Path

from local_ai.catalog import get_model, model_directory


def training_config(data_dir: Path, job_dir: Path, payload: dict) -> dict:
    """Conservative DiT-only adapter: frozen language model, one sample, no loader workers."""
    spec = get_model(payload["model"])
    return {
        "pretrained_path": str(model_directory(data_dir, spec.id)),
        "train_manifest": str(job_dir / "train.jsonl"),
        "val_manifest": "",
        "sample_rate": 44100,
        "batch_size": 1,
        "grad_accum_steps": payload["gradient_accumulation"],
        "num_workers": 0,
        "preprocessing_num_workers": 1,
        "num_iters": payload["steps"],
        "max_steps": payload["steps"],
        "log_interval": 1,
        "valid_interval": payload["steps"] + 1,
        "save_interval": min(payload["save_every"], payload["steps"]),
        "learning_rate": payload["learning_rate"],
        "weight_decay": 0.01,
        "warmup_steps": min(20, max(0, payload["steps"] // 10)),
        "max_batch_tokens": 512,
        "max_grad_norm": 1.0,
        "save_path": str(job_dir / "checkpoints"),
        "tensorboard": str(job_dir / "tensorboard"),
        "lambdas": {"loss/diff": 1.0, "loss/stop": 1.0},
        "lora": {"enable_lm": False, "enable_dit": True, "enable_proj": False,
                 "r": payload["rank"], "alpha": payload["rank"] * 2, "dropout": 0.0},
        "hf_model_id": spec.repo,
        "distribute": True,
    }


def write_training_files(data_dir: Path, job_dir: Path, payload: dict) -> Path:
    manifest = job_dir / "train.jsonl"
    with manifest.open("w", encoding="utf-8", newline="\n") as file:
        for sample in payload["samples"]:
            file.write(json.dumps({"audio": sample["path"], "text": sample["text"],
                                   "duration": sample["duration"]}, ensure_ascii=False) + "\n")
    # JSON is a YAML subset; upstream's YAML parser accepts this exact generated config.
    config_path = job_dir / "training.yaml"
    config_path.write_text(json.dumps(training_config(data_dir, job_dir, payload), indent=2), "utf-8")
    return config_path
