from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data_dir: Path = Path(".local-ai")
    cpu_threads: int = Field(2, ge=1, le=16)
    queue_limit: int = Field(16, ge=1, le=256)
    max_upload_mb: int = Field(32, ge=1, le=256)
    max_dataset_samples: int = Field(5000, ge=2, le=50000)
    max_job_seconds: int = Field(1800, ge=5, le=86400)
    max_training_seconds: int = Field(21600, ge=5, le=604800)
    max_log_mb: int = Field(8, ge=1, le=64)
    reserve_vram_gb: float = Field(1.0, ge=0.5, le=8)
    comfy_port: int = Field(18188, ge=1024, le=65535)
    api_key: str | None = Field(None, exclude=True, repr=False)
    allow_cpu: bool = False

    @classmethod
    def load(cls, config: Path | None = None) -> Settings:
        path = config or Path("local-ai.toml")
        values = tomllib.loads(path.read_text("utf-8")) if path.exists() else {}
        for name in cls.model_fields:
            value = os.environ.get(f"LOCAL_AI_{name.upper()}")
            if value is not None:
                values[name] = value
        settings = cls(**values)
        settings.data_dir = settings.data_dir.expanduser().resolve()
        return settings

    def prepare(self) -> None:
        self.data_dir = self.data_dir.expanduser().resolve()
        for folder in ("files", "jobs", "models", "runtimes", "sources", "datasets"):
            (self.data_dir / folder).mkdir(parents=True, exist_ok=True)

    def python_for(self, runtime: str) -> Path:
        directory = self.data_dir / "runtimes" / runtime
        return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


class InstanceLock:
    """An OS lock prevents two API workers from running simultaneous GPU jobs."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self):
        self.handle = self.path.open("a+b")
        self.handle.seek(0)
        self.handle.write(b"0")
        self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError("Another local-ai server uses this data directory. Use one API worker.") from exc
        return self

    def __exit__(self, *_):
        if self.handle:
            self.handle.close()
