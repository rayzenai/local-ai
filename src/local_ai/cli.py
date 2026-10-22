from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path

from local_ai.config import Settings


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def source_checkout(url: str, path: Path) -> None:
    if not (path / ".git").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", url, str(path)])


def runtime(settings: Settings, name: str) -> Path:
    python = settings.python_for(name)
    if not python.is_file():
        print(f"Creating isolated {name} environment at {python.parent.parent}")
        venv.EnvBuilder(with_pip=True, clear=False).create(python.parent.parent)
    return python


def install_component(settings: Settings, component: str) -> None:
    if component in {"voice", "training"}:
        python = runtime(settings, "voice")
        if component == "voice":
            source = settings.data_dir / "sources" / "voxcpm"
            source_checkout("https://github.com/OpenBMB/VoxCPM.git", source)
            run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
            run([str(python), "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu130"])
            run([str(python), "-m", "pip", "install", "-e", str(source)])
            run([str(python), "-m", "pip", "install", "safetensors"])
        if component == "training":
            if not (settings.data_dir / "sources" / "voxcpm" / "scripts" / "train_voxcpm_finetune.py").is_file():
                install_component(settings, "voice")
            run([str(python), "-m", "pip", "install", "argbind", "tensorboardX", "datasets",
                 "accelerate", "transformers", "soundfile"])
        return

    if component == "images":
        python = runtime(settings, "images")
        source = settings.data_dir / "sources" / "comfyui"
        source_checkout("https://github.com/Comfy-Org/ComfyUI.git", source)
        run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(python), "-m", "pip", "install", "torch", "torchvision", "--index-url",
             "https://download.pytorch.org/whl/cu130"])
        run([str(python), "-m", "pip", "install", "-r", str(source / "requirements.txt")])
        run([str(python), "-m", "pip", "install", "comfy-kitchen[cublas]"])
        return
    raise ValueError(f"Unknown component: {component}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="local-ai", description="RayzenAI local image and voice API")
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("setup", help="Install isolated local inference runtimes")
    setup.add_argument("--components", nargs="+", choices=("voice", "images", "training"),
                       default=("voice", "images", "training"))
    sub.add_parser("serve", help="Run the local API (single worker)")
    sub.add_parser("doctor", help="Show local setup and hardware readiness")
    args = parser.parse_args()
    settings = Settings.load()
    settings.prepare()

    if args.command == "setup":
        if os.name != "nt":
            print("Setup targets Windows with an NVIDIA RTX GPU; this command is safe to inspect but installs Windows-targeted runtimes.")
        for component in args.components:
            install_component(settings, component)
        print("Setup finished. Download the models through GET /v1/models and POST /v1/models/{id}/download.")
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("local_ai.api:app", host=os.environ.get("LOCAL_AI_HOST", "127.0.0.1"),
                    port=int(os.environ.get("LOCAL_AI_PORT", "8000")), workers=1, access_log=False)
    elif args.command == "doctor":
        import psutil

        print(f"Python: {sys.version.split()[0]}")
        print(f"Data: {settings.data_dir}")
        print(f"Available system RAM: {psutil.virtual_memory().available / 1024**3:.1f} GiB")
        for name in ("voice", "images"):
            print(f"{name} runtime: {'ready' if settings.python_for(name).is_file() else 'missing'}")
        for spec in __import__("local_ai.catalog", fromlist=["MODELS"]).MODELS.values():
            print(f"{spec.id}: {'downloaded' if __import__('local_ai.catalog', fromlist=['model_installed']).model_installed(settings.data_dir, spec) else 'not downloaded'}")
        print("Idle API does not load model weights or reserve GPU memory.")


if __name__ == "__main__":
    main()
