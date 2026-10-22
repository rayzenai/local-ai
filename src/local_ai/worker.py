"""Only spawned workers import model frameworks. Every job exits its process."""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path


def download(payload: dict, settings: dict, directory: Path) -> dict:
    from huggingface_hub import hf_hub_download

    from local_ai.catalog import get_model, model_directory

    spec = get_model(payload["model"])
    root = model_directory(Path(settings["data_dir"]), spec.id)
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for filename in spec.files:
        print(f"Downloading {spec.repo}/{filename} at {spec.revision}", flush=True)
        path = hf_hub_download(repo_id=spec.repo, filename=filename, revision=spec.revision, local_dir=root)
        files[filename] = Path(path).stat().st_size
    receipt = {"repo": spec.repo, "revision": spec.revision, "files": files, "license": spec.license}
    temporary = root / "installed.json.tmp"
    temporary.write_text(json.dumps(receipt, indent=2), "utf-8")
    temporary.replace(root / "installed.json")
    return {"model": spec.id, "revision": spec.revision, "downloaded": True}


def require_cuda(settings: dict, *, image: bool = False):
    import torch

    torch.set_num_threads(settings["cpu_threads"])
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        if settings["allow_cpu"] and not image:
            return torch, "cpu"
        raise RuntimeError("CUDA is unavailable. Install the NVIDIA runtime with local-ai setup and update the NVIDIA driver.")
    if image:
        if torch.cuda.get_device_capability(0)[0] < 10:
            raise RuntimeError("The bundled NVFP4 image model requires a Blackwell/RTX 50-series GPU.")
        major = int((torch.version.cuda or "0").split(".")[0])
        if major < 13:
            raise RuntimeError("Native NVFP4 requires a PyTorch CUDA 13.0+ build.")
    torch.cuda.reset_peak_memory_stats()
    return torch, "cuda"


def gpu_metrics(torch, device: str) -> dict:
    if device != "cuda":
        return {"device": device}
    return {"device": torch.cuda.get_device_name(0),
            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved()}


def speech(payload: dict, settings: dict, directory: Path) -> dict:
    import wave

    import numpy as np
    from voxcpm import VoxCPM

    from local_ai.audio import split_text
    from local_ai.catalog import model_directory

    torch, device = require_cuda(settings)
    kwargs = {"load_denoiser": False, "optimize": False, "device": device, "local_files_only": True}
    adapter = payload.get("adapter_path")
    if adapter:
        # Use the trained adapter's actual rank/targets, not the runtime's default LoRA configuration.
        info = json.loads((Path(adapter) / "lora_config.json").read_text("utf-8"))
        from voxcpm.model.voxcpm import LoRAConfig

        kwargs["lora_config"] = LoRAConfig(**info["lora_config"])
        kwargs["lora_weights_path"] = adapter
    model = VoxCPM.from_pretrained(str(model_directory(Path(settings["data_dir"]), payload["model"])), **kwargs)
    rate = model.tts_model.sample_rate
    output = directory / "speech.wav"
    chunks = split_text(payload["input"], payload["chunk_chars"])
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        with torch.inference_mode():
            for index, chunk in enumerate(chunks):
                print(f"Synthesizing passage {index + 1}/{len(chunks)}", flush=True)
                generation = {"text": chunk, "cfg_value": 2.0, "inference_timesteps": payload["steps"],
                              "max_len": 512, "normalize": False, "denoise": False,
                              "retry_badcase": False, "seed": (payload["seed"] + index) % 2**32}
                if payload.get("reference_path"):
                    generation["prompt_wav_path"] = payload["reference_path"]
                    generation["prompt_text"] = payload["reference_text"]
                    if payload["model"] == "voxcpm-2":
                        generation["reference_wav_path"] = payload["reference_path"]
                audio = model.generate(**generation)
                pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
                wav.writeframes(pcm.tobytes())
                if index + 1 < len(chunks):
                    wav.writeframes(b"\x00\x00" * int(rate * 0.12))
                del audio, pcm
    return {"files": [{"path": output.name, "media_type": "audio/wav"}],
            "metrics": {**gpu_metrics(torch, device), "passages": len(chunks), "sample_rate": rate}}


def image(payload: dict, settings: dict, directory: Path) -> dict:
    import subprocess

    import httpx

    from local_ai.catalog import model_directory
    from local_ai.workflows import z_image_graph

    # Do not import torch in this parent process: its CUDA context would remain allocated while
    # ComfyUI starts and compete for this card's 8GB. ComfyUI performs device validation itself.
    root = Path(settings["data_dir"])
    model_root = model_directory(root, payload["model"]) / "split_files"
    extra = directory / "extra-model-paths.yaml"
    extra.write_text(json.dumps({"local_ai": {
        "base_path": str(model_root), "diffusion_models": "diffusion_models",
        "text_encoders": "text_encoders", "vae": "vae"}}), "utf-8")
    comfy = root / "sources" / "comfyui"
    port = settings["comfy_port"]
    command = [sys.executable, str(comfy / "main.py"), "--listen", "127.0.0.1", "--port", str(port),
               "--extra-model-paths-config", str(extra), "--output-directory", str(directory / "comfy-output"),
               "--user-directory", str(directory / "comfy-user"), "--temp-directory", str(directory / "comfy-temp"),
               "--reserve-vram", str(settings["reserve_vram_gb"]), "--cache-none",
               "--use-pytorch-cross-attention", "--disable-all-custom-nodes", "--disable-auto-launch",
               "--disable-metadata", "--disable-partner-nodes"]
    base = f"http://127.0.0.1:{port}"
    with httpx.Client(trust_env=False, timeout=10) as client:
        try:
            client.get(base + "/system_stats")
        except httpx.TransportError:
            pass
        else:
            raise RuntimeError(f"Port {port} is in use. Change LOCAL_AI_COMFY_PORT to a dedicated free port.")
        process = subprocess.Popen(command, cwd=comfy)
        try:
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError("ComfyUI exited during startup. Check the worker log.")
                try:
                    response = client.get(base + "/system_stats")
                    response.raise_for_status()
                    break
                except httpx.TransportError:
                    if time.monotonic() > deadline:
                        raise RuntimeError("ComfyUI startup timed out.") from None
                    time.sleep(1)
            response = client.post(base + "/prompt", json={"prompt": z_image_graph(payload)})
            if response.status_code != 200:
                raise RuntimeError(f"ComfyUI rejected the bundled graph: {response.text[:3000]}")
            prompt_id = response.json()["prompt_id"]
            while True:
                if process.poll() is not None:
                    raise RuntimeError("ComfyUI exited while generating an image.")
                response = client.get(base + f"/history/{prompt_id}")
                response.raise_for_status()
                entry = response.json().get(prompt_id)
                if entry:
                    if entry.get("status", {}).get("status_str") == "error":
                        raise RuntimeError(f"Image generation failed: {json.dumps(entry.get('status'))[:3000]}")
                    images = entry.get("outputs", {}).get("10", {}).get("images", [])
                    if not images:
                        raise RuntimeError("ComfyUI finished without an output image.")
                    params = {k: images[0][k] for k in ("filename", "subfolder", "type")}
                    with client.stream("GET", base + "/view", params=params) as stream:
                        stream.raise_for_status()
                        with (directory / "image.png").open("wb") as target:
                            for block in stream.iter_bytes(1024 * 1024):
                                target.write(block)
                    return {"files": [{"path": "image.png", "media_type": "image/png"}],
                            "metrics": {"backend": "comfyui", "precision": "nvfp4", "seed": payload["seed"]}}
                time.sleep(0.5)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def train(payload: dict, settings: dict, directory: Path) -> dict:
    import runpy

    from local_ai.training import write_training_files

    torch, device = require_cuda(settings)
    if device != "cuda":
        raise RuntimeError("Voice training requires an NVIDIA CUDA GPU; CPU training is disabled.")
    config = write_training_files(Path(settings["data_dir"]), directory, payload)
    script = Path(settings["data_dir"]) / "sources" / "voxcpm" / "scripts" / "train_voxcpm_finetune.py"
    sys.argv = [str(script), "--config_path", str(config)]
    runpy.run_path(str(script), run_name="__main__")
    checkpoint = directory / "checkpoints" / "latest"
    if not (checkpoint / "lora_weights.safetensors").is_file():
        raise RuntimeError("Training completed without a LoRA safetensors checkpoint. Check training logs.")
    import zipfile

    with zipfile.ZipFile(directory / "voice-adapter.zip", "w", compression=zipfile.ZIP_STORED) as archive:
        for name in ("lora_weights.safetensors", "lora_config.json"):
            archive.write(checkpoint / name, name)
    return {"adapter_path": "checkpoints/latest", "files": [{"path": "voice-adapter.zip", "media_type": "application/zip"}],
            "metrics": {**gpu_metrics(torch, device), "training_steps": payload["steps"]}}


def main() -> None:
    request_path = Path(sys.argv[1]).resolve()
    directory = request_path.parent
    start = time.monotonic()
    code = 0
    try:
        request = json.loads(request_path.read_text("utf-8"))
        handlers = {"download": download, "image": image, "speech": speech, "training": train}
        result = handlers[request["kind"]](request["payload"], request["settings"], directory)
        result.setdefault("metrics", {})["wall_seconds"] = round(time.monotonic() - start, 3)
    except Exception as exc:
        traceback.print_exc()
        message = str(exc)
        oom = "out of memory" in message.lower()
        if oom:
            message = "GPU memory limit exceeded. Use the smaller voice model, shorter audio samples, or a smaller image. Close other GPU applications. Logs include the original error."
        result = {"error": {"code": "gpu_out_of_memory" if oom else "runtime_error", "message": message[:3000]}}
        code = 1
    temporary = directory / "result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2), "utf-8")
    temporary.replace(directory / "result.json")
    sys.exit(code)


if __name__ == "__main__":
    main()
