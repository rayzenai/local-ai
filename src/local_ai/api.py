from __future__ import annotations

import hmac
import shutil
import uuid
from contextlib import asynccontextmanager

import psutil
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from local_ai.audio import inspect_wav
from local_ai.catalog import catalog, get_model, model_installed, model_directory
from local_ai.config import InstanceLock, Settings
from local_ai.errors import AppError
from local_ai.executor import Executor, JobRunner
from local_ai.schemas import (
    DatasetCreate, ImageRequest, SamplesCreate, SpeechRequest, TrainingRequest, VoiceCreate,
)
from local_ai.store import Store


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings.load()
    config.prepare()
    store = Store(config.data_dir / "local-ai.sqlite3")
    executor = Executor(config, store)
    runner = JobRunner(store, executor)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        with InstanceLock(config.data_dir / "server.lock"):
            runner.start()
            try:
                yield
            finally:
                await runner.close()

    app = FastAPI(title="RayzenAI Local AI", version="0.1.0", lifespan=lifespan,
                  description="Private local image and voice generation API. Models run only in isolated worker processes.")
    app.state.settings, app.state.store, app.state.runner = config, store, runner

    @app.exception_handler(AppError)
    async def app_error_handler(_, exc: AppError):
        return JSONResponse(
            status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message}}
        )

    async def authorize(authorization: str | None = Header(default=None)):
        if config.api_key:
            token = authorization.removeprefix("Bearer ") if authorization else ""
            if not hmac.compare_digest(token, config.api_key):
                raise HTTPException(401, "Use the configured bearer API key.", headers={"WWW-Authenticate": "Bearer"})

    auth = [Depends(authorize)]

    def create_job(kind: str, model: str, payload: dict) -> dict:
        executor.preflight(kind, model)
        job = store.enqueue(kind, payload, config.queue_limit)
        runner.wake.set()
        return job

    def owned_file(file_id: str, purpose: str | None = None) -> tuple[dict, Path]:
        item = store.get("file", file_id)
        if purpose and item.get("purpose") != purpose:
            raise AppError("wrong_file_purpose", f"This operation requires a {purpose} file.", 422)
        path = (config.data_dir / item["path"]).resolve()
        if not path.is_relative_to(config.data_dir.resolve()) or not path.is_file():
            raise AppError("file_missing", "The stored file is no longer available.", 404)
        return item, path

    @app.get("/health")
    @app.get("/v1/health", dependencies=auth)
    async def health():
        return {"status": "ok", "version": "0.1.0", "gpu_jobs": "serialized",
                "memory": {"system_available_bytes": psutil.virtual_memory().available}}

    @app.get("/v1/models", dependencies=auth)
    async def models():
        return {"data": catalog(config.data_dir)}

    @app.post("/v1/models/{model_id}/download", status_code=202, dependencies=auth)
    async def download_model(model_id: str):
        spec = get_model(model_id)
        if model_installed(config.data_dir, spec):
            return {"status": "installed", "model": model_id}
        return create_job("download", model_id, {"model": model_id})

    @app.post("/v1/images/generations", status_code=202, dependencies=auth)
    @app.post("/v1/images", status_code=202, dependencies=auth)
    async def generate_image(request: ImageRequest):
        return create_job("image", request.model, request.model_dump())

    @app.post("/v1/audio/speech", status_code=202, dependencies=auth)
    @app.post("/v1/speech", status_code=202, dependencies=auth)
    async def speech(request: SpeechRequest):
        payload = request.model_dump()
        voice_id = payload.pop("voice")
        adapter_id = payload.pop("adapter")
        if voice_id:
            voice = store.get("voice", voice_id)
            _, reference = owned_file(voice["file_id"], "voice-reference")
            payload["reference_path"], payload["reference_text"] = str(reference), voice["transcript"]
        if adapter_id:
            adapter = store.get("adapter", adapter_id)
            if adapter["model"] != request.model:
                raise AppError("adapter_model_mismatch", "The adapter was trained for a different speech model.", 422)
            payload["adapter_path"] = str((config.data_dir / adapter["path"]).resolve())
        payload["model"] = request.model
        return create_job("speech", request.model, payload)

    @app.post("/v1/files", status_code=201, dependencies=auth)
    async def upload_file(file: UploadFile = File(...), purpose: str = Form("voice-reference")):
        if purpose not in {"voice-reference", "training-audio"}:
            raise AppError("invalid_purpose", "purpose must be voice-reference or training-audio.")
        limit = config.max_upload_mb * 1024 * 1024
        folder = config.data_dir / "files" / uuid.uuid4().hex
        folder.mkdir(parents=True)
        path = folder / "audio.wav"
        total = 0
        try:
            with path.open("wb") as target:
                while block := await file.read(1024 * 1024):
                    total += len(block)
                    if total > limit:
                        raise AppError("upload_too_large", f"Maximum upload size is {config.max_upload_mb} MB.", 413)
                    target.write(block)
            audio = inspect_wav(path)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        item = store.put("file", {"path": str(path.relative_to(config.data_dir)), "filename": "audio.wav",
                                   "media_type": "audio/wav", "bytes": total, "purpose": purpose, **audio})
        return {key: item[key] for key in ("id", "filename", "media_type", "bytes", "purpose", "duration", "sample_rate", "channels")}

    @app.post("/v1/voices", status_code=201, dependencies=auth)
    async def create_voice(request: VoiceCreate):
        _, path = owned_file(request.file_id, "voice-reference")
        return store.put("voice", {"name": request.name, "file_id": request.file_id,
                                    "transcript": request.transcript, "duration": inspect_wav(path)["duration"]})

    @app.get("/v1/voices", dependencies=auth)
    async def list_voices():
        return {"data": store.list("voice")}

    @app.post("/v1/datasets", status_code=201, dependencies=auth)
    async def create_dataset(request: DatasetCreate):
        return store.put("dataset", {"name": request.name})

    @app.get("/v1/datasets", dependencies=auth)
    async def list_datasets():
        return {"data": store.list("dataset")}

    @app.post("/v1/datasets/{dataset_id}/samples", status_code=201, dependencies=auth)
    async def add_samples(dataset_id: str, request: SamplesCreate):
        rows = []
        for sample in request.samples:
            item, path = owned_file(sample.file_id, "training-audio")
            rows.append({"file_id": item["id"], "text": sample.text,
                         "duration": inspect_wav(path)["duration"], "path": str(path)})
        store.add_samples(dataset_id, rows, config.max_dataset_samples)
        return {"dataset_id": dataset_id, "added": len(rows), "samples": store.samples(dataset_id)}

    @app.get("/v1/datasets/{dataset_id}/samples", dependencies=auth)
    async def get_samples(dataset_id: str):
        return {"dataset_id": dataset_id, "data": store.samples(dataset_id)}

    @app.post("/v1/voices/train", status_code=202, dependencies=auth)
    @app.post("/v1/trainings", status_code=202, dependencies=auth)
    async def train_voice(request: TrainingRequest):
        samples = store.samples(request.dataset_id)
        if len(samples) < 2:
            raise AppError("dataset_too_small", "Voice adaptation requires at least two recorded and transcribed samples.", 422)
        if sum(row["duration"] for row in samples) < 30:
            raise AppError("dataset_too_short", "Use at least 30 seconds of clean speech; 5–20 minutes is a better target.", 422)
        payload = {**request.model_dump(), "samples": [
            {**row, "path": str(owned_file(row["file_id"], "training-audio")[1])} for row in samples
        ]}
        return create_job("training", request.model, payload)

    @app.get("/v1/adapters", dependencies=auth)
    async def list_adapters():
        return {"data": store.list("adapter")}

    @app.get("/v1/jobs", dependencies=auth)
    async def jobs(limit: int = 50):
        return {"data": store.jobs(max(1, min(limit, 100)))}

    @app.get("/v1/jobs/{job_id}", dependencies=auth)
    async def job(job_id: str):
        return store.job(job_id)

    @app.delete("/v1/jobs/{job_id}", dependencies=auth)
    async def cancel_job(job_id: str):
        result = store.cancel(job_id)
        runner.wake.set()
        return result

    @app.get("/v1/files/{file_id}", dependencies=auth)
    async def get_file(file_id: str):
        item, path = owned_file(file_id)
        return FileResponse(path, media_type=item["media_type"], filename=item["filename"])

    return app


app = create_app()
