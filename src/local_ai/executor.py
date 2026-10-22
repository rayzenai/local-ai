from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from local_ai.catalog import get_model, model_installed
from local_ai.config import Settings
from local_ai.errors import AppError
from local_ai.processes import ProcessFailure, run_isolated
from local_ai.store import Store, identifier


class Executor:
    def __init__(self, settings: Settings, store: Store):
        self.settings = settings
        self.store = store

    def preflight(self, kind: str, model_id: str) -> None:
        spec = get_model(model_id)
        if kind == "download":
            return
        if not self.settings.python_for(spec.runtime).is_file():
            raise AppError("runtime_missing", f"Install the {spec.runtime} runtime: local-ai setup --components {spec.runtime}", 503)
        if not model_installed(self.settings.data_dir, spec):
            raise AppError("model_missing", f"Download this model first: POST /v1/models/{model_id}/download", 503)
        if kind == "training" and not (self.settings.data_dir / "sources" / "voxcpm" / "scripts" / "train_voxcpm_finetune.py").is_file():
            raise AppError("trainer_missing", "Run local-ai setup --components training first.", 503)

    async def execute(self, job: dict, stopping: asyncio.Event) -> dict:
        directory = self.settings.data_dir / "jobs" / job["id"]
        directory.mkdir(parents=True, exist_ok=True)
        request = directory / "request.json"
        request.write_text(json.dumps({"kind": job["kind"], "payload": job["payload"],
                                       "settings": self.settings.model_dump(mode="json")}), "utf-8")
        runtime = get_model(job["payload"]["model"]).runtime
        python = Path(sys.executable) if job["kind"] == "download" else self.settings.python_for(runtime)
        env = dict(os.environ)
        env.update({
            "OMP_NUM_THREADS": str(self.settings.cpu_threads),
            "MKL_NUM_THREADS": str(self.settings.cpu_threads),
            "OPENBLAS_NUM_THREADS": str(self.settings.cpu_threads),
            "NUMEXPR_NUM_THREADS": str(self.settings.cpu_threads),
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HOME": str(self.settings.data_dir / "hf-cache"),
            "PYTHONUNBUFFERED": "1",
            "PYTHONUTF8": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        })
        # Add our source, rather than installing the API and its dependencies into every ML environment.
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])

        async def cancelled():
            return stopping.is_set() or self.store.job(job["id"])["status"] == "cancelling"

        timeout = self.settings.max_training_seconds if job["kind"] == "training" else self.settings.max_job_seconds
        if job["kind"] == "download":
            timeout = max(timeout, 14400)
        code = await run_isolated(
            [str(python), "-m", "local_ai.worker", str(request)],
            cwd=directory, env=env, log=directory / "worker.log", timeout=timeout,
            log_limit=self.settings.max_log_mb * 1024**2, cancelled=cancelled,
        )
        report_path = directory / "result.json"
        report = json.loads(report_path.read_text("utf-8")) if report_path.is_file() else {}
        if code or "error" in report:
            error = report.get("error", {})
            raise ProcessFailure(error.get("code", "worker_failed"), error.get("message", f"Worker exited with code {code}. See the job logs."))
        if not report:
            raise ProcessFailure("missing_result", "Worker exited without a result.")
        files = []
        for artifact in report.get("files", []):
            path = (directory / artifact["path"]).resolve()
            if not path.is_relative_to(directory.resolve()) or not path.is_file():
                raise ProcessFailure("invalid_artifact", "Worker returned an invalid output file.")
            record = self.store.put("file", {"path": str(path.relative_to(self.settings.data_dir)),
                                              "filename": path.name, "media_type": artifact["media_type"],
                                              "bytes": path.stat().st_size, "purpose": "output"})
            files.append({"id": record["id"], "url": f"/v1/files/{record['id']}",
                          "media_type": record["media_type"], "bytes": record["bytes"]})
        result = {**report, "files": files}
        if job["kind"] == "training":
            adapter_path = (directory / report["adapter_path"]).resolve()
            if not adapter_path.is_relative_to(directory.resolve()) or not (adapter_path / "lora_config.json").is_file():
                raise ProcessFailure("invalid_adapter", "Trainer did not return a valid adapter directory.")
            adapter = self.store.put("adapter", {
                "name": job["payload"]["name"], "model": job["payload"]["model"],
                "path": str(adapter_path.relative_to(self.settings.data_dir)), "training_job": job["id"],
                "dataset_id": job["payload"]["dataset_id"], "license": "Apache-2.0",
                "reference": job["payload"]["samples"][0],
            }, identifier("adapter"))
            result["adapter_id"] = adapter["id"]
            result.pop("adapter_path", None)
        return result


class JobRunner:
    def __init__(self, store: Store, executor: Executor):
        self.store = store
        self.executor = executor
        self.wake = asyncio.Event()
        self.stopping = asyncio.Event()
        self.task: asyncio.Task | None = None

    def start(self) -> None:
        self.store.recover()
        self.task = asyncio.create_task(self.run(), name="single-gpu-job-runner")
        self.wake.set()

    async def close(self) -> None:
        self.stopping.set()
        self.wake.set()
        if self.task:
            await self.task

    async def run(self) -> None:
        while not self.stopping.is_set():
            self.wake.clear()
            job = self.store.claim()
            if job is None:
                # No polling, model process, or periodic GPU query while idle.
                await self.wake.wait()
                continue
            try:
                result = await self.executor.execute(job, self.stopping)
                if self.stopping.is_set():
                    self.store.finish(job["id"], "failed", error={"code": "server_stopped", "message": "Server stopped during this job."})
                else:
                    self.store.finish(job["id"], "succeeded", result=result)
            except ProcessFailure as exc:
                state = "cancelled" if exc.code == "cancelled" and not self.stopping.is_set() else "failed"
                self.store.finish(job["id"], state, error={"code": exc.code, "message": str(exc)})
            except Exception as exc:
                self.store.finish(job["id"], "failed", error={"code": "job_failed", "message": str(exc)[:2000]})
