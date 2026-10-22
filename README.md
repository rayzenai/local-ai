# RayzenAI Local AI

An open-source, Windows-first local API for image generation, natural voice cloning, and voice LoRA adaptation. It targets an NVIDIA RTX 50-series desktop with 8GB VRAM and keeps the API process light: the API has no PyTorch dependency, GPU work is serialized, each model job runs in a short-lived child process, and models are unloaded when that process exits.

## What it runs

| Task | Model/runtime | Notes |
|---|---|---|
| Images | Z-Image Turbo NVFP4, ComfyUI | 1024px, one output at a time, 8 steps, local generation. CUDA 13+ is needed for the native RTX 50 NVFP4 path. |
| Voiceovers and cloning | VoxCPM 1.5, BF16-capable PyTorch | English and Chinese; text-to-speech or cloning from a short WAV plus transcript. |
| Voice model adaptation | VoxCPM upstream LoRA trainer | Produces an adapter for the same VoxCPM 1.5 voice runtime. Training is experimental on 8GB; use a short dataset, rank 4, batch size 1, and close other GPU applications. It can run out of VRAM. |

The API and application code use Apache-2.0. Model weights and upstream runtimes keep their own licenses; check the model card before using generated media commercially. Z-Image Turbo and VoxCPM 1.5 model cards currently state Apache-2.0. This repository does not bundle model weights.

## Windows install

Use 64-bit Python 3.11, Git, and a current NVIDIA driver. Keep at least 20GB free for Python runtimes, ComfyUI, model weights, and caches. Start PowerShell in this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
local-ai setup --components voice images training
local-ai doctor
local-ai serve
```

Runtime packages live in `.local-ai\runtimes`, separately from the API. Install only `voice` or `images` if that is all you need. The training component adds training dependencies to the voice environment. `setup` downloads upstream source and packages; it does not download the multi-gigabyte model weights. If PowerShell blocks activation, run `.\.venv\Scripts\python.exe -m pip install -e .` and then `.\.venv\Scripts\local-ai.exe serve`.

The API binds to `127.0.0.1` only. Open `http://127.0.0.1:8000/docs` for interactive API documentation. To permit a trusted LAN client, set `LOCAL_AI_HOST=0.0.0.0` and set a long random `LOCAL_AI_API_KEY`; requests then need `Authorization: Bearer <key>`. Do not expose an unauthenticated server to a network.

## First run

List the available models, then download the model you want. Downloads run as background jobs and can take time:

```powershell
curl.exe http://127.0.0.1:8000/v1/models
curl.exe -X POST http://127.0.0.1:8000/v1/models/z-image-turbo/download
curl.exe -X POST http://127.0.0.1:8000/v1/models/voxcpm-1.5/download
```

Poll the returned job ID at `/v1/jobs/{job_id}` until its status is `succeeded`. A model is downloaded only once and stays on disk; inference memory is released after each job.

### Generate an image

```powershell
$body = @{prompt="A cinematic portrait of a Nepalese mountain guide at sunrise"; size="1024x1024"; seed=42} | ConvertTo-Json
$job = Invoke-RestMethod http://127.0.0.1:8000/v1/images/generations -Method Post -ContentType application/json -Body $body
```

Poll the job. Its `result.files[0].url` returns the PNG.

### Make a voice clone

Upload a clean 16-bit PCM mono or stereo WAV (up to 120 seconds and the configured upload limit), then create a voice with the exact spoken transcript:

```powershell
$upload = Invoke-RestMethod http://127.0.0.1:8000/v1/files -Method Post -Form @{file=Get-Item .\reference.wav; purpose="voice-reference"}
$voice = Invoke-RestMethod http://127.0.0.1:8000/v1/voices -Method Post -ContentType application/json -Body (@{name="Guide"; file_id=$upload.id; transcript="The exact words spoken in reference.wav."} | ConvertTo-Json)
$speech = Invoke-RestMethod http://127.0.0.1:8000/v1/audio/speech -Method Post -ContentType application/json -Body (@{model="voxcpm-1.5"; input="Welcome to the Himalayas."; voice=$voice.id} | ConvertTo-Json)
```

Poll the speech job and download its WAV using the returned file URL. A transcript that matches the reference audio improves cloning.

### Train a voice adapter

Upload several clean recordings with transcripts under `purpose=training-audio`. Create a dataset, add at least two samples totalling 30 seconds or more, and submit `/v1/voices/train` with the dataset ID. The trainer currently uses frozen base weights and a DiT LoRA adapter. Rank 4 reduces trainable memory. Use a consistent speaker, room, microphone, and sample rate; remove music and noise. The API rejects tiny datasets and training is queued with image/speech jobs.

Use the returned `adapter_id` on `/v1/audio/speech` to synthesize with the adapter. Voice cloning with a reference recording remains available without training.

## API surface

- `GET /health`, `GET /v1/health`, `GET /v1/models`
- `POST /v1/models/{model_id}/download`
- `POST /v1/images/generations` (alias `/v1/images`)
- `POST /v1/audio/speech` (alias `/v1/speech`)
- `POST /v1/files`, `GET /v1/files/{file_id}`
- `POST/GET /v1/voices`
- `POST/GET /v1/datasets`, `POST/GET /v1/datasets/{id}/samples`
- `POST /v1/voices/train` (alias `/v1/trainings`), `GET /v1/adapters`
- `GET /v1/jobs`, `GET /v1/jobs/{id}`, `DELETE /v1/jobs/{id}`

Generation and training calls return `202 Accepted` with a queued job. Poll its status; the finished result includes files, resource metrics, and download URLs. Only one GPU job runs at once, regardless of queued requests.

## Configuration

Create `local-ai.toml` in the current directory, or set the matching `LOCAL_AI_` environment variables:

```toml
data_dir = ".local-ai"
cpu_threads = 2
queue_limit = 16
max_upload_mb = 32
max_dataset_samples = 5000
reserve_vram_gb = 1.0
comfy_port = 18188
```

Increase `reserve_vram_gb` if the desktop needs more VRAM while generating. Reduce `cpu_threads` to leave more CPU time for other applications. Do not run multiple API workers; the server enforces one process per data directory.

## Privacy and safety

The server binds to loopback by default, sends no prompts or recordings to a hosted inference service, and disables Hugging Face telemetry in workers. Model downloads contact Hugging Face, and runtime setup contacts GitHub and PyPI. Uploaded audio, generated files, voice metadata, datasets, and adapters remain in the configured data directory until you remove them. Only clone voices you have permission to use.

## Development

```powershell
python -m pip install -e ".[dev]"
ruff check src
```

The API is intentionally independent of CUDA and ML libraries. ML dependencies are installed only into the task-specific runtime environments.
