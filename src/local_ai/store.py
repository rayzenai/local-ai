from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from local_ai.errors import AppError

TERMINAL = {"succeeded", "failed", "cancelled"}


def now() -> str:
    return datetime.now(UTC).isoformat()


def identifier(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class Store:
    def __init__(self, path: Path):
        self.path = path
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
                    payload TEXT NOT NULL, result TEXT, error TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    started_at TEXT, finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, created_at);
                CREATE TABLE IF NOT EXISTS records (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, document TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS records_kind ON records(kind, created_at);
                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY, dataset_id TEXT NOT NULL,
                    file_id TEXT NOT NULL, text TEXT NOT NULL, duration REAL NOT NULL,
                    UNIQUE(dataset_id, file_id)
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def recover(self) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status='failed', error=?, updated_at=?, finished_at=? WHERE status IN ('running','cancelling')",
                (json.dumps({"code": "server_interrupted", "message": "The previous server stopped during this job. Submit it again; partial training checkpoints remain on disk."}), now(), now()),
            )

    def enqueue(self, kind: str, payload: dict, limit: int) -> dict:
        job_id, stamp = identifier("job"), now()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running','cancelling')").fetchone()[0]
            if count >= limit:
                raise AppError("queue_full", "Job queue is full. Retry after a job finishes.", 429)
            db.execute(
                "INSERT INTO jobs(id,kind,status,payload,created_at,updated_at) VALUES(?,?,'queued',?,?,?)",
                (job_id, kind, json.dumps(payload), stamp, stamp),
            )
        return self.job(job_id)

    @staticmethod
    def decode_job(row) -> dict:
        obj = dict(row)
        for key in ("payload", "result", "error"):
            obj[key] = json.loads(obj[key]) if obj[key] is not None else None
        return obj

    def job(self, job_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise AppError("job_not_found", "Job not found.", 404)
        return self.decode_job(row)

    def jobs(self, limit: int = 50) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self.decode_job(row) for row in rows]

    def claim(self) -> dict | None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running', started_at=?, updated_at=? WHERE id=?", (now(), now(), row["id"]))
        return self.job(row["id"])

    def finish(self, job_id: str, status: str, result=None, error=None) -> dict:
        assert status in TERMINAL
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status=CASE WHEN status='cancelling' THEN 'cancelled' ELSE ? END, result=?, error=?, updated_at=?, finished_at=? WHERE id=? AND status IN ('running','cancelling')",
                (status, json.dumps(result) if result is not None else None, json.dumps(error) if error is not None else None, now(), now(), job_id),
            )
        return self.job(job_id)

    def cancel(self, job_id: str) -> dict:
        with self.connect() as db:
            db.execute("UPDATE jobs SET status='cancelled',updated_at=?,finished_at=? WHERE id=? AND status='queued'", (now(), now(), job_id))
            db.execute("UPDATE jobs SET status='cancelling',updated_at=? WHERE id=? AND status='running'", (now(), job_id))
        return self.job(job_id)

    def put(self, kind: str, document: dict, record_id: str | None = None) -> dict:
        obj = {**document, "id": record_id or identifier(kind), "created_at": now()}
        with self.connect() as db:
            db.execute("INSERT INTO records VALUES(?,?,?,?)", (obj["id"], kind, json.dumps(obj), obj["created_at"]))
        return obj

    def get(self, kind: str, record_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT document FROM records WHERE kind=? AND id=?", (kind, record_id)).fetchone()
        if row is None:
            raise AppError("not_found", f"{kind.capitalize()} not found.", 404)
        return json.loads(row["document"])

    def list(self, kind: str, limit: int = 100) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT document FROM records WHERE kind=? ORDER BY created_at DESC LIMIT ?", (kind, limit)).fetchall()
        return [json.loads(row["document"]) for row in rows]

    def add_samples(self, dataset_id: str, samples: list[dict], limit: int) -> None:
        self.get("dataset", dataset_id)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT COUNT(*) FROM samples WHERE dataset_id=?", (dataset_id,)).fetchone()[0]
            if count + len(samples) > limit:
                raise AppError("dataset_full", f"Dataset limit is {limit} samples.")
            try:
                db.executemany("INSERT INTO samples(dataset_id,file_id,text,duration) VALUES(?,?,?,?)", [(dataset_id, s["file_id"], s["text"], s["duration"]) for s in samples])
            except sqlite3.IntegrityError as exc:
                raise AppError("duplicate_sample", "An audio file can appear only once in a dataset.", 409) from exc

    def samples(self, dataset_id: str) -> list[dict]:
        self.get("dataset", dataset_id)
        with self.connect() as db:
            rows = db.execute("SELECT file_id,text,duration FROM samples WHERE dataset_id=? ORDER BY id", (dataset_id,)).fetchall()
        return [dict(row) for row in rows]
