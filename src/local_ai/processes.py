from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path


async def stop_tree(process: asyncio.subprocess.Process, grace: float = 5) -> None:
    if process.returncode is not None:
        return
    if os.name == "nt":
        # taskkill /T is essential: terminating only the wrapper can strand a GPU process.
        killer = await asyncio.create_subprocess_exec(
            "taskkill", "/PID", str(process.pid), "/T", "/F",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        await killer.wait()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        await asyncio.wait_for(process.wait(), timeout=grace)
    except TimeoutError:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        await process.wait()


class ProcessFailure(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


async def run_isolated(
    command: list[str], *, cwd: Path, env: dict[str, str], log: Path,
    timeout: int, log_limit: int, cancelled: Callable[[], Awaitable[bool]],
) -> int:
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with log.open("ab", buffering=0) as output:
        process = await asyncio.create_subprocess_exec(
            *command, cwd=cwd, env=env, stdout=output, stderr=asyncio.subprocess.STDOUT, **options,
        )
        waiter = asyncio.create_task(process.wait())
        start = asyncio.get_running_loop().time()
        try:
            while not waiter.done():
                if await cancelled():
                    raise ProcessFailure("cancelled", "Job cancelled.")
                if asyncio.get_running_loop().time() - start > timeout:
                    raise ProcessFailure("job_timeout", f"Job exceeded its {timeout}-second time limit.")
                if log.stat().st_size > log_limit:
                    raise ProcessFailure("log_limit", "Worker exceeded the configured log size limit.")
                await asyncio.wait({waiter}, timeout=0.5)
            return waiter.result()
        finally:
            # Covers server shutdown, cancellation, errors, and subprocess descendants.
            await stop_tree(process)
            if not waiter.done():
                await waiter
