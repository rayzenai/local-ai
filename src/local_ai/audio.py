from __future__ import annotations

import re
import wave
from pathlib import Path

from local_ai.errors import AppError


def inspect_wav(path: Path) -> dict:
    try:
        with wave.open(str(path), "rb") as wav:
            channels, width, rate, frames = wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
            if wav.getcomptype() != "NONE" or width != 2 or channels not in (1, 2):
                raise ValueError("Use uncompressed 16-bit PCM WAV, mono or stereo.")
            if rate < 8000 or rate > 96000 or frames <= 0:
                raise ValueError("Use a non-empty WAV with a sample rate from 8kHz to 96kHz.")
            if wav.getnframes() * channels * width > path.stat().st_size:
                raise ValueError("WAV header declares more audio data than the file contains.")
            duration = frames / rate
            if duration > 120:
                raise ValueError("Each uploaded recording must be at most 120 seconds.")
            return {"duration": duration, "sample_rate": rate, "channels": channels}
    except (wave.Error, EOFError, ValueError, OSError) as exc:
        raise AppError("invalid_audio", str(exc)) from exc


def split_text(text: str, budget: int) -> list[str]:
    """Bound each inference, preserving every character except separator whitespace."""
    pieces: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?。！？])\s+|\n+", text.strip()):
        for word in sentence.split():
            if current and len(current) + len(word) + 1 > budget:
                pieces.append(current)
                current = ""
            while len(word) > budget:
                if current:
                    pieces.append(current)
                    current = ""
                pieces.append(word[:budget])
                word = word[budget:]
            if word:
                current = f"{current} {word}".strip()
        if current:
            pieces.append(current)
            current = ""
    return pieces
