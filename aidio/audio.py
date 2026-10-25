"""Audio preprocessing pipeline.

Converts any supported audio file (WAV, MP3, …) into a list of
base64-encoded WAV chunks ready for the LLM.

Pipeline per chunk:
    1. Load with ``soundfile``
    2. Resample to 16 kHz (model requirement)
    3. Mix to mono (channel averaging, NOT XOR)
    4. Normalize to float32 in [-1, 1]
    5. Encode as base64 WAV blob
"""

from __future__ import annotations

import base64
import io
import wave
from dataclasses import dataclass

import librosa
import numpy as np
import soundfile as sf


@dataclass
class AudioChunk:
    """A single chunk of preprocessed audio ready for the API."""

    base64_wav: str
    duration_seconds: float


def _wav_bytes_to_base64(y: np.ndarray, sr: int) -> str:
    """Encode a float32 mono array as a WAV file and return base64.

    Parameters
    ----------
    y : np.ndarray
        Audio samples in [-1, 1], float32, mono.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    str
        Base64-encoded WAV bytes.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        # Convert float32 [-1,1] → int16
        int16_data = np.clip(y * 32767.0, -32768, 32767).astype(np.int16)
        wf.writeframes(int16_data.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _load_and_preprocess(path: str, target_sr: int) -> tuple[np.ndarray, int]:
    """Load an audio file and resample to ``target_sr``.

    Returns
    -------
    tuple[np.ndarray, int]
        (stereo/mono float32 array, original sample rate)
    """
    y, sr = sf.read(path, dtype="float32")
    # Convert to mono before resampling for efficiency
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y, target_sr


def _to_mono(y: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono by averaging channels.

    Uses channel averaging (NOT XOR) to avoid destructive interference.
    """
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    return y


def _normalize(y: np.ndarray) -> np.ndarray:
    """Normalize audio to the range [-1, 1]."""
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)


def preprocess_audio(
    path: str,
    target_sr: int = 16000,
    max_chunk_seconds: float = 2400,
) -> list[AudioChunk]:
    """Preprocess an audio file into API-ready chunks.

    Parameters
    ----------
    path : str
        Path to the audio file (WAV, MP3, FLAC, …).
    target_sr : int
        Target sample rate in Hz (default 16000).
    max_chunk_seconds : float
        Maximum duration per chunk in seconds.

    Returns
    -------
    list[AudioChunk]
        One or more chunks, each base64-encoded and ready for the LLM.
    """
    y, sr = _load_and_preprocess(path, target_sr)
    y = _to_mono(y)
    y = _normalize(y)

    chunk_size = int(max_chunk_seconds * sr)
    chunks: list[AudioChunk] = []
    for start in range(0, len(y), chunk_size):
        segment = y[start : start + chunk_size]
        b64 = _wav_bytes_to_base64(segment, sr)
        chunks.append(
            AudioChunk(
                base64_wav=b64,
                duration_seconds=len(segment) / sr,
            )
        )
    return chunks
