"""Audio loading and vocal separation.

Connects to: asr.py — provides mono 16kHz float32 numpy arrays that the
transcription backends expect. Vocal separation (demucs) is optional but
strongly improves lyric intelligibility on produced music.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

from aidio.lyrics.stats import timed_operation_with_stats

logger = logging.getLogger(__name__)


def load_audio_16k(path: str | Path) -> tuple[np.ndarray, float]:
    """Load any audio file as mono 16kHz float32 via ffmpeg.

    Returns (samples, sample_rate). This is the canonical input format for
    faster-whisper's transcribe() and most ASR frontends.
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "f32le",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    samples = np.frombuffer(result.stdout, dtype=np.float32)
    return samples, 16000.0


def get_duration(path: str | Path) -> float:
    """Get audio duration in seconds via ffprobe (used for progress reporting)."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def separate_vocals(path: str | Path, output_dir: str | Path, device: str = "cuda") -> Path:
    """Separate vocals from a song using Demucs (htdemucs model).

    Runs demucs as a subprocess to keep its heavy model loading isolated
    from the main process. Returns the path to the isolated vocals WAV.

    Connects to: asr.py — the returned vocals file is fed to transcription
    instead of the full mix, removing drums/bass/instrumental bleed that
    otherwise corrupts lyric hypotheses.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        "htdemucs",
        "--two-stems",
        "vocals",
        "-d",
        device,
        "-o",
        str(output_dir),
        str(path),
    ]
    logger.info("Running demucs vocal separation: %s", " ".join(cmd))
    with timed_operation_with_stats("demucs separation"):
        subprocess.run(cmd, check=True, capture_output=True)

    stem_dir = output_dir / "htdemucs" / Path(path).stem
    vocals = stem_dir / "vocals.wav"
    if not vocals.exists():
        msg = f"Demucs did not produce expected vocals file: {vocals}"
        raise FileNotFoundError(msg)
    return vocals
