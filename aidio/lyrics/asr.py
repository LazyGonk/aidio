"""Transcription backends.

Connects to: audio.py (consumes audio files) and lyrics.py (produces
segment lists that get merged into lyric lines).

Two backends are provided:
- LocalWhisperBackend: faster-whisper running on the local NVIDIA GPU.
  large-v3-turbo is fast and accurate; running locally avoids the remote
  endpoint's short effective context window, so no chunking is needed.
- RemoteWhisperBackend: OpenAI-compatible Whisper API (the endpoint from
  the handoff doc), with 30s chunking to work around its context limit.

Both return a list of segment dicts: {"start": float, "end": float, "text": str}.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from aidio.lyrics.cuda_dlls import (
    cuda_library_help,
    enable_cuda_dlls,
    is_cuda_library_error,
)
from aidio.lyrics.device import whisper_args
from aidio.lyrics.stats import timed_operation_with_stats

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """A single transcribed segment with global timestamps."""

    start: float
    end: float
    text: str


ProgressFn = Callable[[float], None]


class TranscribeBackend(Protocol):
    """Protocol all backends implement — enables easy swapping/testing."""

    def transcribe(
        self, audio_path: str | Path, progress: ProgressFn | None = None
    ) -> list[Segment]: ...


class LocalWhisperBackend:
    """faster-whisper on local GPU (CUDA). No chunking needed — the model
    handles full songs natively, which is the main quality win over the
    remote endpoint described in HANDOFF.md."""

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str | None = None,
        language: str | None = "en",
    ) -> None:
        # Register torch/nvidia DLL dirs before ctranslate2 loads cuBLAS/cuDNN.
        enable_cuda_dlls()
        from faster_whisper import WhisperModel  # deferred: heavy import

        resolved = whisper_args(device)
        self.device = resolved["device"]
        self.compute_type = compute_type or resolved["compute_type"]
        self.language = language
        try:
            self.model = WhisperModel(
                model_size, device=self.device, compute_type=self.compute_type
            )
        except Exception as e:
            if self.device == "cuda" and is_cuda_library_error(e):
                raise RuntimeError(cuda_library_help()) from e
            raise

    def transcribe(
        self, audio_path: str | Path, progress: ProgressFn | None = None
    ) -> list[Segment]:
        """Transcribe a full audio file, returning timestamped segments."""
        logger.info("Transcribing %s with local faster-whisper", audio_path)
        with timed_operation_with_stats("whisper transcription"):
            segments_iter, info = self.model.transcribe(
                str(audio_path),
                language=self.language,
                beam_size=5,
                condition_on_previous_text=False,
            )
            duration = float(getattr(info, "duration", 0) or 0)
            segments: list[Segment] = []
            for seg in segments_iter:
                if not seg.text.strip():
                    continue
                segments.append(Segment(start=seg.start, end=seg.end, text=seg.text.strip()))
                if progress is not None and duration > 0:
                    progress(min(max(seg.end / duration, 0.0), 0.99))
        if progress is not None:
            progress(1.0)
        logger.info("Got %d segments (duration %.1fs)", len(segments), info.duration)
        return segments


class RemoteWhisperBackend:
    """OpenAI-compatible Whisper API with 30s chunking.

    The remote endpoint's large-v3-turbo truncates on long input, so we
    split via ffmpeg and stitch segments using global timestamps
    (chunk_start + seg.start), same approach as the original script.
    """

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        chunk_duration: int = 30,
        language: str | None = "en",
    ) -> None:
        from openai import OpenAI  # deferred: only needed for remote backend

        self.client = OpenAI(
            base_url=url or os.environ.get("WHISPER_URL", "http://localhost:2389/v1"),
            api_key=api_key or os.environ.get("WHISPER_API_KEY", ""),
        )
        self.model = model or os.environ.get("WHISPER_MODEL", "large-v3-turbo")
        self.chunk_duration = chunk_duration
        self.language = language

    def transcribe(
        self, audio_path: str | Path, progress: ProgressFn | None = None
    ) -> list[Segment]:
        """Split audio into chunks, transcribe each, stitch by global time."""
        logger.info("Transcribing %s with remote Whisper API", audio_path)
        with timed_operation_with_stats("whisper transcription (remote)"):
            duration = self._get_duration(audio_path)
            all_segments: list[Segment] = []
            with tempfile.TemporaryDirectory() as tmp:
                start = 0.0
                idx = 0
                while start < duration:
                    chunk_path = Path(tmp) / f"chunk_{idx:03d}.mp3"
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-v",
                            "error",
                            "-ss",
                            str(start),
                            "-t",
                            str(self.chunk_duration),
                            "-i",
                            str(audio_path),
                            "-vn",
                            "-acodec",
                            "libmp3lame",
                            "-q:a",
                            "2",
                            str(chunk_path),
                        ],
                        check=True,
                        capture_output=True,
                    )
                    all_segments.extend(self._transcribe_chunk(chunk_path, start))
                    start += self.chunk_duration
                    idx += 1
                    if progress is not None and duration > 0:
                        progress(min(start / duration, 0.99))
        if progress is not None:
            progress(1.0)
        return sorted(all_segments, key=lambda s: s.start)

    def _transcribe_chunk(self, chunk_path: Path, chunk_start: float) -> list[Segment]:
        """Transcribe one chunk and offset its timestamps to global time."""
        with chunk_path.open("rb") as f:
            response = self.client.audio.transcriptions.create(
                model=self.model,
                file=f,
                response_format="verbose_json",
                language=self.language,
                temperature=0.0,
                timestamp_granularities=["segment"],
            )
        return [
            Segment(
                start=chunk_start + seg.start,
                end=chunk_start + seg.end,
                text=seg.text.strip(),
            )
            for seg in response.segments
            if seg.text.strip()
        ]

    @staticmethod
    def _get_duration(path: str | Path) -> float:
        """Duration via ffprobe (same as audio.get_duration, kept local to
        avoid importing torch-dependent audio module for remote-only use)."""
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


def get_backend(name: str, **kwargs) -> TranscribeBackend:
    """Factory: create a backend by name ('local' or 'remote').

    Connects to: cli.py — the CLI maps --backend to this factory.
    """
    if name == "local":
        return LocalWhisperBackend(**kwargs)
    if name == "remote":
        return RemoteWhisperBackend(**kwargs)
    msg = f"Unknown backend: {name!r} (expected 'local' or 'remote')"
    raise ValueError(msg)
