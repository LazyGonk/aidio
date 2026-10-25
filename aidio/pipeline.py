"""Shared lyrics-transcription pipeline orchestration.

Connects to: aidio/lyrics/* (backend, demucs, line building, correction,
output) and every entry point — aidio/cli.py (transcribe subcommand),
aidio/lyrics/cli.py, and aidio/web.py. No business logic lives in entry
points; they all call transcribe_lyrics().
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from aidio.lyrics.asr import Segment, TranscribeBackend, get_backend
from aidio.lyrics.audio import separate_vocals as _separate_vocals
from aidio.lyrics.correct import correct_lyrics
from aidio.lyrics.cuda_dlls import enable_cuda_dlls
from aidio.lyrics.device import resolve_device
from aidio.lyrics.lyrics import build_lyrics_text

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


@dataclass
class LyricsResult:
    """Result of a transcription run.

    Attributes:
        lyrics: Final lyric lines (possibly LLM-corrected).
        segments: Raw timestamped Whisper segments.
        device: Validated compute device used ("cuda" | "cpu" | "rocm").
        used_demucs: Whether vocal separation ran.
        corrections: LLM-corrected text if llm_correct was enabled, else None.
    """

    lyrics: str
    segments: list[Segment] = field(default_factory=list)
    device: str = "cuda"
    used_demucs: bool = False
    corrections: str | None = None


def _make_backend(
    backend_name: str, model: str, device: str, language: str | None = "en"
) -> TranscribeBackend:
    """Factory returning the transcription backend for a pipeline run.

    Only the local backend takes a device; the remote one ignores it.
    """
    if backend_name == "remote":
        return get_backend("remote", model=model, language=language)
    return get_backend(backend_name, model_size=model, device=device, language=language)


def transcribe_lyrics(
    audio_path: str | Path,
    *,
    device: str | None = None,
    model: str = "large-v3-turbo",
    backend: str = "local",
    separate_vocals: bool = False,
    timestamps: bool = False,
    llm_correct: bool = False,
    correction_context: str | None = None,
    endpoint: str | None = None,
    model_name: str | None = None,
    embed_into: str | Path | None = None,
    language: str | None = "en",
    progress: ProgressCallback | None = None,
) -> LyricsResult:
    """Full transcription pipeline for one audio file.

    Orchestrates: (optional demucs) -> Whisper -> line building ->
    (optional LLM correction) -> (optional ID3 embedding).

    Args:
        audio_path: Audio file to transcribe.
        device: Compute device override; falls back to AIDIO_DEVICE env,
            then "cuda". Validated before the pipeline starts.
        model: Whisper model name (local backend).
        backend: "local" (faster-whisper) or "remote" (Whisper API).
        separate_vocals: Run demucs vocal isolation pre-pass.
        timestamps: Prefix lines with LRC-style timestamps.
        llm_correct: Enable LLM post-hoc correction of the transcription.
        correction_context: Optional characteristics summary used by the
            LLM correction step only.
        endpoint: LLM endpoint for correction (OpenAI-compatible).
        model_name: LLM model name for correction.
        embed_into: If given, embed final lyrics into this file (usually
            the original MP3).
        language: Whisper language code, or None for auto-detection.
        progress: Optional callback receiving (fraction, stage label).

    Returns:
        LyricsResult with final text, segments, device, and correction info.
    """
    resolved = resolve_device(device)
    audio_path = Path(audio_path)
    enable_cuda_dlls()

    def report(fraction: float, stage: str) -> None:
        if progress is not None:
            progress(max(0.0, min(1.0, fraction)), stage)

    report(0.01, "Preparing audio")

    audio_source = audio_path
    used_demucs = False
    if separate_vocals and backend == "local":
        report(0.03, "Separating vocals (Demucs)")
        audio_source = _separate_vocals(
            audio_path,
            output_dir=audio_path.parent / "_stems",
            device="cuda" if resolved in ("cuda", "rocm") else "cpu",
        )
        used_demucs = True

    report(0.12, "Loading transcription model")
    transcription_backend = _make_backend(backend, model=model, device=resolved, language=language)

    def asr_progress(fraction: float) -> None:
        report(0.15 + 0.75 * fraction, "Transcribing lyrics")

    segments = transcription_backend.transcribe(audio_source, progress=asr_progress)
    report(0.92, "Building lyric lines")
    lyrics = build_lyrics_text(segments, timestamps=timestamps)

    corrected: str | None = None
    if llm_correct:
        report(0.94, "Polishing lyrics with LLM")
        corrected = correct_lyrics(
            lyrics, context=correction_context, endpoint=endpoint, model=model_name
        )
        lyrics = corrected

    if embed_into is not None:
        report(0.98, "Embedding lyrics")
        from aidio.lyrics.output import embed_lyrics  # deferred: mutagen

        embed_lyrics(lyrics, embed_into)

    report(1.0, "Done")
    return LyricsResult(
        lyrics=lyrics,
        segments=segments,
        device=resolved,
        used_demucs=used_demucs,
        corrections=corrected,
    )
