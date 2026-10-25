"""Command-line entry point for Aidio.

Usage::

    python -m aidio.cli analyze path/to/audio.mp3 [--prompt "Custom prompt"]
    python -m aidio.cli transcribe path/to/song.mp3 [--device cpu] [--llm-correct]

The LLM is always optional: analyze uses it for audio characteristics,
transcribe uses it only for post-hoc lyrics correction (--llm-correct).
"""

from __future__ import annotations

import argparse
import sys

from . import __copyright__, __license__, __version__, banner
from .analyzer import DEFAULT_PROMPT


def _add_analyze_sub(subparsers) -> None:
    p = subparsers.add_parser("analyze", help="LLM analysis of audio characteristics")
    p.add_argument("audio", help="Path to the audio file to analyze")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="Custom analysis prompt")
    p.add_argument("--model", default=None, help="Override model name")
    p.add_argument("--sr", type=int, default=None, help="Override sample rate")
    p.add_argument("--chunk", type=float, default=None, help="Override chunk duration (s)")


def _add_transcribe_sub(subparsers) -> None:
    p = subparsers.add_parser("transcribe", help="Whisper transcription of lyrics")
    p.add_argument("audio", help="Path to the audio file")
    p.add_argument(
        "--device",
        choices=["cuda", "cpu", "rocm"],
        default=None,
        help="Compute device (default: AIDIO_DEVICE env or cuda). ROCm is Linux-only.",
    )
    p.add_argument("--model", default="large-v3-turbo", help="Whisper model name")
    p.add_argument(
        "--backend",
        choices=["local", "remote"],
        default="local",
        help="local = faster-whisper on device, remote = Whisper API",
    )
    p.add_argument("--separate-vocals", action="store_true", help="Demucs vocal pre-pass")
    p.add_argument("--timestamps", action="store_true", help="LRC-style timestamps")
    p.add_argument("--embed", action="store_true", help="Embed USLT tag into the file")
    p.add_argument("--llm-correct", action="store_true", help="Optional LLM correction")
    p.add_argument("--endpoint", default=None, help="LLM endpoint for correction")
    p.add_argument("--model-name", default=None, help="LLM model for correction")
    p.add_argument("--out", default=None, help="Write lyrics to this .txt path")


def main(argv: list[str] | None = None) -> int:
    """Parse CLI args and dispatch to the analysis or transcription pipeline."""
    parser = argparse.ArgumentParser(
        prog="aidio",
        description="Aidio - Audio Analysis & Lyrics Transcription",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__} ({__copyright__}, {__license__} License)",
    )
    subparsers = parser.add_subparsers(dest="command")
    _add_analyze_sub(subparsers)
    _add_transcribe_sub(subparsers)

    args = parser.parse_args(argv)

    if args.command == "transcribe":
        return _run_transcribe(args)
    if args.command == "analyze":
        return _run_analyze(args)

    print(banner())
    print()
    parser.print_help()
    return 0


def _run_analyze(args) -> int:
    from .analyzer import analyze_audio as _analyze

    result = _analyze(
        args.audio,
        prompt=args.prompt,
        model=args.model,
        target_sr=args.sr,
        max_chunk_seconds=args.chunk,
    )
    print(f"Analyzed {result.chunk_count} chunk(s), {result.total_duration_seconds:.1f}s total\n")
    print(result.full_text)
    return 0


def _run_transcribe(args) -> int:
    from aidio.pipeline import transcribe_lyrics

    result = transcribe_lyrics(
        args.audio,
        device=args.device,
        model=args.model,
        backend=args.backend,
        separate_vocals=args.separate_vocals,
        timestamps=args.timestamps,
        llm_correct=args.llm_correct,
        endpoint=args.endpoint,
        model_name=args.model_name,
        embed_into=args.audio if args.embed else None,
    )
    if args.out:
        from pathlib import Path

        Path(args.out).write_text(result.lyrics, encoding="utf-8")
        print(f"Wrote lyrics to {args.out}")
    else:
        print(result.lyrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
