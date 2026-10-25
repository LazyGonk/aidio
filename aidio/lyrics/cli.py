"""CLI entry point for Aidio's lyrics extraction subpackage.

Connects everything via aidio.pipeline.transcribe_lyrics(): audio.py
(optional demucs) -> asr.py (backend choice) -> lyrics.py (line building)
-> output.py (txt + ID3v2) -> quality.py (optional reference comparison).

Usage examples:
    python -m aidio.lyrics.cli --song "song.mp3" --reference ref.txt
    python -m aidio.lyrics.cli                    # all MP3s in cwd
    python -m aidio.lyrics.cli --backend remote   # Whisper API endpoint
    python -m aidio.lyrics.cli --separate-vocals  # demucs pre-pass
    python -m aidio.lyrics.cli --timestamps       # LRC-style [MM:SS] lines
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from aidio.lyrics.output import save_lyrics_txt
from aidio.lyrics.quality import compare_with_reference
from aidio.lyrics.stats import get_stats, print_summary
from aidio.pipeline import LyricsResult, transcribe_lyrics

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="lyrics-extractor",
        description="Extract lyrics from MP3s via local/remote Whisper, "
        "save as .txt and embed as ID3v2 USLT tags.",
    )
    parser.add_argument("--song", help="Single MP3 file (default: all MP3s in --dir)")
    parser.add_argument("--dir", default=".", help="Directory of MP3s (default: cwd)")
    parser.add_argument(
        "--backend",
        choices=["local", "remote"],
        default="local",
        help="local = faster-whisper on device, remote = Whisper API",
    )
    parser.add_argument("--model", default="large-v3-turbo", help="Whisper model name")
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "rocm"],
        default=None,
        help="Compute device (default: AIDIO_DEVICE env or cuda). ROCm is Linux-only.",
    )
    parser.add_argument("--reference", help="Reference lyrics txt for quality report")
    parser.add_argument(
        "--separate-vocals",
        action="store_true",
        help="Run demucs vocal isolation before transcription",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Prefix lines with LRC-style [MM:SS.xx] timestamps",
    )
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip embedding lyrics into the MP3 ID3v2 tag",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Run LLM-based analysis of lyrics (requires OpenAI-compatible endpoint)",
    )
    parser.add_argument(
        "--explain-endpoint",
        default=None,
        help="OpenAI-compatible API base URL for explanation (default: env "
        "OPENAI_BASE_URL or http://localhost:1234/v1)",
    )
    parser.add_argument(
        "--explain-model",
        default=None,
        help="Model name for explanation (default: env OPENAI_MODEL or gemma-3-12b)",
    )
    parser.add_argument(
        "--llm-correct",
        action="store_true",
        help="Run LLM post-hoc correction of the transcription (optional)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def process_song(mp3, args: argparse.Namespace) -> LyricsResult:
    """Run the full pipeline for one MP3 and return the transcription result.

    Delegates to aidio.pipeline.transcribe_lyrics(); the CLI only handles
    txt saving.
    """
    result = transcribe_lyrics(
        mp3,
        device=args.device,
        model=args.model,
        backend=args.backend,
        separate_vocals=args.separate_vocals,
        timestamps=args.timestamps,
        llm_correct=args.llm_correct,
        endpoint=args.explain_endpoint,
        model_name=args.explain_model,
        embed_into=None if args.no_embed else mp3,
    )
    save_lyrics_txt(result.lyrics, mp3)
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: orchestrate backend creation and per-song processing."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    mp3s = [Path(args.song)] if args.song else sorted(Path(args.dir).glob("*.mp3"))
    if not mp3s:
        logger.error("No MP3 files found")
        return 1

    reference = Path(args.reference).read_text(encoding="utf-8") if args.reference else None
    failures = 0
    all_stats: dict[str, dict] = {}
    for mp3 in mp3s:
        logger.info("=== %s ===", mp3.name)
        try:
            result = process_song(mp3, args)
        except Exception:
            logger.exception("Failed to process %s", mp3.name)
            failures += 1
            continue
        if reference and mp3.stem.endswith("By_Blade__By_Bone"):
            report = compare_with_reference(result.lyrics, reference)
            print(f"Quality for {mp3.name}: {report.summary()}")
            if report.line_recall < 0.9:
                logger.warning("Below 90%% line-recall target")
        if args.explain:
            from aidio.lyrics.explain import explain_lyrics  # deferred: heavy

            explain = explain_lyrics(
                result.lyrics,
                endpoint=args.explain_endpoint,
                model=args.explain_model,
            )
            print(explain.summary())
        # Merge per-song stats into the summary
        song_stats = get_stats()
        for name, data in song_stats.items():
            all_stats[name] = {
                "duration": all_stats.get(name, {}).get("duration", 0) + data["duration"],
                "peak_gpu_mb": max(
                    all_stats.get(name, {}).get("peak_gpu_mb", 0),
                    data["peak_gpu_mb"],
                ),
            }
    if all_stats:
        print_summary(all_stats)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
