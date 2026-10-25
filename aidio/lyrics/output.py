"""Output: writing .txt files and embedding ID3v2 lyrics tags.

Connects to: lyrics.py (consumes the final lyrics text) and cli.py.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _mutagen():
    """Import mutagen lazily (optional extra; GPL-licensed, keep it optional)."""
    try:
        import mutagen
    except ImportError as e:
        msg = (
            "mutagen is required for saving/embedding lyrics — "
            'install it with: pip install -e ".[embed]"'
        )
        raise ImportError(msg) from e
    return mutagen


def save_lyrics_txt(lyrics: str, mp3_path: str | Path) -> Path:
    """Write lyrics next to the MP3 as <stem>.txt. Returns the txt path."""
    start = time.perf_counter()
    txt_path = Path(mp3_path).with_suffix(".txt")
    txt_path.write_text(lyrics, encoding="utf-8", newline="\n")
    logger.info("Wrote %s (%.4fs)", txt_path, time.perf_counter() - start)
    return txt_path


def embed_lyrics_id3(lyrics: str, mp3_path: str | Path) -> None:
    """Embed lyrics as an ID3v2 USLT frame (UTF-8, lang 'eng') in the MP3.

    Creates an ID3 header if the file has none. Overwrites any existing
    'eng' USLT frame so re-runs are idempotent.
    """
    mutagen = _mutagen()
    ID3, USLT, Encoding, ID3NoHeaderError = (
        mutagen.id3.ID3,
        mutagen.id3.USLT,
        mutagen.id3.Encoding,
        mutagen.id3.ID3NoHeaderError,
    )
    start = time.perf_counter()
    try:
        tags = ID3(str(mp3_path))
    except ID3NoHeaderError:
        tags = ID3()
    # Remove existing english unsynchronized lyrics to avoid duplicates.
    tags.delall("USLT")
    tags.add(USLT(encoding=Encoding.UTF8, lang="eng", desc="", text=lyrics))
    tags.save(str(mp3_path))
    logger.info("Embedded USLT lyrics into %s (%.4fs)", mp3_path, time.perf_counter() - start)


def embed_lyrics_flac(lyrics: str, flac_path: str | Path) -> None:
    """Embed lyrics as Vorbis comment tags in a FLAC file."""
    _mutagen()
    from mutagen.flac import FLAC

    start = time.perf_counter()
    audio = FLAC(str(flac_path))
    audio["LYRICS"] = lyrics
    audio["UNSYNCEDLYRICS"] = lyrics
    audio.save()
    logger.info("Embedded lyrics into %s (%.4fs)", flac_path, time.perf_counter() - start)


def embed_lyrics_m4a(lyrics: str, m4a_path: str | Path) -> None:
    """Embed lyrics as the iTunes/MP4 ``©lyr`` tag in an M4A/MP4 file."""
    _mutagen()
    from mutagen.mp4 import MP4

    start = time.perf_counter()
    audio = MP4(str(m4a_path))
    audio["\xa9lyr"] = lyrics
    audio.save()
    logger.info("Embedded lyrics into %s (%.4fs)", m4a_path, time.perf_counter() - start)


SUPPORTED_EMBED_SUFFIXES = (".mp3", ".flac", ".m4a", ".mp4")


def embed_lyrics(lyrics: str, audio_path: str | Path) -> None:
    """Embed lyrics into a supported audio container.

    Raises:
        ValueError: If the container format cannot carry lyrics tags here.
    """
    suffix = Path(audio_path).suffix.lower()
    if suffix == ".mp3":
        embed_lyrics_id3(lyrics, audio_path)
    elif suffix == ".flac":
        embed_lyrics_flac(lyrics, audio_path)
    elif suffix in (".m4a", ".mp4"):
        embed_lyrics_m4a(lyrics, audio_path)
    else:
        supported = ", ".join(SUPPORTED_EMBED_SUFFIXES)
        msg = (
            f"Embedding lyrics is not supported for {suffix or 'this file'}. "
            f"Supported formats: {supported}."
        )
        raise ValueError(msg)
