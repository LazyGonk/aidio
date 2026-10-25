"""Tests for lyrics output (txt + metadata embedding)."""

import numpy as np
import pytest
import soundfile as sf

from aidio.lyrics.output import embed_lyrics, save_lyrics_txt


def test_save_lyrics_txt_uses_audio_stem(tmp_path):
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"")
    txt = save_lyrics_txt("line one\nline two\n", audio)
    assert txt == tmp_path / "song.txt"
    assert txt.read_text(encoding="utf-8") == "line one\nline two\n"


def test_embed_lyrics_flac(tmp_path):
    pytest.importorskip("mutagen")
    from mutagen.flac import FLAC

    audio_path = tmp_path / "song.flac"
    sf.write(audio_path, np.zeros(1600, dtype=np.float32), 16000)
    embed_lyrics("line one\nline two", audio_path)

    tags = FLAC(str(audio_path))
    assert tags["LYRICS"] == ["line one\nline two"]
    assert tags["UNSYNCEDLYRICS"] == ["line one\nline two"]


def test_embed_lyrics_rejects_unsupported_format(tmp_path):
    audio_path = tmp_path / "song.wav"
    audio_path.write_bytes(b"")
    with pytest.raises(ValueError, match="not supported"):
        embed_lyrics("line", audio_path)
