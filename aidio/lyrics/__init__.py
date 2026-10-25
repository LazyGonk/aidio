"""Lyrics extraction package.

Modular pipeline for extracting song lyrics from MP3 files:
- audio: audio loading / vocal separation (demucs)
- asr: transcription backends (local faster-whisper GPU, remote Whisper API)
- lyrics: text post-processing (line merging, deduplication)
- output: txt writing and ID3v2 USLT embedding
- quality: comparison against reference lyrics
"""

from aidio.lyrics.asr import get_backend
from aidio.lyrics.quality import compare_with_reference

__all__ = [
    "compare_with_reference",
    "get_backend",
]
