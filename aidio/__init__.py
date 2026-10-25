"""Aidio — audio analysis and lyrics transcription.

A modular Python package that sends audio to an OpenAI-compatible LLM for
analysis (instruments, genre, vocals, etc.) and transcribes lyrics with
offline Whisper (faster-whisper on CUDA/CPU/ROCm) plus optional Demucs
vocal separation.
"""

__version__ = "0.2.0"
__copyright__ = "Copyright (c) 2026 LazyGonk"
__license__ = "MIT"


def banner() -> str:
    """One-line version / copyright greeting for CLIs and server startup."""
    return f"Aidio v{__version__} — {__copyright__} — {__license__} License"
