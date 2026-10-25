"""Configuration loader.

Reads settings from ``.env`` (via python-dotenv) and exposes them as
typed constants.  All other modules import from here so a single
``.env`` file controls the whole engine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _load_env(env_file: str | None = None) -> None:
    """Load environment variables from .env file.

    Parameters
    ----------
    env_file : str, optional
        Path to .env file. If None, uses default search path.
    """
    if env_file:
        load_dotenv(env_file, override=True)
    else:
        load_dotenv(override=True)


@dataclass
class Settings:
    """All tunable settings for the engine."""

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    sample_rate: int = 0
    max_chunk_seconds: int = 0

    def __post_init__(self) -> None:
        """Load values from environment at instance creation time."""
        self.base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:4000/v1")
        self.api_key = os.getenv("OPENAI_API_KEY", "sk-not-needed-for-local")
        self.model = os.getenv("ANALYSIS_MODEL", "gemma4:12b")
        self.sample_rate = int(os.getenv("MODEL_SAMPLE_RATE", "16000"))
        self.max_chunk_seconds = int(os.getenv("MAX_CHUNK_SECONDS", "2400"))


_settings: Settings | None = None
_env_loaded: bool = False


def get_settings(env_file: str | None = None) -> Settings:
    """Return the singleton ``Settings`` instance.

    Parameters
    ----------
    env_file : str, optional
        Path to .env file. If None, uses default search path.
    """
    global _settings, _env_loaded

    if not _env_loaded:
        _load_env(env_file)
        _env_loaded = True

    if _settings is None:
        _settings = Settings()
    return _settings
