"""Tests for the configuration module."""

import os
import tempfile
from unittest.mock import patch

from aidio.config import Settings, get_settings


class TestSettings:
    """Test the Settings dataclass."""

    def test_defaults(self):
        """Test default values when no env vars are set."""
        # Clear all relevant env vars
        env_vars_to_clear = [
            "OPENAI_BASE_URL",
            "OPENAI_API_KEY",
            "ANALYSIS_MODEL",
            "MODEL_SAMPLE_RATE",
            "MAX_CHUNK_SECONDS",
        ]

        with patch.dict(os.environ, {}, clear=True):
            # Remove the vars we want to test defaults for
            for var in env_vars_to_clear:
                os.environ.pop(var, None)

            settings = Settings()

            assert settings.base_url == "http://127.0.0.1:4000/v1"
            assert settings.api_key == "sk-not-needed-for-local"
            assert settings.model == "gemma4:12b"
            assert settings.sample_rate == 16000
            assert settings.max_chunk_seconds == 2400

    def test_custom_values(self):
        """Test custom values from environment variables."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_BASE_URL": "http://custom:4000/v1",
                "OPENAI_API_KEY": "custom-key",
                "ANALYSIS_MODEL": "custom-model",
                "MODEL_SAMPLE_RATE": "8000",
                "MAX_CHUNK_SECONDS": "1200",
            },
        ):
            settings = Settings()

            assert settings.base_url == "http://custom:4000/v1"
            assert settings.api_key == "custom-key"
            assert settings.model == "custom-model"
            assert settings.sample_rate == 8000
            assert settings.max_chunk_seconds == 1200

    def test_settings_are_dynamic(self):
        """Test that Settings loads values dynamically."""
        settings1 = Settings()
        settings2 = Settings()

        # Both should have the same values (from current env)
        assert settings1.base_url == settings2.base_url
        assert settings1.api_key == settings2.api_key


class TestGetSettings:
    """Test the get_settings singleton."""

    def test_returns_singleton(self):
        """Test that get_settings returns the same instance."""
        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_creates_new_if_none(self):
        """Test that get_settings creates a new instance if none exists."""
        # Reset the singleton
        import aidio.config

        aidio.config._settings = None
        aidio.config._env_loaded = False

        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_loads_from_env_file(self):
        """Test loading settings from a specific .env file."""
        # Create a temporary .env file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("OPENAI_BASE_URL=http://test-env:4000/v1\n")
            f.write("OPENAI_API_KEY=test-env-key\n")
            env_file = f.name

        try:
            # Reset singleton to force reload
            import aidio.config

            aidio.config._settings = None
            aidio.config._env_loaded = False

            settings = get_settings(env_file=env_file)

            assert settings.base_url == "http://test-env:4000/v1"
            assert settings.api_key == "test-env-key"
        finally:
            os.unlink(env_file)
