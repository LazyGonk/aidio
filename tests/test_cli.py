"""Tests for the CLI entry point."""

from unittest.mock import MagicMock, patch

from aidio.cli import main


class TestCLI:
    """Test the command-line interface (analyze subcommand)."""

    def test_main_analyze_valid_args(self):
        with patch("aidio.cli._run_analyze") as mock_run:
            mock_run.return_value = 0
            result = main(["analyze", "test.mp3"])
            assert result == 0
            mock_run.assert_called_once()

    def test_main_with_custom_prompt(self):
        with patch("aidio.analyzer.analyze_audio") as mock_analyze:
            mock_analyze.return_value = MagicMock(
                chunk_count=1,
                total_duration_seconds=1.0,
                full_text="Result",
            )

            result = main(["analyze", "test.mp3", "--prompt", "Custom prompt"])

            assert result == 0
            call_kwargs = mock_analyze.call_args[1]
            assert call_kwargs["prompt"] == "Custom prompt"

    def test_main_with_override_args(self):
        with patch("aidio.analyzer.analyze_audio") as mock_analyze:
            mock_analyze.return_value = MagicMock(
                chunk_count=1,
                total_duration_seconds=1.0,
                full_text="Result",
            )

            result = main(["analyze", "test.mp3", "--sr", "8000", "--chunk", "10.0"])

            assert result == 0
            call_kwargs = mock_analyze.call_args[1]
            assert call_kwargs["target_sr"] == 8000
            assert call_kwargs["max_chunk_seconds"] == 10.0

    def test_main_passes_model_override(self):
        with patch("aidio.analyzer.analyze_audio") as mock_analyze:
            mock_analyze.return_value = MagicMock(
                chunk_count=1,
                total_duration_seconds=1.0,
                full_text="Result",
            )

            result = main(["analyze", "test.mp3", "--model", "custom-model"])

            assert result == 0
            assert mock_analyze.call_args[1]["model"] == "custom-model"
