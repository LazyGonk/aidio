"""Tests for the analyzer orchestration module."""

from unittest.mock import MagicMock, patch

from aidio.analyzer import AnalysisResult, analyze_audio
from aidio.audio import AudioChunk


class TestAnalyzeAudio:
    """Test the analyzer orchestration."""

    def test_analyzes_single_chunk(self):
        # Mock preprocess to return one chunk
        mock_chunk = AudioChunk(
            base64_wav="dGVzdA==",
            duration_seconds=1.0,
        )

        # Mock client to return analysis
        mock_client = MagicMock()
        mock_client.analyze.return_value = "Instrument: guitar, Genre: rock"

        with patch("aidio.analyzer.preprocess_audio") as mock_preprocess:
            mock_preprocess.return_value = [mock_chunk]

            result = analyze_audio("test.mp3", client=mock_client)

            assert isinstance(result, AnalysisResult)
            assert result.chunk_count == 1
            assert result.total_duration_seconds == 1.0
            assert "Instrument: guitar" in result.full_text

    def test_analyzes_multiple_chunks(self):
        mock_chunks = [
            AudioChunk(base64_wav="chunk1", duration_seconds=2.0),
            AudioChunk(base64_wav="chunk2", duration_seconds=2.0),
        ]

        mock_client = MagicMock()
        mock_client.analyze.side_effect = [
            "Chunk 1 analysis",
            "Chunk 2 analysis",
        ]

        with patch("aidio.analyzer.preprocess_audio") as mock_preprocess:
            mock_preprocess.return_value = mock_chunks

            result = analyze_audio("long.mp3", client=mock_client)

            assert result.chunk_count == 2
            assert result.total_duration_seconds == 4.0
            assert "Chunk 1 analysis" in result.full_text
            assert "Chunk 2 analysis" in result.full_text
            assert "--- Chunk 1/2" in result.full_text
            assert "--- Chunk 2/2" in result.full_text

    def test_creates_client_if_not_provided(self):
        mock_chunk = AudioChunk(base64_wav="test", duration_seconds=1.0)

        mock_client = MagicMock()
        mock_client.analyze.return_value = "Analysis"

        with (
            patch("aidio.analyzer.preprocess_audio") as mock_preprocess,
            patch("aidio.analyzer.AnalysisClient") as mock_client_cls,
        ):
            mock_preprocess.return_value = [mock_chunk]
            mock_client_cls.return_value = mock_client

            result = analyze_audio("test.mp3")

            mock_client_cls.assert_called_once()
            assert "Analysis" in result.full_text

    def test_uses_custom_prompt(self):
        mock_chunk = AudioChunk(base64_wav="test", duration_seconds=1.0)

        mock_client = MagicMock()
        mock_client.analyze.return_value = "Result"

        custom_prompt = "Custom prompt for testing"

        with (
            patch("aidio.analyzer.preprocess_audio") as mock_preprocess,
            patch("aidio.analyzer.AnalysisClient") as mock_client_cls,
        ):
            mock_preprocess.return_value = [mock_chunk]
            mock_client_cls.return_value = mock_client

            analyze_audio("test.mp3", prompt=custom_prompt, client=mock_client)

            # Verify the prompt was passed to the client
            call_args = mock_client.analyze.call_args
            assert call_args[0][0] == custom_prompt

    def test_reads_preprocessing_defaults_from_settings(self):
        mock_chunk = AudioChunk(base64_wav="test", duration_seconds=1.0)
        mock_client = MagicMock()
        mock_client.analyze.return_value = "Analysis"
        settings = MagicMock(sample_rate=8000, max_chunk_seconds=1200)

        with (
            patch("aidio.analyzer.preprocess_audio") as mock_preprocess,
            patch("aidio.analyzer.get_settings", return_value=settings),
        ):
            mock_preprocess.return_value = [mock_chunk]

            analyze_audio("test.mp3", client=mock_client)

            mock_preprocess.assert_called_once_with(
                "test.mp3", target_sr=8000, max_chunk_seconds=1200
            )

    def test_passes_model_to_created_client(self):
        mock_chunk = AudioChunk(base64_wav="test", duration_seconds=1.0)
        settings = MagicMock(sample_rate=16000, max_chunk_seconds=2400)

        with (
            patch("aidio.analyzer.preprocess_audio") as mock_preprocess,
            patch("aidio.analyzer.get_settings", return_value=settings),
            patch("aidio.analyzer.AnalysisClient") as mock_client_cls,
        ):
            mock_preprocess.return_value = [mock_chunk]
            mock_client_cls.return_value.analyze.return_value = "Analysis"

            analyze_audio("test.mp3", model="custom-model")

            mock_client_cls.assert_called_once_with(model="custom-model")
