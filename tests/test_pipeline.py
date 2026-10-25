"""Tests for the shared transcription pipeline and LLM correction."""

from unittest.mock import MagicMock, patch

import pytest

from aidio.lyrics.asr import Segment
from aidio.lyrics.correct import CORRECT_SYSTEM_PROMPT, correct_lyrics
from aidio.lyrics.device import DeviceError
from aidio.pipeline import LyricsResult, transcribe_lyrics


class _FakeBackend:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def transcribe(self, audio_path, progress=None):
        if progress is not None:
            progress(0.5)
        return [
            Segment(start=0.0, end=2.0, text="line one"),
            Segment(start=3.0, end=5.0, text="line two"),
        ]


@pytest.fixture()
def fake_backend(monkeypatch):
    saved: dict = {}
    created: list = []

    def factory(name, **kwargs):
        created.append((name, kwargs))
        return _FakeBackend(**kwargs)

    monkeypatch.setattr("aidio.pipeline._make_backend", factory)
    saved["created"] = created
    monkeypatch.delenv("AIDIO_DEVICE", raising=False)
    return saved


class TestTranscribeLyrics:
    def test_returns_result_with_lyrics(self, fake_backend):
        result = transcribe_lyrics("song.mp3")
        assert isinstance(result, LyricsResult)
        assert result.lyrics
        assert len(result.segments) == 2

    def test_device_validated_before_pipeline(self, fake_backend):
        with pytest.raises(DeviceError):
            transcribe_lyrics("song.mp3", device="vulkan")

    def test_env_device_used_when_no_arg(self, fake_backend, monkeypatch):
        monkeypatch.setenv("AIDIO_DEVICE", "cpu")
        result = transcribe_lyrics("song.mp3")
        assert result.device == "cpu"
        assert fake_backend["created"][0][1]["device"] == "cpu"

    def test_explicit_device_overrides_env(self, fake_backend, monkeypatch):
        monkeypatch.setenv("AIDIO_DEVICE", "cpu")
        result = transcribe_lyrics("song.mp3", device="cuda")
        assert result.device == "cuda"

    def test_no_demucs_by_default(self, fake_backend):
        with patch("aidio.pipeline._separate_vocals") as mock_demucs:
            result = transcribe_lyrics("song.mp3")
            assert result.used_demucs is False
            mock_demucs.assert_not_called()

    def test_demucs_runs_when_requested(self, fake_backend):
        with patch("aidio.pipeline._separate_vocals") as mock_demucs:
            result = transcribe_lyrics("song.mp3", device="cpu", separate_vocals=True)
            assert result.used_demucs is True
            mock_demucs.assert_called_once()
            # demucs device follows torch naming (cpu)
            assert mock_demucs.call_args.kwargs["device"] == "cpu"

    def test_demucs_uses_cuda_for_rocm(self, fake_backend):
        with (
            patch("aidio.pipeline._separate_vocals") as mock_demucs,
            patch("sys.platform", "linux"),
        ):
            transcribe_lyrics("song.mp3", device="rocm", separate_vocals=True)
            assert mock_demucs.call_args.kwargs["device"] == "cuda"

    def test_llm_correct_populates_corrections(self, fake_backend):
        with patch("aidio.pipeline.correct_lyrics") as mock_correct:
            mock_correct.return_value = "CORRECTED"
            result = transcribe_lyrics("song.mp3", llm_correct=True)
            assert result.corrections == "CORRECTED"
            assert result.lyrics == "CORRECTED"

    def test_no_llm_without_correct_flag(self, fake_backend):
        with patch("aidio.pipeline.correct_lyrics") as mock_correct:
            result = transcribe_lyrics("song.mp3")
            mock_correct.assert_not_called()
            assert result.corrections is None

    def test_embed_into_passes_original_file(self, fake_backend):
        with patch("aidio.lyrics.output.embed_lyrics_id3") as mock_embed:
            transcribe_lyrics("song.mp3", embed_into="original.mp3")
            mock_embed.assert_called_once()
            assert mock_embed.call_args.args[1] == "original.mp3"

    def test_progress_callback_reaches_completion(self, fake_backend):
        updates: list[tuple[float, str]] = []
        transcribe_lyrics(
            "song.mp3", progress=lambda fraction, stage: updates.append((fraction, stage))
        )
        assert updates
        assert updates[-1] == (1.0, "Done")
        fractions = [fraction for fraction, _ in updates]
        assert fractions == sorted(fractions)
        assert any("Transcribing" in stage for _, stage in updates)

    def test_language_is_forwarded_to_backend(self, fake_backend):
        transcribe_lyrics("song.mp3", language="de")
        assert fake_backend["created"][0][1]["language"] == "de"


class TestCorrectLyrics:
    def _fake_client(self, content="fixed lyrics"):
        client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content=content))]
        client.chat.completions.create.return_value = response

        constructor = MagicMock(return_value=client)
        return constructor, client

    def test_correction_success(self):
        constructor, _ = self._fake_client("fixed lyrics")
        with patch("aidio.lyrics.correct.OpenAI", constructor):
            out = correct_lyrics("raw lyrics", endpoint="http://x/v1")
        assert out == "fixed lyrics"

    def test_correction_with_context(self):
        constructor, _ = self._fake_client("ok")
        with patch("aidio.lyrics.correct.OpenAI", constructor):
            correct_lyrics("raw", context="fast tempo, male vocal")
        user_msg = constructor.call_args
        _ = user_msg
        client = constructor.return_value
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        assert "fast tempo" in messages[1]["content"]

    def test_env_var_replaces_system_prompt(self, monkeypatch):
        monkeypatch.setenv("AIDIO_CORRECTION_PROMPT", "custom system prompt")
        constructor, _ = self._fake_client("ok")
        with patch("aidio.lyrics.correct.OpenAI", constructor):
            correct_lyrics("raw", endpoint="http://x/v1")
        messages = constructor.return_value.chat.completions.create.call_args.kwargs["messages"]
        assert messages[0]["content"] == "custom system prompt"

    def test_default_prompt_protects_timestamps(self):
        assert "timestamps" in CORRECT_SYSTEM_PROMPT
        assert "Return ONLY the corrected lyrics" in CORRECT_SYSTEM_PROMPT

    def test_correction_failure_returns_original(self):
        constructor = MagicMock(side_effect=RuntimeError("down"))
        with patch("aidio.lyrics.correct.OpenAI", constructor):
            out = correct_lyrics("original text")
        assert out == "original text"

    def test_empty_correction_returns_original(self):
        constructor, _ = self._fake_client("   ")
        with patch("aidio.lyrics.correct.OpenAI", constructor):
            out = correct_lyrics("original text")
        assert out == "original text"
