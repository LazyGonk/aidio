"""End-to-end tests — real faster-whisper tiny model, real ffmpeg/ffprobe.

Runs the full transcription pipeline on a generated WAV. The Whisper model
("tiny.en", ~19 MB) is downloaded on first use and cached. Marked `e2e` so it
can be excluded quickly if needed: `pytest -m "not e2e"`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

pytest.importorskip("faster_whisper", reason="transcription extras not installed")

from aidio.lyrics.asr import Segment  # noqa: E402
from aidio.pipeline import LyricsResult, transcribe_lyrics  # noqa: E402
from aidio.web import app  # noqa: E402

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def tiny_model_device() -> str:
    """Real local backend on CPU; tiny.en keeps runtime and download small."""
    return "cpu"


@pytest.fixture(scope="module")
def music_wav(tmp_path_factory) -> Path:
    """Short WAV file: guitar-fret-like tone bursts (VAD off, so it yields
    some segments even though there are no intelligible lyrics)."""
    tmp = tmp_path_factory.mktemp("e2e")
    path = tmp / "e2e_tone_test.wav"
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # speech-band-ish frequencies so tiny.en produces at least one segment
    audio = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 550 * t)
    fade = np.minimum(1.0, np.maximum(0.0, (0.1 - np.abs(t - 0.5)) / 0.05))
    audio = (audio * fade).astype(np.float32)
    sf.write(path, audio, sr)
    return path


@pytest.fixture(scope="module")
def need_ffmPEG():
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not on PATH")


class TestEndToEnd:
    def test_pipeline_real_whisper(self, music_wav, need_ffmPEG, tiny_model_device):
        """Full local pipeline: real tiny.en Whisper on a real WAV."""
        result = transcribe_lyrics(
            music_wav,
            device=tiny_model_device,
            model="tiny.en",
            backend="local",
        )
        assert isinstance(result, LyricsResult)
        assert result.device == "cpu"
        assert result.used_demucs is False
        assert result.corrections is None
        assert all(isinstance(s, Segment) for s in result.segments)

    def test_pipeline_saves_txt_next_to_mp3(self, music_wav, need_ffmPEG, tiny_model_device):
        """txt output file has exactly the same stem as the mp3/wav input."""
        result = transcribe_lyrics(music_wav, device=tiny_model_device, model="tiny.en")
        from aidio.lyrics.output import save_lyrics_txt

        txt = save_lyrics_txt(result.lyrics, music_wav)
        assert txt.exists()
        assert txt.stem == music_wav.stem
        assert txt.read_text(encoding="utf-8") == result.lyrics

    def test_web_flow_real_whisper(self, music_wav, need_ffmPEG, tiny_model_device):
        """End-to-end through the web layer: upload -> job -> result -> txt.

        The device dropdown default and job flow run exactly as a user would.
        """
        client = TestClient(app)
        with open(music_wav, "rb") as f:
            response = client.post(
                "/transcribe",
                files={"audio_file": ("e2e_tone_test.wav", f, "audio/wav")},
                data={"device": tiny_model_device, "model": "tiny.en"},
            )
        assert response.status_code == 200
        start = response.json()
        job_id = start["job_id"]

        job = client.get(f"/jobs/{job_id}").json()
        assert job["status"] == "done", job.get("error")
        assert "lyrics" in job["result"]
        assert job["original_name"] == "e2e_tone_test.wav"

        # txt download named after the original file
        txt_res = client.get(f"/jobs/{job_id}/lyrics.txt")
        assert txt_res.status_code == 200
        assert 'filename="e2e_tone_test.txt"' in txt_res.headers["content-disposition"]
        assert txt_res.text == job["result"]["lyrics"]

        # lyrics analysis step completes against a real (unreachable) LLM:
        # explain_lyrics degrades gracefully to an "unavailable" result.
        explain = client.post("/explain", json={"lyrics": job["result"]["lyrics"]})
        assert explain.status_code == 200
