"""Tests for the audio preprocessing module."""

import base64
import io
import wave

import numpy as np
import soundfile as sf

from aidio.audio import (
    AudioChunk,
    _normalize,
    _to_mono,
    _wav_bytes_to_base64,
    preprocess_audio,
)


def _generate_test_audio(
    duration: float = 1.0,
    sample_rate: int = 16000,
    channels: int = 1,
    freq: float = 440.0,
) -> str:
    """Generate a sine wave test audio file and return its path."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    y = np.sin(2 * np.pi * freq * t).astype(np.float32)

    if channels > 1:
        y = np.column_stack([y] * channels)

    path = "test_audio.wav"
    sf.write(path, y, sample_rate)
    return path


class TestWavBytesToBase64:
    """Test WAV encoding to base64."""

    def test_encodes_mono_audio(self):
        y = np.array([0.5, -0.5, 0.0], dtype=np.float32)
        b64 = _wav_bytes_to_base64(y, 16000)

        assert isinstance(b64, str)
        assert len(b64) > 0

        # Decode and verify
        decoded = base64.b64decode(b64)
        with wave.open(io.BytesIO(decoded), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000

    def test_preserves_sample_values(self):
        y = np.array([1.0, -1.0, 0.0], dtype=np.float32)
        b64 = _wav_bytes_to_base64(y, 8000)

        decoded = base64.b64decode(b64)
        with wave.open(io.BytesIO(decoded), "rb") as wf:
            data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            assert data[0] > 30000  # 1.0 → ~32767
            assert data[1] < -30000  # -1.0 → ~-32767


class TestToMono:
    """Test mono conversion."""

    def test_mono_passthrough(self):
        y = np.array([0.5, -0.5, 0.0], dtype=np.float32)
        result = _to_mono(y)
        np.testing.assert_array_equal(result, y)

    def test_stereo_averages_channels(self):
        y = np.array([[0.8, 0.2], [0.6, 0.4], [0.0, 1.0]], dtype=np.float32)
        result = _to_mono(y)

        expected = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)

    def test_multichannel_averages(self):
        y = np.random.rand(100, 4).astype(np.float32)
        result = _to_mono(y)
        assert result.shape == (100,)
        assert result.dtype == np.float32


class TestNormalize:
    """Test audio normalization."""

    def test_normalizes_to_unit_range(self):
        y = np.array([0.1, 0.5, -0.5, 0.0], dtype=np.float32)
        result = _normalize(y)

        assert np.max(np.abs(result)) == 1.0
        assert np.min(result) == -1.0

    def test_handles_zero_audio(self):
        y = np.zeros(100, dtype=np.float32)
        result = _normalize(y)
        np.testing.assert_array_equal(result, y)

    def test_preserves_dtype(self):
        y = np.array([0.5, -0.5], dtype=np.float32)
        result = _normalize(y)
        assert result.dtype == np.float32


class TestPreprocessAudio:
    """Test the full preprocessing pipeline."""

    def test_preprocesses_mono_file(self):
        path = _generate_test_audio(duration=0.5, sample_rate=44100, channels=1)
        chunks = preprocess_audio(path, target_sr=16000, max_chunk_seconds=10)

        assert len(chunks) == 1
        assert isinstance(chunks[0], AudioChunk)
        assert chunks[0].duration_seconds > 0
        assert len(chunks[0].base64_wav) > 0

    def test_preprocesses_stereo_file(self):
        path = _generate_test_audio(duration=0.5, sample_rate=44100, channels=2)
        chunks = preprocess_audio(path, target_sr=16000, max_chunk_seconds=10)

        assert len(chunks) == 1
        # Verify mono conversion happened
        decoded = base64.b64decode(chunks[0].base64_wav)
        with wave.open(io.BytesIO(decoded), "rb") as wf:
            assert wf.getnchannels() == 1

    def test_chunks_long_audio(self):
        path = _generate_test_audio(duration=5.0, sample_rate=16000, channels=1)
        chunks = preprocess_audio(path, target_sr=16000, max_chunk_seconds=2.0)

        assert len(chunks) == 3  # 5s / 2s per chunk = 3 chunks (last one shorter)
        total_duration = sum(c.duration_seconds for c in chunks)
        assert abs(total_duration - 5.0) < 0.1

    def test_resamples_to_target_sr(self):
        path = _generate_test_audio(duration=1.0, sample_rate=44100, channels=1)
        chunks = preprocess_audio(path, target_sr=8000, max_chunk_seconds=10)

        decoded = base64.b64decode(chunks[0].base64_wav)
        with wave.open(io.BytesIO(decoded), "rb") as wf:
            assert wf.getframerate() == 8000

    def test_normalizes_audio(self):
        path = _generate_test_audio(duration=1.0, sample_rate=16000, channels=1)
        chunks = preprocess_audio(path, target_sr=16000, max_chunk_seconds=10)

        decoded = base64.b64decode(chunks[0].base64_wav)
        with wave.open(io.BytesIO(decoded), "rb") as wf:
            data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            assert np.max(np.abs(data)) > 30000  # Should be near full scale
