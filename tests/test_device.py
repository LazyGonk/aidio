"""Tests for device selection and mapping."""

import sys
from unittest.mock import patch

import pytest

from aidio.lyrics.device import (
    DEMUCS_DEVICE_MAP,
    WHISPER_DEVICE_MAP,
    DeviceError,
    demucs_device,
    resolve_device,
    validate_device,
    whisper_args,
)


class TestValidateDevice:
    def test_valid_lowercase(self):
        assert validate_device("cuda") == "cuda"
        assert validate_device("cpu") == "cpu"

    def test_normalizes_whitespace_and_case(self):
        assert validate_device(" CUDA ") == "cuda"
        assert validate_device("CPU") == "cpu"

    def test_none_defaults_to_cuda(self):
        assert validate_device(None) == "cuda"

    def test_unknown_device_raises(self):
        with pytest.raises(DeviceError):
            validate_device("vulkan")

    def test_rocm_rejected_on_windows(self):
        with patch.object(sys, "platform", "win32"):
            with pytest.raises(DeviceError, match="Linux"):
                validate_device("rocm")

    def test_rocm_rejected_on_macos(self):
        with patch.object(sys, "platform", "darwin"):
            with pytest.raises(DeviceError):
                validate_device("rocm")

    def test_rocm_allowed_on_linux(self):
        with patch.object(sys, "platform", "linux"):
            assert validate_device("rocm") == "rocm"


class TestResolveDevice:
    def test_env_fallback(self):
        with patch.dict("os.environ", {"AIDIO_DEVICE": "cpu"}, clear=False):
            assert resolve_device() == "cpu"

    def test_explicit_beats_env(self):
        with patch.dict("os.environ", {"AIDIO_DEVICE": "cpu"}, clear=False):
            assert resolve_device("cuda") == "cuda"

    def test_default_without_env(self, monkeypatch):
        monkeypatch.delenv("AIDIO_DEVICE", raising=False)
        assert resolve_device() == "cuda"


class TestDeviceMapping:
    def test_whisper_cuda_float16(self):
        assert whisper_args("cuda") == {
            "device": "cuda",
            "compute_type": "float16",
        }

    def test_whisper_cpu_int8(self):
        assert whisper_args("cpu") == {"device": "cpu", "compute_type": "int8"}

    def test_rocm_maps_to_cuda_backend(self):
        with patch.object(sys, "platform", "linux"):
            assert whisper_args("rocm")["device"] == "cuda"

    def test_demucs_mapping(self):
        assert demucs_device("cuda") == "cuda"
        assert demucs_device("cpu") == "cpu"
        with patch.object(sys, "platform", "linux"):
            assert demucs_device("rocm") == "cuda"

    def test_maps_are_consistent(self):
        assert set(WHISPER_DEVICE_MAP) == set(DEMUCS_DEVICE_MAP)
