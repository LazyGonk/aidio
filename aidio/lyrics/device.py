"""Device selection and validation.

Connects to: asr.py (Whisper device/compute_type) and audio.py (demucs
device flag). Centralizes device mapping so unknown values error at
startup, not mid-pipeline.

Choices: cuda (default), cpu, rocm (Linux only -- hard-errors on
win32/macOS). Override order: explicit argument > AIDIO_DEVICE env var.
"""

from __future__ import annotations

import os
import sys

VALID_DEVICES = ("cuda", "cpu", "rocm")

DEFAULT_DEVICE = "cuda"

# Whisper device/compute_type per logical device.
# rocm: CTranslate2 ROCm builds expose themselves as CUDA.
WHISPER_DEVICE_MAP: dict[str, dict[str, str]] = {
    "cuda": {"device": "cuda", "compute_type": "float16"},
    "cpu": {"device": "cpu", "compute_type": "int8"},
    "rocm": {"device": "cuda", "compute_type": "float16"},
}

# Demucs `-d` flag value per logical device (torch device names).
DEMUCS_DEVICE_MAP: dict[str, str] = {
    "cuda": "cuda",
    "cpu": "cpu",
    "rocm": "cuda",
}


class DeviceError(ValueError):
    """Raised when a device is unknown or unsupported on this platform."""


def validate_device(device: str) -> str:
    """Validate a device name and platform compatibility; return canonical value.

    Connects to: pipeline.py and cli.py -- every entry point validates here
    so errors surface before the pipeline starts.
    """
    device = (device or DEFAULT_DEVICE).strip().lower()
    if device not in VALID_DEVICES:
        raise DeviceError(f"Unknown device {device!r}. Valid: {', '.join(VALID_DEVICES)}")
    if device == "rocm" and sys.platform in ("win32", "darwin"):
        raise DeviceError("Device 'rocm' is only supported on Linux. Use --device cpu instead.")
    return device


def resolve_device(device: str | None = None) -> str:
    """Resolve device from explicit argument, falling back to AIDIO_DEVICE env var."""
    chosen = device or os.environ.get("AIDIO_DEVICE") or DEFAULT_DEVICE
    return validate_device(chosen)


def whisper_args(device: str) -> dict[str, str]:
    """Map a validated device to faster-whisper kwargs (device, compute_type)."""
    return WHISPER_DEVICE_MAP[validate_device(device)]


def demucs_device(device: str) -> str:
    """Map a validated device to the demucs `-d` flag value."""
    return DEMUCS_DEVICE_MAP[validate_device(device)]
