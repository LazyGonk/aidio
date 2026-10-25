"""Windows CUDA DLL discovery for CTranslate2 / faster-whisper.

CTranslate2 loads cuBLAS and cuDNN through the Windows DLL loader. The
CUDA runtime bundled with torch wheels lives in ``site-packages/torch/lib``
and ``site-packages/nvidia/*/bin``, which are not on ``PATH``. Without
registering those directories, faster-whisper fails with::

    Library cublas64_12.dll is not found or cannot be loaded

This module registers the bundled directories via ``os.add_dll_directory``
(and prepends them to ``PATH`` for libraries that still rely on it).

Connects to: asr.py (before importing faster_whisper) and web.py's device
detection (before importing ctranslate2). No-op on non-Windows platforms.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_handles: list[object] = []
_enabled = False


def _torch_dirs() -> list[Path]:
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        return []
    return [Path(location) / "lib" for location in spec.submodule_search_locations]


def _nvidia_dirs() -> list[Path]:
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return []
    dirs: list[Path] = []
    for location in spec.submodule_search_locations:
        root = Path(location)
        if not root.is_dir():
            continue
        for package in root.iterdir():
            for folder in ("bin", "lib"):
                candidate = package / folder
                if candidate.is_dir():
                    dirs.append(candidate)
    return dirs


def candidate_dll_dirs() -> list[Path]:
    """Existing directories that may contain the CUDA runtime DLLs."""
    seen: set[str] = set()
    result: list[Path] = []
    for directory in [*_torch_dirs(), *_nvidia_dirs()]:
        key = str(directory).lower()
        if key in seen or not directory.is_dir():
            continue
        seen.add(key)
        result.append(directory)
    return result


def enable_cuda_dlls() -> list[Path]:
    """Register bundled CUDA DLL directories; returns the ones registered.

    Safe to call repeatedly: only the first call touches the process.
    """
    global _enabled
    if sys.platform != "win32":
        return []
    if _enabled:
        return candidate_dll_dirs()

    registered: list[Path] = []
    add_dll_directory = getattr(os, "add_dll_directory", None)
    for directory in candidate_dll_dirs():
        if add_dll_directory is not None:
            try:
                _handles.append(add_dll_directory(str(directory)))
            except OSError:
                continue
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
        registered.append(directory)

    _enabled = True
    return registered


def is_cuda_library_error(error: BaseException) -> bool:
    """True when an exception looks like a missing CUDA runtime DLL."""
    message = str(error).lower()
    return any(name in message for name in ("cublas", "cudnn", "cudart"))


def cuda_library_help() -> str:
    """Actionable hint appended to CUDA library loading failures."""
    return (
        "CTranslate2 (faster-whisper) needs cuBLAS 12, but the installed torch "
        "wheel may ship cuBLAS 13. Install the matching runtime with "
        "'pip install \"nvidia-cublas-cu12>=12,<13\"' (or use a cu12 torch "
        "wheel: pip install torch --index-url "
        "https://download.pytorch.org/whl/cu130), then restart Aidio. "
        "Alternatively choose the CPU device."
    )
