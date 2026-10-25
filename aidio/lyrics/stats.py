"""Timing and memory statistics for the transcription pipeline.

Provides:
- Wall-clock timing for demucs and whisper operations
- CPU/RAM usage via psutil
- GPU VRAM usage via pynvml (graceful fallback if no NVIDIA GPU)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Shared registry: operation name -> {"duration": float, "peak_gpu_mb": float}
_stats_registry: dict[str, dict] = {}


def get_stats() -> dict[str, dict]:
    """Return the shared stats registry (populated by context managers)."""
    return _stats_registry


def clear_stats() -> None:
    """Clear the stats registry between songs."""
    _stats_registry.clear()


@dataclass
class MemorySnapshot:
    """CPU/RAM and GPU VRAM usage at a point in time."""

    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    gpu_vram_used_mb: float
    gpu_vram_total_mb: float
    gpu_util_percent: float


def _get_gpu_stats() -> tuple[float, float, float]:
    """Return (vram_used_mb, vram_total_mb, util_percent) via pynvml.

    Returns (0, 0, 0) if pynvml is unavailable or no GPU detected.
    """
    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return (
            info.used / (1024 * 1024),
            info.total / (1024 * 1024),
            util.gpu,
        )
    except Exception:  # noqa: BLE001
        return 0.0, 0.0, 0.0


def get_memory_snapshot() -> MemorySnapshot:
    """Capture current CPU, RAM, and GPU VRAM usage."""
    try:
        import psutil

        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
    except ImportError:
        cpu = 0.0
        ram = None
    gpu_used, gpu_total, gpu_util = _get_gpu_stats()
    return MemorySnapshot(
        cpu_percent=cpu,
        ram_used_gb=(ram.used / (1024**3) if ram is not None else 0.0),
        ram_total_gb=(ram.total / (1024**3) if ram is not None else 0.0),
        gpu_vram_used_mb=gpu_used,
        gpu_vram_total_mb=gpu_total,
        gpu_util_percent=gpu_util,
    )


@contextmanager
def timed_operation(name: str) -> Generator[None, None, None]:
    """Context manager that logs wall-clock duration for an operation.

    Usage:
        with timed_operation("demucs separation"):
            separate_vocals(...)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        logger.info("%s: %.2fs", name, duration)


@contextmanager
def timed_operation_with_stats(name: str) -> Generator[None, None, None]:
    """Context manager that logs duration + memory stats before/after an operation.

    Usage:
        with timed_operation_with_stats("whisper transcription"):
            backend.transcribe(...)
    """
    before = get_memory_snapshot()
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        after = get_memory_snapshot()
        logger.info(
            "%s: %.2fs | CPU: %.1f%% | RAM: %.1f/%.1f GB | GPU VRAM: %.0f/%.0f MB (%.0f%% util)",
            name,
            duration,
            after.cpu_percent,
            after.ram_used_gb,
            after.ram_total_gb,
            after.gpu_vram_used_mb,
            after.gpu_vram_total_mb,
            after.gpu_util_percent,
        )
        # Register for summary table
        peak_gpu = max(before.gpu_vram_used_mb, after.gpu_vram_used_mb)
        _stats_registry[name] = {
            "duration": duration,
            "peak_gpu_mb": peak_gpu,
        }


def print_summary(stats: dict[str, dict]) -> None:
    """Print a summary table of all recorded operations.

    Args:
        stats: dict mapping operation name to {"duration": float,
               "peak_gpu_mb": float, "peak_ram_gb": float}.
    """
    if not stats:
        return
    print("\n" + "=" * 70)
    print("Pipeline Statistics")
    print("=" * 70)
    print(f"{'Operation':<30} {'Duration':>10} {'Peak GPU VRAM':>15}")
    print("-" * 70)
    total = 0.0
    for name, data in stats.items():
        dur = data["duration"]
        gpu = data.get("peak_gpu_mb", 0)
        total += dur
        print(f"{name:<30} {dur:>9.2f}s {gpu:>14.0f} MB")
    print("-" * 70)
    print(f"{'TOTAL':<30} {total:>9.2f}s")
    print("=" * 70)
