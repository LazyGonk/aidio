"""Tests for Windows CUDA DLL discovery."""

from pathlib import Path

import aidio.lyrics.cuda_dlls as cuda_dlls


def test_enable_cuda_dlls_is_noop_off_windows(monkeypatch):
    monkeypatch.setattr(cuda_dlls.sys, "platform", "linux")
    monkeypatch.setattr(cuda_dlls, "_enabled", False)
    assert cuda_dlls.enable_cuda_dlls() == []


def test_enable_cuda_dlls_registers_bundled_dirs(monkeypatch, tmp_path):
    torch_lib = tmp_path / "torch" / "lib"
    nvidia_bin = tmp_path / "nvidia" / "cublas" / "bin"
    torch_lib.mkdir(parents=True)
    nvidia_bin.mkdir(parents=True)

    monkeypatch.setattr(cuda_dlls.sys, "platform", "win32")
    monkeypatch.setattr(cuda_dlls, "_enabled", False)
    monkeypatch.setattr(cuda_dlls, "_torch_dirs", lambda: [torch_lib])
    monkeypatch.setattr(cuda_dlls, "_nvidia_dirs", lambda: [nvidia_bin])

    added: list[str] = []
    monkeypatch.setattr(
        cuda_dlls.os,
        "add_dll_directory",
        lambda path: added.append(path) or object(),
        raising=False,
    )
    monkeypatch.setenv("PATH", "C:\\existing")

    try:
        dirs = cuda_dlls.enable_cuda_dlls()
    finally:
        cuda_dlls._handles.clear()
        cuda_dlls._enabled = False

    assert dirs == [torch_lib, nvidia_bin]
    assert added == [str(torch_lib), str(nvidia_bin)]
    assert cuda_dlls.os.environ["PATH"].startswith(str(nvidia_bin))


def test_is_cuda_library_error():
    assert cuda_dlls.is_cuda_library_error(RuntimeError("Library cublas64_12.dll is not found"))
    assert cuda_dlls.is_cuda_library_error(RuntimeError("cudnn_ops64_9.dll missing"))
    assert not cuda_dlls.is_cuda_library_error(RuntimeError("model file not found"))


def test_candidate_dirs_skip_missing(tmp_path, monkeypatch):
    existing = tmp_path / "exists"
    existing.mkdir()
    monkeypatch.setattr(cuda_dlls, "_torch_dirs", lambda: [existing, Path(tmp_path / "missing")])
    monkeypatch.setattr(cuda_dlls, "_nvidia_dirs", lambda: [existing])
    assert cuda_dlls.candidate_dll_dirs() == [existing]
