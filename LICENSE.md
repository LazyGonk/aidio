# MIT License

Copyright (c) 2026 LazyGonk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

Third-party dependency licenses (see pyproject.toml optional-dependencies):

- Demucs — MIT (optional extra, transcription)
- faster-whisper — MIT (optional extra, transcription)
- mutagen — GPLv2+ (optional `embed` extra only; NOT installed by `transcribe`,
  `gpu`, or `rocm`. ID3/Vorbis/MP4 embedding needs it; `.txt` export does not.)
- All core dependencies (fastapi, uvicorn, httpx, pydantic, librosa, numpy, …) —
  permissive licenses (MIT, BSD, Apache-2.0, ISC and similar).
- The `gpu` extra on Windows also installs `nvidia-cublas-cu12`, which is
  NVIDIA proprietary (not copyleft, but not permissive either); it is a
  separate redistributable runtime, not part of Aidio's source.
- The complete per-package license list lives in
  [`aidio/THIRD-PARTY-NOTICES.md`](aidio/THIRD-PARTY-NOTICES.md).

PyTorch is used for transcription via its own permissive license (BSD-style);
CUDA runtime libraries bundled in its wheels are NVIDIA proprietary.
