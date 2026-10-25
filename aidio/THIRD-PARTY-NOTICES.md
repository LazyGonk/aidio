# Third-Party Notices

Generated from installed distribution metadata by
`python -m scripts.generate_notices` — re-run it after dependency changes.

The Aidio source code itself is MIT licensed (see LICENSE.md). The packages
below are installed unmodified by pip and remain under their own licenses.
`transcribe`, `gpu` and `rocm` install no copyleft packages; GPLv2+
`mutagen` is confined to the optional `embed` extra.

## Core dependencies (required)

| Package | Version | License | Author |
|---|---|---|---|
| fastapi | 0.141.1 | MIT | - |
| httpx | 0.28.1 | BSD-3-Clause | - |
| librosa | 0.11.0 | ISC | Brian McFee, librosa development team |
| numpy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | Travis E. Oliphant et al. |
| openai | 2.50.0 | Apache-2.0 | - |
| pydantic | 2.13.4 | MIT | - |
| python-dotenv | 1.2.2 | BSD-3-Clause | - |
| python-multipart | 0.0.32 | Apache-2.0 | - |
| soundfile | 0.13.1 | BSD 3-Clause License | Bastian Bechtold |
| uvicorn | 0.52.0 | BSD-3-Clause | - |

## Optional dependencies (extras: dev / transcribe / embed / gpu / rocm)

| Package | Version | License | Author |
|---|---|---|---|
| av | 18.1.0 | BSD-3-Clause | - |
| demucs | 4.1.0 | MIT License | - |
| faster-whisper | 1.2.1 | MIT | Guillaume Klein |
| mutagen | 1.48.1 | GPL-2.0-or-later | - |
| nvidia-cublas-cu12 | 12.9.2.10 | LicenseRef-NVIDIA-Proprietary | Nvidia CUDA Installer Team |
| pytest | 9.1.1 | MIT | Brianna Laugher, Bruno Oliveira, Floris  |
| pytest-asyncio | 1.4.0 | Apache-2.0 | - |
| ruff | 0.16.0 | MIT | - |
| torch | 2.14.0+cu130 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clau | - |

Platform note: this list reflects the reference environment the file was
generated in (Linux + CUDA torch). Windows GPU installs also pull
`nvidia-cublas-cu12` under the NVIDIA Software License Agreement, and torch
wheels bundle NVIDIA CUDA/cuDNN runtime libraries under the same terms.
Neither is redistributed by Aidio.

<details><summary>Full transitive dependency list</summary>

| Package | Version | License |
|---|---|---|
| annotated-doc | 0.0.5 | MIT |
| annotated-types | 0.8.0 | MIT |
| anyio | 4.14.2 | MIT |
| audioop-lts | 0.2.2 | PSF-2.0 |
| audioread | 3.1.0 | MIT |
| av | 18.1.0 | BSD-3-Clause |
| build | 1.6.1 | MIT |
| certifi | 2026.7.22 | MPL-2.0 |
| cffi | 2.1.0 | MIT-0 |
| charset-normalizer | 3.4.9 | MIT |
| click | 8.4.2 | BSD-3-Clause |
| colorama | 0.4.6 | BSD License |
| ctranslate2 | 4.8.2 | MIT |
| decorator | 5.3.1 | BSD-2-Clause |
| demucs | 4.1.0 | MIT License |
| distro | 1.9.0 | Apache License, Version 2.0 |
| einops | 0.8.2 | MIT |
| fastapi | 0.141.1 | MIT |
| faster-whisper | 1.2.1 | MIT |
| filelock | 3.32.3 | MIT |
| flatbuffers | 25.12.19 | Apache 2.0 |
| fsspec | 2026.7.0 | BSD-3-Clause |
| h11 | 0.16.0 | MIT |
| hf-xet | 1.7.0 | Apache-2.0 |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpcore2 | 2.13.1 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| httpx2 | 2.13.1 | BSD-3-Clause |
| huggingface_hub | 2.2.0 | Apache-2.0 |
| idna | 3.18 | BSD-3-Clause |
| iniconfig | 2.3.0 | MIT |
| jinja2 | 3.1.6 | BSD License |
| jiter | 0.16.0 | MIT |
| joblib | 1.5.3 | BSD-3-Clause |
| julius | 0.2.8 | MIT License |
| lameenc | 1.8.4 | LGPL-3.0-or-later |
| lazy-loader | 0.5 | BSD-3-Clause |
| librosa | 0.11.0 | ISC |
| llvmlite | 0.48.0 | BSD-2-Clause AND Apache-2.0 WITH LLVM-exception |
| markupsafe | 3.0.3 | BSD-3-Clause |
| mpmath | 1.3.0 | BSD |
| msgpack | 1.2.1 | Apache-2.0 |
| mutagen | 1.48.1 | GPL-2.0-or-later |
| narwhals | 2.24.0 | MIT |
| networkx | 3.6.1 | BSD-3-Clause |
| numba | 0.66.0 | BSD |
| numpy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| nvidia-cublas-cu12 | 12.9.2.10 | LicenseRef-NVIDIA-Proprietary |
| nvidia-cuda-nvrtc-cu12 | 12.9.86 | LicenseRef-NVIDIA-Proprietary |
| nvidia-ml-py | 13.615.71 | BSD |
| onnxruntime | 1.31.0 | MIT License |
| openai | 2.50.0 | Apache-2.0 |
| packaging | 26.2 | Apache-2.0 OR BSD-2-Clause |
| pip | 26.2.1 | MIT |
| pip-tools | 7.6.2 | BSD-3-Clause |
| platformdirs | 4.11.0 | MIT |
| pluggy | 1.6.0 | MIT |
| pooch | 1.9.0 | BSD-3-Clause |
| protobuf | 7.36.2 | 3-Clause BSD License |
| pycparser | 3.0 | BSD-3-Clause |
| pydantic | 2.13.4 | MIT |
| pydantic_core | 2.46.4 | MIT |
| pygments | 2.20.0 | BSD-2-Clause |
| pynvml | 13.0.1 | BSD-3-Clause |
| pyproject_hooks | 1.3.3 | MIT |
| pytest | 9.1.1 | MIT |
| pytest-asyncio | 1.4.0 | Apache-2.0 |
| python-dotenv | 1.2.2 | BSD-3-Clause |
| python-multipart | 0.0.32 | Apache-2.0 |
| pyyaml | 6.0.3 | MIT |
| requests | 2.34.2 | Apache-2.0 |
| ruff | 0.16.0 | MIT |
| safetensors | 0.8.0 | Apache Software License |
| scikit-learn | 1.9.0 | BSD-3-Clause |
| scipy | 1.18.0 | Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Develope |
| setuptools | 78.1.0 | MIT License |
| sniffio | 1.3.1 | MIT OR Apache-2.0 |
| soundfile | 0.13.1 | BSD 3-Clause License |
| soxr | 1.1.0 | LGPL-2.1-or-later |
| sphn | 0.2.1 | Apache-2.0 |
| standard-aifc | 3.13.0 | PSF-2.0 |
| standard-chunk | 3.13.0 | PSF-2.0 |
| standard-sunau | 3.13.0 | PSF-2.0 |
| starlette | 1.3.1 | BSD-3-Clause |
| sympy | 1.14.0 | BSD |
| threadpoolctl | 3.6.0 | BSD-3-Clause |
| tokenizers | 0.23.3 | Apache Software License |
| torch | 2.14.0+cu130 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clau |
| tqdm | 4.70.0 | MPL-2.0 AND MIT |
| truststore | 0.10.4 | MIT |
| typing-inspection | 0.4.2 | MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |
| urllib3 | 2.7.0 | MIT |
| uvicorn | 0.52.0 | BSD-3-Clause |
| wheel | 0.48.0 | MIT |

</details>
