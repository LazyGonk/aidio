"""Generate THIRD-PARTY-NOTICES.md from installed distribution metadata.

Usage::

    python -m scripts.generate_notices            # writes aidio/THIRD-PARTY-NOTICES.md

Collects license expression/classifier/license-file and author from each
installed distribution (`pip install -e ".[dev,transcribe,embed]"` first for
full coverage; add `,gpu` on Windows so the cuBLAS runtime appears).
"""

from __future__ import annotations

import pathlib
import re
import tomllib
from importlib import metadata

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "aidio" / "THIRD-PARTY-NOTICES.md"

SKIP = {"aidio"}

HEADER = """\
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
{core}

## Optional dependencies (extras: dev / transcribe / embed / gpu / rocm)

| Package | Version | License | Author |
|---|---|---|---|
{optional}

Platform note: this list reflects the reference environment the file was
generated in (Linux + CUDA torch). Windows GPU installs also pull
`nvidia-cublas-cu12` under the NVIDIA Software License Agreement, and torch
wheels bundle NVIDIA CUDA/cuDNN runtime libraries under the same terms.
Neither is redistributed by Aidio.

<details><summary>Full transitive dependency list</summary>

| Package | Version | License |
|---|---|---|
{all}

</details>
"""


def _license(dist: metadata.Distribution) -> str:
    lic = (dist.metadata.get("License-Expression") or dist.metadata.get("License") or "").strip()
    if lic:
        return " ".join(lic.split())[:60]
    classifiers = [
        c.split(":: ")[-1]
        for c in dist.metadata.get_all("Classifier") or []
        if c.startswith("License ::")
    ]
    if classifiers:
        return "; ".join(classifiers[:2])
    return _license_from_files(dist)


_LICENSE_FILE_LIMIT = 64 * 1024

# Ordered most-specific first: LGPL before GPL, Apache before MIT (Apache
# texts often mention permissive terms for bundled code).
_LICENSE_PATTERNS = (
    ("LGPL", r"GNU LESSER GENERAL PUBLIC LICENSE"),
    ("GPL", r"GNU GENERAL PUBLIC LICENSE"),
    ("AGPL", r"GNU AFFERO GENERAL PUBLIC LICENSE"),
    ("Apache-2.0", r"Apache License"),
    ("MPL-2.0", r"Mozilla Public License"),
    ("MIT", r"MIT License|Permission is hereby granted, free of charge"),
    ("BSD-3-Clause", r"Redistribution and use in source and binary forms"),
    ("ISC", r"Permission to use, copy, modify, and/or distribute this software"),
)


def _license_from_files(dist: metadata.Distribution) -> str:
    """Sniff a package's license file when metadata carries no license field."""
    for file in dist.files or ():
        name = pathlib.PurePath(str(file)).name.upper()
        if not name.startswith(("LICENSE", "LICENCE", "COPYING")):
            continue
        try:
            path = dist.locate_file(file)
            text = path.read_text(encoding="utf-8", errors="ignore")[:_LICENSE_FILE_LIMIT]
        except (OSError, UnicodeError):
            continue
        for license_id, pattern in _LICENSE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return license_id
    return "unknown"


def _author(dist: metadata.Distribution) -> str:
    for key in ("Author", "Maintainer"):
        val = dist.metadata.get(key)
        if val:
            return " ".join(val.split())[:40]
    return "-"


def _dist_name(req: str) -> str:
    name = re.split(r"[<>=!;~\[ ]", req)[0]
    return name.strip().lower()


def main() -> None:
    dists = {
        d.metadata["Name"].lower(): d for d in metadata.distributions() if d.metadata.get("Name")
    }

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    core_names = {_dist_name(d) for d in pyproject["project"]["dependencies"]}
    optional_names: set[str] = set()
    for deps in pyproject["project"]["optional-dependencies"].values():
        for d in deps:
            pkg = _dist_name(d)
            if pkg not in core_names:
                optional_names.add(pkg)

    core_rows: list[str] = []
    optional_rows: list[str] = []
    all_rows: list[str] = []
    for name in sorted(dists):
        if name in SKIP:
            continue
        dist = dists[name]
        version = dist.metadata["Version"]
        lic = _license(dist).replace("|", "/")
        all_rows.append(f"| {name} | {version} | {lic} |")
        row = f"| {name} | {version} | {lic} | {_author(dist)} |"
        if name in core_names:
            core_rows.append(row)
        elif name in optional_names:
            optional_rows.append(row)

    OUTPUT.write_text(
        HEADER.format(
            core="\n".join(core_rows),
            optional="\n".join(optional_rows),
            all="\n".join(all_rows),
        ),
        encoding="utf-8",
    )
    print(
        f"Wrote {OUTPUT.name} (core: {len(core_rows)}, optional: "
        f"{len(optional_rows)}, total: {len(all_rows)})"
    )


if __name__ == "__main__":
    main()
