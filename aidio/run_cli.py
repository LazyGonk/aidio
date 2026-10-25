"""CLI entry point for Aidio.

Run with::

    python -m aidio.run_cli analyze path/to/audio.mp3

Prefer the installed console script:

    aidio analyze path/to/audio.mp3
"""

from __future__ import annotations

import sys

from aidio.cli import main

if __name__ == "__main__":
    sys.exit(main())
