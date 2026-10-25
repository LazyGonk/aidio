"""Test configuration.

Adds the parent directory to sys.path so tests can import the aidio package.
"""

import sys
from pathlib import Path

# Add the project root to the Python path so tests can import the aidio package
# conftest.py lives in tests/, so the parent directory is the project root.
aidio_dir = Path(__file__).resolve().parent.parent
if str(aidio_dir) not in sys.path:
    sys.path.insert(0, str(aidio_dir))
