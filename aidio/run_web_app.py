"""Web server entry point for Aidio.

Run with::

    aidio-web
    python -m aidio.run_web_app

Starts a FastAPI server with uvicorn on http://127.0.0.1:8000.
"""

from __future__ import annotations

import argparse
import sys

import uvicorn

from aidio import banner
from aidio.web import app


def main(argv: list[str] | None = None) -> int:
    """Start the web server."""
    parser = argparse.ArgumentParser(description="Aidio Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args(argv)

    print(banner())
    print(f"🎵 Aidio Web UI running at http://{args.host}:{args.port}")
    print("   Press Ctrl+C to stop")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
