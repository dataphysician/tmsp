"""Entry point for running the server as a module

Usage:
    uv run tmsp-server
    uv run tmsp-server --port 8080
"""

import sys

import uvicorn

from .app import app


def main():
    """Run the server."""
    port = 8000
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--port" and i < len(sys.argv):
            port = int(sys.argv[i + 1])

    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
