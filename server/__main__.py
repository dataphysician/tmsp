"""Entry point for running the server as a module

Usage:
    uv run python -m server
    uvicorn server.app:app --reload
    uv run tmsp-server
"""

import uvicorn

from .app import app


def main():
    """Run the server."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
