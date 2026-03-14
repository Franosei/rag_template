"""Local entrypoint for running the FastAPI server."""

from __future__ import annotations

import uvicorn

from src.app.settings import settings


def main() -> None:
    """Run the API server with the configured host and port."""

    uvicorn.run(
        "src.api.server:app",
        host=settings.host,
        port=settings.port,
        reload=settings.app_env == "development",
    )
