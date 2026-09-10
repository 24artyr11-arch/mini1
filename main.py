from __future__ import annotations

import logging

import uvicorn

from app.core import get_settings
from app.main import build_runtime


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

settings = get_settings()
_, _, app, _ = build_runtime(settings)


if __name__ == "__main__":
    # Bothost may inject PORT; get_settings() gives it priority over API_PORT.
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
        access_log=True,
    )
