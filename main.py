from __future__ import annotations

import asyncio
import logging

import uvicorn
from fastapi import FastAPI

from app.main import build_runtime, configure_miniapp_menu
from app.core import get_settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("mental_trader.entrypoint")


# Build the shared runtime once.
# IMPORTANT: unlike app.main.main(), this file does NOT start a second Uvicorn server.
settings = get_settings()
bot, dp, api, shared_cache = build_runtime(settings)

# Expose a real FastAPI application at module level so Bothost detects
# this repository as a web application and does not need an HTTP wrapper.
app: FastAPI = api


def _log_polling_result(task: asyncio.Task) -> None:
    """Log an unexpected Telegram polling stop without crashing FastAPI."""
    if task.cancelled():
        return
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return

    if exc is not None:
        logger.exception(
            "Telegram polling stopped with an error.",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    else:
        logger.warning("Telegram polling stopped unexpectedly.")


@app.on_event("startup")
async def startup_telegram_bot() -> None:
    """Start Telegram polling inside the same process as FastAPI."""
    if getattr(app.state, "telegram_polling_task", None):
        return

    if not settings.admin_telegram_id:
        logger.warning(
            "ADMIN_TELEGRAM_ID is not configured; admin/payment review features are unavailable."
        )

    if not settings.backend_api_key:
        logger.warning(
            "BACKEND_API_KEY is not configured; protected test/admin API endpoints may be unavailable."
        )

    if not settings.miniapp_origin:
        logger.warning(
            "MINIAPP_ORIGIN is not configured; configure the exact Mini App HTTPS origin for CORS."
        )

    await configure_miniapp_menu(bot, settings)

    task = asyncio.create_task(
        dp.start_polling(bot),
        name="telegram-polling",
    )
    task.add_done_callback(_log_polling_result)
    app.state.telegram_polling_task = task

    logger.info(
        "MENTAL TRADER started: FastAPI + Telegram polling | %s:%s | FCS API + Twelve Data | M15",
        settings.api_host,
        settings.api_port,
    )


@app.on_event("shutdown")
async def shutdown_telegram_bot() -> None:
    """Stop Telegram polling and close the Bot API session cleanly."""
    task = getattr(app.state, "telegram_polling_task", None)

    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    await bot.session.close()
    logger.info("MENTAL TRADER stopped.")


if __name__ == "__main__":
    # Bothost should run this file directly. PORT supplied by Bothost is already
    # respected by get_settings(), with API_PORT used as a fallback.
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
        access_log=True,
    )
