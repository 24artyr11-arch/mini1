from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from aiogram import Bot, Dispatcher
from aiogram.types import MenuButtonWebApp, WebAppInfo

from .api import create_api
from .core import (
    AccessStore,
    Settings,
    SharedMarketCache,
    SignalCache,
    SignalService,
    get_settings,
    router,
)

logger = logging.getLogger(__name__)


async def configure_miniapp_menu(bot: Bot, settings: Settings) -> None:
    if not settings.miniapp_url:
        logger.info("MINIAPP_URL is not configured; Telegram menu button is unchanged.")
        return
    if not settings.miniapp_url.startswith("https://"):
        logger.error("MINIAPP_URL must use HTTPS; Telegram menu button was not changed.")
        return
    try:
        await bot.set_chat_menu_button(
            menu_button=MenuButtonWebApp(
                text="Open Trader",
                web_app=WebAppInfo(url=settings.miniapp_url),
            )
        )
        logger.info("Telegram menu button configured for %s", settings.miniapp_url)
    except Exception:
        logger.exception("Unable to configure the Telegram Mini App menu button.")


def build_runtime(settings: Settings):
    """Build one FastAPI process; Telegram polling is managed by its lifespan."""
    shared_cache = SharedMarketCache()
    signal_service = SignalService(settings, shared_cache)
    access_store = AccessStore(settings.database_path)

    bot = Bot(token=settings.telegram_bot_token)
    dp = Dispatcher()
    dp.include_router(router)
    dp["signal_service"] = signal_service
    dp["signal_cache"] = SignalCache()
    dp["access_store"] = access_store
    dp["settings"] = settings

    @asynccontextmanager
    async def lifespan(app):
        await configure_miniapp_menu(bot, settings)
        polling_task = asyncio.create_task(dp.start_polling(bot), name="telegram-polling")
        app.state.telegram_polling_task = polling_task
        logger.info(
            "MENTAL TRADER started: Mini App backend + admin bot | %s:%s | FCS API + Twelve Data | M15",
            settings.api_host,
            settings.api_port,
        )
        try:
            yield
        finally:
            if not polling_task.done():
                polling_task.cancel()
                try:
                    await polling_task
                except asyncio.CancelledError:
                    pass
            await bot.session.close()
            logger.info("MENTAL TRADER stopped.")

    # Pass lifespan directly into FastAPI: no deprecated @app.on_event hooks and
    # no second Uvicorn server.
    api = create_api(
        settings,
        signal_service,
        access_store,
        shared_cache,
        bot,
        lifespan=lifespan,
    )
    return bot, dp, api, shared_cache


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    settings = get_settings()
    if not settings.admin_telegram_id:
        logger.warning(
            "ADMIN_TELEGRAM_ID is not set; payment review/admin access will be unavailable."
        )
    if not settings.backend_api_key:
        logger.warning(
            "BACKEND_API_KEY is not set; protected admin/test API endpoints will return 503."
        )
    if not settings.miniapp_origin:
        logger.warning(
            "MINIAPP_ORIGIN is not set. Configure it to the exact Mini App HTTPS origin before production."
        )

    _, _, api, _ = build_runtime(settings)
    server = uvicorn.Server(
        uvicorn.Config(
            api,
            host=settings.api_host,
            port=settings.api_port,
            log_level="info",
            access_log=True,
        )
    )
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
