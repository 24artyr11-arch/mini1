from __future__ import annotations

import asyncio
import logging

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


def build_runtime(settings: Settings):
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

    api = create_api(settings, signal_service, access_store, shared_cache)
    return bot, dp, api, shared_cache


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

    bot, dp, api, _ = build_runtime(settings)
    await configure_miniapp_menu(bot, settings)

    uv_config = uvicorn.Config(
        api,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
        access_log=True,
    )
    server = uvicorn.Server(uv_config)

    logger.info(
        "Starting MENTAL TRADER: Telegram + FastAPI on %s:%s | FCS API + Twelve Data | M15",
        settings.api_host,
        settings.api_port,
    )

    api_task = asyncio.create_task(server.serve(), name="fastapi")
    bot_task = asyncio.create_task(dp.start_polling(bot), name="telegram")

    try:
        done, pending = await asyncio.wait(
            {api_task, bot_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            exc = task.exception()
            if exc:
                raise exc
        for task in pending:
            task.cancel()
    finally:
        server.should_exit = True
        if not api_task.done():
            try:
                await asyncio.wait_for(api_task, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                api_task.cancel()
        if not bot_task.done():
            bot_task.cancel()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
