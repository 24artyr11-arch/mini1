from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from aiogram import Bot, Dispatcher
from aiogram.types import MenuButtonWebApp, Update, WebAppInfo
from fastapi import Header, HTTPException, Request

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



def _public_backend_base_url() -> str:
    """
    Resolve the public backend URL.

    Bothost automatically exposes DOMAIN inside the container. A manual
    TELEGRAM_WEBHOOK_URL may be used on other hosts or for custom routing.
    """
    explicit = (os.getenv("TELEGRAM_WEBHOOK_URL") or "").strip().rstrip("/")
    if explicit:
        if explicit.endswith("/api/telegram/webhook"):
            return explicit[: -len("/api/telegram/webhook")]
        return explicit

    domain = (os.getenv("DOMAIN") or "").strip().rstrip("/")
    if not domain:
        return ""
    if domain.startswith("https://"):
        return domain
    if domain.startswith("http://"):
        return "https://" + domain[len("http://"):]
    return f"https://{domain}"


def _telegram_webhook_url() -> str:
    explicit = (os.getenv("TELEGRAM_WEBHOOK_URL") or "").strip()
    if explicit:
        return explicit.rstrip("/")
    base = _public_backend_base_url()
    return f"{base}/api/telegram/webhook" if base else ""


def _telegram_webhook_secret(settings: Settings) -> str:
    explicit = (os.getenv("TELEGRAM_WEBHOOK_SECRET") or "").strip()
    if explicit:
        # Telegram permits A-Z, a-z, 0-9, underscore and hyphen.
        safe = "".join(ch for ch in explicit if ch.isalnum() or ch in "_-")
        if safe:
            return safe[:256]
    # Deterministic server-only fallback: no extra env variable required.
    raw = f"mental-trader-webhook:{settings.telegram_bot_token}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()



def build_runtime(settings: Settings):
    """
    Build one FastAPI process.

    Production/Bothost uses a Telegram webhook so admin callback buttons do not
    depend on long-polling. Local development falls back to polling when no
    public domain/webhook URL is available.
    """
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

    webhook_url = _telegram_webhook_url()
    webhook_secret = _telegram_webhook_secret(settings)

    @asynccontextmanager
    async def lifespan(app):
        await configure_miniapp_menu(bot, settings)

        polling_task = None
        if webhook_url:
            await bot.set_webhook(
                url=webhook_url,
                secret_token=webhook_secret,
                allowed_updates=dp.resolve_used_update_types(),
                drop_pending_updates=False,
            )
            logger.info(
                "MENTAL TRADER started: Mini App backend + admin bot WEBHOOK | %s:%s | webhook=%s | M15",
                settings.api_host,
                settings.api_port,
                webhook_url,
            )
        else:
            # Local fallback only. In production, DOMAIN is provided by Bothost.
            await bot.delete_webhook(drop_pending_updates=False)
            polling_task = asyncio.create_task(
                dp.start_polling(bot),
                name="telegram-polling",
            )
            app.state.telegram_polling_task = polling_task
            logger.warning(
                "No public DOMAIN/TELEGRAM_WEBHOOK_URL found; using Telegram polling fallback."
            )

        try:
            yield
        finally:
            if polling_task is not None and not polling_task.done():
                polling_task.cancel()
                try:
                    await polling_task
                except asyncio.CancelledError:
                    pass
            await bot.session.close()
            logger.info("MENTAL TRADER stopped.")

    api = create_api(
        settings,
        signal_service,
        access_store,
        shared_cache,
        bot,
        lifespan=lifespan,
    )

    @api.post("/api/telegram/webhook", include_in_schema=False)
    async def telegram_webhook(
        request: Request,
        x_telegram_bot_api_secret_token: str | None = Header(
            default=None,
            alias="X-Telegram-Bot-Api-Secret-Token",
        ),
    ) -> dict:
        # Reject forged requests before parsing the update.
        if x_telegram_bot_api_secret_token != webhook_secret:
            raise HTTPException(status_code=403, detail="Invalid Telegram webhook secret.")

        try:
            payload = await request.json()
            update = Update.model_validate(payload, context={"bot": bot})
            await dp.feed_update(bot, update)
        except HTTPException:
            raise
        except Exception:
            logger.exception("Telegram webhook update failed")
            # A 500 tells Telegram to retry this update.
            raise HTTPException(status_code=500, detail="Telegram update failed.")
        return {"ok": True}

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
