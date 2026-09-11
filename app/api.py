from __future__ import annotations

import hmac
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from aiogram import Bot
from aiogram.types import BufferedInputFile
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from .auth import MiniAppAuthError, MiniAppUser, validate_telegram_init_data
from .core import (
    INSTRUMENTS,
    MARKET_TITLES,
    AccessStore,
    Market,
    MarketDataError,
    MarketDataRateLimitError,
    FcsInstrumentUnavailable,
    Settings,
    SharedMarketCache,
    Signal,
    SignalService,
    admin_review_keyboard,
    get_instrument,
    payment_amount,
)


def _iso(dt):
    return dt.astimezone(timezone.utc).isoformat() if dt else None


def _signal_payload(signal: Signal, cached: bool) -> dict:
    return {
        "market": signal.instrument.market.value,
        "instrument_id": signal.instrument.id,
        "symbol": signal.instrument.label,
        "provider_symbol": signal.provider_symbol,
        "data_source": signal.data_source,
        "timeframe": signal.timeframe,
        "direction": signal.direction.value,
        "score": signal.score,
        "current_price": signal.current_price,
        "digits": signal.digits,
        "entry": signal.entry,
        "stop_loss": signal.stop_loss,
        "take_profit_1": signal.take_profit_1,
        "take_profit_2": signal.take_profit_2,
        "risk_reward": signal.risk_reward,
        "analysis": {
            "atr": signal.atr,
            "rsi": signal.rsi,
            "ema20": signal.ema20,
            "ema50": signal.ema50,
            "ema200": signal.ema200,
            "macd": signal.macd,
            "macd_signal": signal.macd_signal,
            "support": signal.support,
            "resistance": signal.resistance,
            "setup": signal.setup,
            "reasons": signal.reasons,
            "blockers": signal.blockers,
        },
        "price_fetched_at": _iso(signal.price_fetched_at),
        "cached": cached,
    }


@dataclass(frozen=True, slots=True)
class ApiIdentity:
    telegram_id: int
    username: str | None
    first_name: str | None
    auth_mode: str


class PaymentIntentRequest(BaseModel):
    plan_key: Literal["monthly", "lifetime"]
    currency: Literal["usdt", "btc"]


MAX_RECEIPT_BYTES = 8 * 1024 * 1024


def _payment_row_payload(row) -> dict | None:
    if row is None:
        return None
    return {
        "id": int(row["id"]),
        "plan_key": row["plan_key"],
        "currency": row["currency"],
        "amount": row["amount_text"],
        "status": row["status"],
        "created_at": row["created_at"],
        "reviewed_at": row["reviewed_at"],
    }


def create_api(
    settings: Settings,
    signal_service: SignalService,
    access_store: AccessStore,
    cache: SharedMarketCache,
    bot: Bot,
    lifespan=None,
) -> FastAPI:
    app = FastAPI(
        title="MENTAL TRADER Backend",
        version="3.2.0",
        description="Mini App-first MENTAL TRADER backend. Customer actions live in the Mini App; Telegram chat is reserved for admin operations and notifications.",
        lifespan=lifespan,
    )

    origin = settings.miniapp_origin
    if not origin and settings.miniapp_url:
        parsed = urlsplit(settings.miniapp_url)
        if parsed.scheme and parsed.netloc:
            origin = f"{parsed.scheme}://{parsed.netloc}"

    if origin:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[origin],
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=[
                "Accept",
                "Content-Type",
                "X-Telegram-Init-Data",
                "X-Backend-Key",
            ],
            max_age=3600,
        )

    def require_backend_key(x_backend_key: str | None) -> None:
        if not settings.backend_api_key:
            raise HTTPException(
                status_code=503,
                detail="BACKEND_API_KEY is not configured on the server.",
            )
        if not x_backend_key or not hmac.compare_digest(
            x_backend_key, settings.backend_api_key
        ):
            raise HTTPException(status_code=401, detail="Invalid backend API key.")

    def parse_market(raw: str) -> Market:
        try:
            return Market(raw.lower())
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Unknown market.") from exc

    def miniapp_identity(init_data: str | None) -> ApiIdentity:
        if not init_data:
            raise HTTPException(
                status_code=401,
                detail="Open this Mini App from Telegram.",
            )
        try:
            user: MiniAppUser = validate_telegram_init_data(
                init_data,
                settings.telegram_bot_token,
                max_age_seconds=settings.miniapp_auth_max_age_seconds,
            )
        except MiniAppAuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

        access_store.touch_user(user.telegram_id, user.username, user.first_name)
        return ApiIdentity(
            telegram_id=user.telegram_id,
            username=user.username,
            first_name=user.first_name,
            auth_mode="telegram_init_data",
        )

    def resolve_identity(
        init_data: str | None,
        backend_key: str | None,
        test_telegram_id: int | None,
    ) -> ApiIdentity:
        """Authenticate Mini App users, with a server-side key fallback for /docs testing."""
        if init_data:
            return miniapp_identity(init_data)
        if backend_key and test_telegram_id:
            require_backend_key(backend_key)
            row = access_store.get_user(test_telegram_id)
            return ApiIdentity(
                telegram_id=test_telegram_id,
                username=row["username"] if row else None,
                first_name=row["first_name"] if row else None,
                auth_mode="backend_key_test",
            )
        raise HTTPException(status_code=401, detail="Open this Mini App from Telegram.")

    def access_payload(identity: ApiIdentity) -> dict:
        info = access_store.access_info(identity.telegram_id)
        row = access_store.get_user(identity.telegram_id)
        is_admin = identity.telegram_id == settings.admin_telegram_id
        return {
            "telegram_id": identity.telegram_id,
            "username": identity.username or (row["username"] if row else None),
            "first_name": identity.first_name or (row["first_name"] if row else None),
            "active": is_admin or info.active,
            "plan": "admin" if is_admin else info.plan,
            "expires_at": _iso(info.expires_at),
            "status": "active" if is_admin else (row["status"] if row else "inactive"),
            "is_admin": is_admin,
            "auth_mode": identity.auth_mode,
        }

    def require_active(identity: ApiIdentity) -> None:
        if identity.telegram_id == settings.admin_telegram_id:
            return
        if not access_store.access_info(identity.telegram_id).active:
            raise HTTPException(status_code=403, detail="Active subscription required.")

    @app.get("/health")
    async def health() -> dict:
        return {
            "ok": True,
            "service": "mental-trader-backend",
            "version": "3.2.0",
            "timeframe": settings.timeframe_label,
            "miniapp_configured": bool(settings.miniapp_url),
            "cors_origin": origin or None,
            "providers": {
                "forex": "FCS API",
                "metals": "FCS API",
                "crypto": "Twelve Data",
                "nasdaq": "Twelve Data",
            },
            "cache": cache.stats(),
        }

    @app.get("/api/me")
    async def me(
        x_telegram_init_data: str | None = Header(
            default=None, alias="X-Telegram-Init-Data"
        ),
        x_backend_key: str | None = Header(default=None, alias="X-Backend-Key"),
        telegram_id: int | None = Query(default=None, ge=1),
    ) -> dict:
        identity = resolve_identity(x_telegram_init_data, x_backend_key, telegram_id)
        return access_payload(identity)

    @app.get("/api/markets")
    async def markets(
        x_telegram_init_data: str | None = Header(
            default=None, alias="X-Telegram-Init-Data"
        ),
        x_backend_key: str | None = Header(default=None, alias="X-Backend-Key"),
        telegram_id: int | None = Query(default=None, ge=1),
    ) -> list[dict]:
        resolve_identity(x_telegram_init_data, x_backend_key, telegram_id)
        return [
            {
                "id": market.value,
                "title": MARKET_TITLES[market],
                "provider": signal_service.market_data.provider_name(INSTRUMENTS[market][0]),
                "instruments": len(INSTRUMENTS[market]),
            }
            for market in Market
        ]

    @app.get("/api/instruments/{market_name}")
    async def instruments(
        market_name: str,
        x_telegram_init_data: str | None = Header(
            default=None, alias="X-Telegram-Init-Data"
        ),
        x_backend_key: str | None = Header(default=None, alias="X-Backend-Key"),
        telegram_id: int | None = Query(default=None, ge=1),
    ) -> list[dict]:
        resolve_identity(x_telegram_init_data, x_backend_key, telegram_id)
        market = parse_market(market_name)
        return [
            {
                "id": item.id,
                "label": item.label,
                "name": item.display_name,
                "digits": item.digits,
                "provider": signal_service.market_data.provider_name(item),
            }
            for item in INSTRUMENTS[market]
        ]

    @app.get("/api/price/{market_name}/{instrument_id}")
    async def price(
        market_name: str,
        instrument_id: str,
        x_telegram_init_data: str | None = Header(
            default=None, alias="X-Telegram-Init-Data"
        ),
        x_backend_key: str | None = Header(default=None, alias="X-Backend-Key"),
        telegram_id: int | None = Query(default=None, ge=1),
    ) -> dict:
        identity = resolve_identity(x_telegram_init_data, x_backend_key, telegram_id)
        require_active(identity)
        market = parse_market(market_name)
        instrument = get_instrument(market, instrument_id)
        if instrument is None:
            raise HTTPException(status_code=404, detail="Unknown instrument.")
        try:
            live, cached = await signal_service.current_price_with_meta(instrument)
        except FcsInstrumentUnavailable as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except MarketDataRateLimitError as exc:
            raise HTTPException(
                status_code=429,
                detail="Please wait a minute, the tokens have run out.",
            ) from exc
        except MarketDataError as exc:
            raise HTTPException(
                status_code=502, detail="Market data is temporarily unavailable."
            ) from exc
        return {
            "market": market.value,
            "instrument_id": instrument.id,
            "symbol": instrument.label,
            "price": live.price,
            "digits": live.digits,
            "data_source": live.data_source,
            "fetched_at": _iso(live.fetched_at),
            "cached": cached,
        }

    @app.get("/api/signal/{market_name}/{instrument_id}")
    async def signal(
        market_name: str,
        instrument_id: str,
        x_telegram_init_data: str | None = Header(
            default=None, alias="X-Telegram-Init-Data"
        ),
        x_backend_key: str | None = Header(default=None, alias="X-Backend-Key"),
        telegram_id: int | None = Query(default=None, ge=1),
    ) -> dict:
        identity = resolve_identity(x_telegram_init_data, x_backend_key, telegram_id)
        require_active(identity)
        market = parse_market(market_name)
        instrument = get_instrument(market, instrument_id)
        if instrument is None:
            raise HTTPException(status_code=404, detail="Unknown instrument.")
        try:
            result, cached = await signal_service.calculate_with_meta(instrument)
        except FcsInstrumentUnavailable as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except MarketDataRateLimitError as exc:
            raise HTTPException(
                status_code=429,
                detail="Please wait a minute, the tokens have run out.",
            ) from exc
        except (MarketDataError, ValueError) as exc:
            raise HTTPException(
                status_code=502, detail="Unable to calculate signal right now."
            ) from exc
        return _signal_payload(result, cached)

    @app.get("/api/payment/options")
    async def payment_options(
        x_telegram_init_data: str | None = Header(
            default=None, alias="X-Telegram-Init-Data"
        ),
        x_backend_key: str | None = Header(default=None, alias="X-Backend-Key"),
        telegram_id: int | None = Query(default=None, ge=1),
    ) -> dict:
        identity = resolve_identity(x_telegram_init_data, x_backend_key, telegram_id)
        latest = access_store.latest_payment(identity.telegram_id)
        return {
            "plans": [
                {"id": "monthly", "name": "Monthly", "usd": settings.monthly_usd, "duration": "30 days"},
                {"id": "lifetime", "name": "Lifetime", "usd": settings.lifetime_usd, "duration": "Permanent"},
            ],
            "methods": [
                {"id": "usdt", "name": "USDT TRC20", "network": "TRC20", "available": bool(settings.usdt_trc20_address)},
                {"id": "btc", "name": "BTC", "network": "Bitcoin", "available": bool(settings.btc_address)},
            ],
            "latest_payment": _payment_row_payload(latest),
        }

    @app.post("/api/payment/intent")
    async def create_payment_intent(
        payload: PaymentIntentRequest,
        x_telegram_init_data: str | None = Header(
            default=None, alias="X-Telegram-Init-Data"
        ),
        x_backend_key: str | None = Header(default=None, alias="X-Backend-Key"),
        telegram_id: int | None = Query(default=None, ge=1),
    ) -> dict:
        identity = resolve_identity(x_telegram_init_data, x_backend_key, telegram_id)
        if identity.telegram_id != settings.admin_telegram_id and access_store.access_info(identity.telegram_id).active:
            raise HTTPException(status_code=409, detail="Your access is already active.")
        latest = access_store.latest_payment(identity.telegram_id)
        if latest is not None and latest["status"] == "pending":
            raise HTTPException(status_code=409, detail="A payment is already pending administrator review.")
        amount = payment_amount(settings, payload.currency, payload.plan_key)
        if not amount:
            raise HTTPException(status_code=400, detail="Invalid payment selection.")
        if payload.currency == "usdt":
            wallet = settings.usdt_trc20_address
            network = "TRC20"
            method_name = "USDT TRC20"
        else:
            wallet = settings.btc_address
            network = "Bitcoin"
            method_name = "BTC"
        if not wallet:
            raise HTTPException(status_code=503, detail="This payment method is temporarily unavailable.")
        access_store.set_intent(
            identity.telegram_id, identity.username, identity.first_name,
            payload.plan_key, payload.currency, amount,
        )
        return {
            "plan_key": payload.plan_key,
            "currency": payload.currency,
            "method": method_name,
            "network": network,
            "amount": amount,
            "address": wallet,
        }

    @app.post("/api/payment/receipt")
    async def upload_payment_receipt(
        receipt: UploadFile = File(...),
        plan_key: str | None = Form(default=None),
        currency: str | None = Form(default=None),
        x_telegram_init_data: str | None = Header(
            default=None, alias="X-Telegram-Init-Data"
        ),
        x_backend_key: str | None = Header(default=None, alias="X-Backend-Key"),
        telegram_id: int | None = Query(default=None, ge=1),
    ) -> dict:
        identity = resolve_identity(x_telegram_init_data, x_backend_key, telegram_id)
        if not settings.admin_telegram_id:
            raise HTTPException(status_code=503, detail="Payment verification is temporarily unavailable.")

        # Check submitted payments first. A missing temporary intent must not hide
        # the fact that the user already has a receipt awaiting review.
        latest = access_store.latest_payment(identity.telegram_id)
        if latest is not None and latest["status"] == "pending":
            raise HTTPException(status_code=409, detail="A payment is already pending administrator review.")

        intent = access_store.get_intent(identity.telegram_id)

        # The Mini App also submits plan_key/currency with the receipt. This makes
        # checkout resilient to a Bothost process/container restart between the
        # payment-details screen and the receipt upload. Never trust an amount
        # from the client: derive it from server-side plan settings.
        if plan_key is not None or currency is not None:
            if plan_key not in {"monthly", "lifetime"} or currency not in {"usdt", "btc"}:
                raise HTTPException(status_code=400, detail="Invalid payment selection.")

            amount = payment_amount(settings, currency, plan_key)
            if not amount:
                raise HTTPException(status_code=400, detail="Invalid payment selection.")

            wallet = settings.usdt_trc20_address if currency == "usdt" else settings.btc_address
            if not wallet:
                raise HTTPException(status_code=503, detail="This payment method is temporarily unavailable.")

            if (
                intent is None
                or intent["plan_key"] != plan_key
                or intent["currency"] != currency
                or intent["amount_text"] != amount
            ):
                access_store.set_intent(
                    identity.telegram_id,
                    identity.username,
                    identity.first_name,
                    plan_key,
                    currency,
                    amount,
                )
                intent = access_store.get_intent(identity.telegram_id)

        if intent is None:
            raise HTTPException(
                status_code=409,
                detail="Payment session expired. Return to Get Access and choose the plan and payment method again.",
            )
        content_type = (receipt.content_type or "").lower()
        if not (content_type.startswith("image/") or content_type == "application/pdf"):
            raise HTTPException(status_code=415, detail="Upload an image or PDF payment receipt.")
        data = await receipt.read(MAX_RECEIPT_BYTES + 1)
        if not data:
            raise HTTPException(status_code=400, detail="The receipt file is empty.")
        if len(data) > MAX_RECEIPT_BYTES:
            raise HTTPException(status_code=413, detail="Receipt file is too large. Maximum size is 8 MB.")
        filename = Path(receipt.filename or "payment-receipt").name[:120] or "payment-receipt"
        plan_name = "30 Days" if intent["plan_key"] == "monthly" else "Lifetime"
        method_name = "USDT TRC20" if intent["currency"] == "usdt" else "BTC"
        username = f"@{identity.username}" if identity.username else "—"
        caption = (
            "MENTAL TRADER payment receipt\n"
            f"User: {username}\n"
            f"Telegram ID: {identity.telegram_id}\n"
            f"Plan: {plan_name}\n"
            f"Method: {method_name}\n"
            f"Expected amount: {intent['amount_text']}"
        )
        try:
            receipt_message = await bot.send_document(
                chat_id=settings.admin_telegram_id,
                document=BufferedInputFile(data, filename=filename),
                caption=caption,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail="Could not deliver the receipt for verification. Please try again.",
            ) from exc
        receipt_file_id = receipt_message.document.file_id if receipt_message.document else ""
        payment_id = access_store.consume_intent_and_create_payment(
            identity.telegram_id,
            receipt_message.chat.id,
            receipt_message.message_id,
            receipt_file_id,
            "miniapp_upload",
        )
        if payment_id is None:
            raise HTTPException(status_code=409, detail="Payment session expired. Start the payment process again.")
        try:
            await bot.send_message(
                settings.admin_telegram_id,
                "🧾 <b>Новый чек из Mini App</b>\n\n"
                f"Платёж ID: <code>{payment_id}</code>\n"
                f"Пользователь: {username}\n"
                f"Telegram ID: <code>{identity.telegram_id}</code>\n"
                f"Тариф: <b>{plan_name}</b>\n"
                f"Метод: <b>{method_name}</b>\n"
                f"Ожидаемая сумма: <b>{intent['amount_text']}</b>\n\n"
                "Перед подтверждением проверь поступление средств независимо от скриншота.",
                parse_mode="HTML",
                reply_markup=admin_review_keyboard(payment_id),
            )
        except Exception:
            pass
        return {
            "ok": True,
            "payment_id": payment_id,
            "status": "pending",
            "message": "Receipt received. Waiting for administrator review.",
        }

    @app.get("/api/payment/status")
    async def payment_status(
        x_telegram_init_data: str | None = Header(
            default=None, alias="X-Telegram-Init-Data"
        ),
        x_backend_key: str | None = Header(default=None, alias="X-Backend-Key"),
        telegram_id: int | None = Query(default=None, ge=1),
    ) -> dict:
        identity = resolve_identity(x_telegram_init_data, x_backend_key, telegram_id)
        latest = access_store.latest_payment(identity.telegram_id)
        return {
            "access": access_payload(identity),
            "payment": _payment_row_payload(latest),
        }

    @app.get("/api/admin/fcs/availability")
    async def fcs_availability(
        x_backend_key: str | None = Header(default=None, alias="X-Backend-Key"),
    ) -> dict:
        require_backend_key(x_backend_key)
        try:
            available = await __import__("asyncio").to_thread(
                signal_service.market_data.fcs.symbol_availability
            )
        except MarketDataError as exc:
            raise HTTPException(
                status_code=502, detail="Unable to query FCS API instruments."
            ) from exc

        forex_symbols = available.get("forex", set())
        commodity_symbols = available.get("commodity", set())
        wanted = [
            item
            for market in (Market.FOREX, Market.METALS)
            for item in INSTRUMENTS[market]
        ]

        requested = []
        for item in wanted:
            if item.market == Market.FOREX:
                direct = item.symbol in forex_symbols
                mode = "direct" if direct else "unavailable"
            else:
                aliases = signal_service.market_data.fcs._metal_aliases(item)
                direct = any(symbol in commodity_symbols for symbol in aliases)
                derived = item.id in signal_service.market_data.fcs.DERIVED_METALS
                mode = "direct" if direct else ("derived" if derived else "unavailable")
            requested.append({
                "market": item.market.value,
                "id": item.id,
                "label": item.label,
                "symbol": item.symbol,
                "available": mode != "unavailable",
                "mode": mode,
            })

        return {
            "provider": "FCS API",
            "requested": requested,
        }

    @app.get("/api/admin/cache")
    async def cache_status(
        x_backend_key: str | None = Header(default=None, alias="X-Backend-Key"),
    ) -> dict:
        require_backend_key(x_backend_key)
        return {
            **cache.stats(),
            "twelve_data_price_ttl_seconds": settings.price_cache_seconds,
            "fcs_price_ttl_seconds": settings.fcs_price_cache_seconds,
            "m15_strategy": "candles and signals expire at the next M15 close + grace period",
        }

    return app
