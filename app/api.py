from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qsl, urlsplit

from aiogram import Bot
from aiogram.types import BufferedInputFile
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from .auth import MiniAppAuthError, MiniAppUser, validate_telegram_init_data
from .core import (
    INSTRUMENTS,
    MARKET_TITLES,
    AccessStore,
    CapitalInstrumentUnavailable,
    Market,
    MarketDataError,
    MarketDataRateLimitError,
    Settings,
    SharedMarketCache,
    Signal,
    SignalService,
    admin_payout_review_keyboard,
    admin_review_keyboard,
    get_instrument,
    payment_amount,
)

logger = logging.getLogger(__name__)


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


class ReferralPayoutRequest(BaseModel):
    amount_usdt: float
    wallet_address: str


MAX_RECEIPT_BYTES = 8 * 1024 * 1024


def _payment_row_payload(row) -> dict | None:
    if row is None:
        return None
    keys = set(row.keys())
    return {
        "id": int(row["id"]),
        "plan_key": row["plan_key"],
        "currency": row["currency"],
        "amount": row["amount_text"],
        "status": row["status"],
        "created_at": row["created_at"],
        "reviewed_at": row["reviewed_at"],
        "provider": row["provider"] if "provider" in keys and row["provider"] else "manual",
        "provider_status": row["provider_status"] if "provider_status" in keys else None,
        "provider_payment_id": row["provider_payment_id"] if "provider_payment_id" in keys else None,
        "price_amount_usd": row["price_amount_usd"] if "price_amount_usd" in keys else None,
        "pay_amount": row["pay_amount"] if "pay_amount" in keys else None,
        "pay_currency": row["pay_currency"] if "pay_currency" in keys else None,
        "address": row["pay_address"] if "pay_address" in keys else None,
    }



class NowPaymentsError(RuntimeError):
    pass


def _nowpayments_request(
    settings: Settings,
    method: str,
    path: str,
    payload: dict | None = None,
) -> dict:
    if not settings.nowpayments_api_key:
        raise NowPaymentsError("NOWPayments is not configured.")
    url = f"{settings.nowpayments_base_url}{path}"
    data = None
    headers = {
        "Accept": "application/json",
        "x-api-key": settings.nowpayments_api_key,
    }
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(request, timeout=settings.request_timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")[:1000]
        except Exception:
            detail = str(exc)
        raise NowPaymentsError(f"NOWPayments HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise NowPaymentsError("NOWPayments is temporarily unavailable.") from exc
    try:
        result = json.loads(body) if body else {}
    except json.JSONDecodeError as exc:
        raise NowPaymentsError("NOWPayments returned an invalid response.") from exc
    if isinstance(result, dict) and result.get("message") and not result.get("payment_id") and method.upper() == "POST":
        raise NowPaymentsError(str(result.get("message")))
    return result if isinstance(result, dict) else {}


def _nowpayments_signature(payload: dict, secret: str) -> str:
    # NOWPayments signs canonical JSON with alphabetically sorted object keys.
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hmac.new(secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha512).hexdigest()


def _verify_nowpayments_ipn(payload: dict, signature: str | None, secret: str) -> bool:
    if not signature or not secret:
        return False
    expected = _nowpayments_signature(payload, secret)
    return hmac.compare_digest(expected.lower(), signature.strip().lower())


def _provider_terminal_failure(status: str | None) -> bool:
    return str(status or "").lower() in {"failed", "expired", "refunded"}


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
        version="3.4.0",
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
            logger.warning(
                "Mini App authentication failed: %s",
                exc,
            )
            raise HTTPException(status_code=401, detail=str(exc)) from exc

        access_store.touch_user(user.telegram_id, user.username, user.first_name)
        # Count Mini App opens only after Telegram initData has been verified.
        # /start in the bot registers a user but does not inflate this funnel step.
        access_store.record_miniapp_opened(user.telegram_id)

        # Telegram includes start_param inside the signed initData when the Mini
        # App is opened through ?startapp=ref_CODE. Because initData was already
        # HMAC-validated above, this referral parameter is trusted server-side.
        try:
            signed_params = dict(parse_qsl(init_data, keep_blank_values=True))
            start_param = signed_params.get("start_param")
            if start_param:
                access_store.attach_referrer(user.telegram_id, start_param)
        except Exception:
            logger.exception("Could not attach Mini App referral parameter")

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
            "referral_program_unlocked": bool(is_admin or info.active),
        }

    def require_active(identity: ApiIdentity) -> None:
        if identity.telegram_id == settings.admin_telegram_id:
            return
        if not access_store.access_info(identity.telegram_id).active:
            raise HTTPException(status_code=403, detail="Active subscription required.")

    def nowpayments_enabled() -> bool:
        return bool(
            settings.nowpayments_api_key
            and settings.nowpayments_ipn_secret
            and settings.nowpayments_ipn_url
        )

    async def finalize_nowpayments(payload: dict) -> tuple[dict | None, bool]:
        payment, newly_activated = access_store.update_nowpayments_payment(payload)
        if payment is None:
            return None, False
        if newly_activated:
            payment_id = int(payment["id"])
            try:
                access_store.create_referral_reward_for_payment(
                    payment_id,
                    settings.referral_default_percent,
                    settings.monthly_usd,
                    settings.lifetime_usd,
                )
            except Exception:
                logger.exception("Could not create referral reward for NOWPayments payment %s", payment_id)
            try:
                info = access_store.access_info(int(payment["telegram_id"]))
                plan_text = "Lifetime" if payment["plan_key"] == "lifetime" else "30-day"
                expiry = _iso(info.expires_at) if info.expires_at else None
                expiry_line = f"\nExpires: <b>{expiry[:10]}</b>" if expiry else ""
                await bot.send_message(
                    int(payment["telegram_id"]),
                    "✅ <b>Payment confirmed automatically!</b>\n\n"
                    f"Your <b>{plan_text}</b> MENTAL TRADER access is active.{expiry_line}\n\n"
                    "Open the Mini App to use trading signals.",
                    parse_mode="HTML",
                )
            except Exception:
                logger.exception("Could not notify user about NOWPayments activation")
        return _payment_row_payload(payment), newly_activated

    @app.get("/health")
    async def health() -> dict:
        return {
            "ok": True,
            "service": "mental-trader-backend",
            "version": "3.5.0",
            "timeframe": settings.timeframe_label,
            "miniapp_configured": bool(settings.miniapp_url),
            "cors_origin": origin or None,
            "providers": {
                "forex": signal_service.market_data.capital.provider_label,
                "metals": signal_service.market_data.capital.provider_label,
                "crypto": "Twelve Data",
                "nasdaq": "Twelve Data",
            },
            "payments": {
                "provider": "NOWPayments" if settings.nowpayments_api_key else "manual",
                "nowpayments_configured": bool(
                    settings.nowpayments_api_key
                    and settings.nowpayments_ipn_secret
                    and settings.nowpayments_ipn_url
                ),
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
        except CapitalInstrumentUnavailable as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except MarketDataRateLimitError as exc:
            raise HTTPException(
                status_code=429,
                detail="The market-data request limit was reached. Please wait a minute.",
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
        except CapitalInstrumentUnavailable as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except MarketDataRateLimitError as exc:
            raise HTTPException(
                status_code=429,
                detail="The market-data request limit was reached. Please wait a minute.",
            ) from exc
        except (MarketDataError, ValueError) as exc:
            raise HTTPException(
                status_code=502, detail="Unable to calculate signal right now."
            ) from exc
        return _signal_payload(result, cached)

    @app.get("/api/payment/options")
    async def payment_options(
        x_telegram_init_data: str | None = Header(default=None, alias="X-Telegram-Init-Data"),
        x_backend_key: str | None = Header(default=None, alias="X-Backend-Key"),
        telegram_id: int | None = Query(default=None, ge=1),
    ) -> dict:
        identity = resolve_identity(x_telegram_init_data, x_backend_key, telegram_id)
        latest = access_store.latest_payment(identity.telegram_id)
        automated = nowpayments_enabled()
        return {
            "payment_mode": "nowpayments" if automated else "manual",
            "provider": "NOWPayments" if automated else "Manual verification",
            "plans": [
                {"id": "monthly", "name": "Monthly", "usd": settings.monthly_usd, "duration": "30 days"},
                {"id": "lifetime", "name": "Lifetime", "usd": settings.lifetime_usd, "duration": "Permanent"},
            ],
            "methods": [
                {
                    "id": "usdt", "name": "USDT TRC20", "network": "TRC20",
                    "available": automated or bool(settings.usdt_trc20_address),
                },
                {
                    "id": "btc", "name": "BTC", "network": "Bitcoin",
                    "available": automated or bool(settings.btc_address),
                },
            ],
            "latest_payment": _payment_row_payload(latest),
        }

    @app.post("/api/payment/intent")
    async def create_payment_intent(
        payload: PaymentIntentRequest,
        x_telegram_init_data: str | None = Header(default=None, alias="X-Telegram-Init-Data"),
        x_backend_key: str | None = Header(default=None, alias="X-Backend-Key"),
        telegram_id: int | None = Query(default=None, ge=1),
    ) -> dict:
        identity = resolve_identity(x_telegram_init_data, x_backend_key, telegram_id)
        if identity.telegram_id != settings.admin_telegram_id and access_store.access_info(identity.telegram_id).active:
            raise HTTPException(status_code=409, detail="Your access is already active.")

        latest = access_store.latest_payment(identity.telegram_id)
        if latest is not None and latest["status"] == "pending":
            same_selection = latest["plan_key"] == payload.plan_key and latest["currency"] == payload.currency
            if same_selection and "provider" in latest.keys() and latest["provider"] == "nowpayments":
                return {
                    "mode": "nowpayments",
                    "local_payment_id": int(latest["id"]),
                    "provider_payment_id": latest["provider_payment_id"],
                    "plan_key": latest["plan_key"],
                    "currency": latest["currency"],
                    "method": "USDT TRC20" if latest["currency"] == "usdt" else "BTC",
                    "network": "TRC20" if latest["currency"] == "usdt" else "Bitcoin",
                    "amount": latest["amount_text"],
                    "price_usd": latest["price_amount_usd"],
                    "address": latest["pay_address"],
                    "provider_status": latest["provider_status"],
                }
            raise HTTPException(status_code=409, detail="A payment is already in progress.")

        if payload.plan_key == "monthly":
            price_usd = float(settings.monthly_usd)
        elif payload.plan_key == "lifetime":
            price_usd = float(settings.lifetime_usd)
        else:
            raise HTTPException(status_code=400, detail="Invalid payment selection.")

        if nowpayments_enabled():
            pay_currency = (
                settings.nowpayments_usdt_currency
                if payload.currency == "usdt"
                else settings.nowpayments_btc_currency
            )
            method_name = "USDT TRC20" if payload.currency == "usdt" else "BTC"
            network = "TRC20" if payload.currency == "usdt" else "Bitcoin"
            order_id = (
                f"mt-{identity.telegram_id}-{payload.plan_key[:1]}-"
                f"{int(time.time())}-{secrets.token_hex(3)}"
            )
            provider_request = {
                "price_amount": round(price_usd, 2),
                "price_currency": "usd",
                "pay_currency": pay_currency,
                "ipn_callback_url": settings.nowpayments_ipn_url,
                "order_id": order_id,
                "order_description": f"MENTAL TRADER {'Monthly 30 days' if payload.plan_key == 'monthly' else 'Lifetime'}",
            }
            try:
                provider = await asyncio.to_thread(
                    _nowpayments_request, settings, "POST", "/payment", provider_request
                )
                payment = access_store.record_nowpayments_payment(
                    identity.telegram_id, identity.username, identity.first_name,
                    payload.plan_key, payload.currency, provider, order_id, price_usd,
                )
            except (NowPaymentsError, ValueError) as exc:
                logger.exception("NOWPayments create payment failed")
                raise HTTPException(
                    status_code=502,
                    detail="Unable to create the crypto payment right now. Please try again.",
                ) from exc
            return {
                "mode": "nowpayments",
                "local_payment_id": int(payment["id"]),
                "provider_payment_id": payment["provider_payment_id"],
                "plan_key": payload.plan_key,
                "currency": payload.currency,
                "method": method_name,
                "network": network,
                "amount": payment["amount_text"],
                "price_usd": price_usd,
                "address": payment["pay_address"],
                "provider_status": payment["provider_status"],
            }

        # Manual fallback for installations that have not configured NOWPayments.
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
            "mode": "manual",
            "plan_key": payload.plan_key,
            "currency": payload.currency,
            "method": method_name,
            "network": network,
            "amount": amount,
            "price_usd": price_usd,
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

    @app.post("/api/payment/cancel")
    async def cancel_payment(
        x_telegram_init_data: str | None = Header(default=None, alias="X-Telegram-Init-Data"),
        x_backend_key: str | None = Header(default=None, alias="X-Backend-Key"),
        telegram_id: int | None = Query(default=None, ge=1),
    ) -> dict:
        identity = resolve_identity(x_telegram_init_data, x_backend_key, telegram_id)
        payment = access_store.cancel_latest_pending_payment(identity.telegram_id)

        if payment is None:
            latest = access_store.latest_payment(identity.telegram_id)
            if latest is not None and latest["status"] == "cancelled":
                return {
                    "ok": True,
                    "already_cancelled": True,
                    "payment": _payment_row_payload(latest),
                }
            raise HTTPException(status_code=409, detail="No pending payment to cancel.")

        logger.info(
            "User %s cancelled local payment attempt #%s",
            identity.telegram_id,
            payment["id"],
        )
        return {
            "ok": True,
            "payment": _payment_row_payload(payment),
        }

    @app.get("/api/payment/status")
    async def payment_status(
        x_telegram_init_data: str | None = Header(default=None, alias="X-Telegram-Init-Data"),
        x_backend_key: str | None = Header(default=None, alias="X-Backend-Key"),
        telegram_id: int | None = Query(default=None, ge=1),
    ) -> dict:
        identity = resolve_identity(x_telegram_init_data, x_backend_key, telegram_id)
        latest = access_store.latest_payment(identity.telegram_id)
        if (
            latest is not None
            and "provider" in latest.keys()
            and latest["provider"] == "nowpayments"
            and latest["status"] == "pending"
            and latest["provider_payment_id"]
            and settings.nowpayments_api_key
        ):
            try:
                provider = await asyncio.to_thread(
                    _nowpayments_request,
                    settings,
                    "GET",
                    f"/payment/{latest['provider_payment_id']}",
                    None,
                )
                await finalize_nowpayments(provider)
                latest = access_store.latest_payment(identity.telegram_id)
            except NowPaymentsError:
                logger.warning("Could not synchronize NOWPayments status for %s", latest["provider_payment_id"])
        return {
            "access": access_payload(identity),
            "payment": _payment_row_payload(latest),
        }

    @app.post("/api/payments/nowpayments/ipn", include_in_schema=False)
    async def nowpayments_ipn(
        request: Request,
        x_nowpayments_sig: str | None = Header(default=None, alias="x-nowpayments-sig"),
    ) -> dict:
        if not settings.nowpayments_ipn_secret:
            raise HTTPException(status_code=503, detail="NOWPayments IPN is not configured.")
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="Invalid IPN payload.")
        if not _verify_nowpayments_ipn(payload, x_nowpayments_sig, settings.nowpayments_ipn_secret):
            logger.warning("Rejected NOWPayments IPN with invalid signature")
            raise HTTPException(status_code=401, detail="Invalid NOWPayments signature.")
        payment, newly_activated = await finalize_nowpayments(payload)
        if payment is None:
            logger.warning(
                "NOWPayments IPN references an unknown payment: %s",
                payload.get("payment_id") or payload.get("order_id"),
            )
        return {"ok": True, "activated": newly_activated}

    @app.get("/api/referral")
    async def referral_dashboard(
        x_telegram_init_data: str | None = Header(
            default=None, alias="X-Telegram-Init-Data"
        ),
        x_backend_key: str | None = Header(default=None, alias="X-Backend-Key"),
        telegram_id: int | None = Query(default=None, ge=1),
    ) -> dict:
        identity = resolve_identity(x_telegram_init_data, x_backend_key, telegram_id)
        require_active(identity)
        dashboard = access_store.referral_dashboard(
            identity.telegram_id,
            settings.referral_default_percent,
            settings.referral_min_payout_usdt,
        )
        bot_username = (settings.bot_username or "mental_traderbot").lstrip("@")
        dashboard["referral_link"] = (
            f"https://t.me/{bot_username}?startapp=ref_{dashboard['referral_code']}"
        )
        dashboard["active_subscription_required"] = True
        dashboard["commission_scope"] = "first_approved_purchase"
        return dashboard

    @app.post("/api/referral/payout")
    async def referral_payout(
        payload: ReferralPayoutRequest,
        x_telegram_init_data: str | None = Header(
            default=None, alias="X-Telegram-Init-Data"
        ),
        x_backend_key: str | None = Header(default=None, alias="X-Backend-Key"),
        telegram_id: int | None = Query(default=None, ge=1),
    ) -> dict:
        identity = resolve_identity(x_telegram_init_data, x_backend_key, telegram_id)
        require_active(identity)
        wallet = (payload.wallet_address or "").strip()
        # Basic TRC20 address validation. This prevents obvious typos but does
        # not replace the administrator's final payout verification.
        if len(wallet) != 34 or not wallet.startswith("T") or not wallet.isalnum():
            raise HTTPException(
                status_code=400,
                detail="Enter a valid USDT TRC20 wallet address.",
            )
        try:
            payout = access_store.create_payout_request(
                identity.telegram_id,
                payload.amount_usdt,
                wallet,
                settings.referral_min_payout_usdt,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if settings.admin_telegram_id:
            username = f"@{identity.username}" if identity.username else "—"
            try:
                await bot.send_message(
                    settings.admin_telegram_id,
                    "💸 <b>Новый запрос на реферальную выплату</b>\n\n"
                    f"ID выплаты: <code>{payout['id']}</code>\n"
                    f"Пользователь: {username}\n"
                    f"Telegram ID: <code>{identity.telegram_id}</code>\n"
                    f"Сумма: <b>{float(payout['amount_usdt']):.2f} USDT</b>\n"
                    f"Кошелёк TRC20: <code>{wallet}</code>\n\n"
                    "Сначала отправь USDT вручную, затем нажми «Выплачено».",
                    parse_mode="HTML",
                    reply_markup=admin_payout_review_keyboard(int(payout["id"])),
                )
            except Exception:
                logger.exception("Could not notify admin about referral payout")

        return {
            "ok": True,
            "payout_id": int(payout["id"]),
            "status": payout["status"],
            "amount_usdt": float(payout["amount_usdt"]),
        }

    @app.get("/api/admin/capital/availability")
    async def capital_availability(
        x_backend_key: str | None = Header(default=None, alias="X-Backend-Key"),
    ) -> dict:
        require_backend_key(x_backend_key)
        wanted = [
            item
            for market in (Market.FOREX, Market.METALS)
            for item in INSTRUMENTS[market]
        ]
        try:
            requested = await asyncio.to_thread(
                signal_service.market_data.capital.instrument_availability,
                wanted,
            )
        except MarketDataError as exc:
            raise HTTPException(
                status_code=502, detail="Unable to query Capital.com instruments."
            ) from exc

        return {
            "provider": signal_service.market_data.capital.provider_label,
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
            "capital_price_ttl_seconds": settings.capital_price_cache_seconds,
            "m15_strategy": "candles and signals expire at the next M15 close + grace period",
        }

    return app
