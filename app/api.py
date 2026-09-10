from __future__ import annotations

import hmac
from dataclasses import dataclass
from datetime import timezone
from urllib.parse import urlsplit

from fastapi import FastAPI, Header, HTTPException, Query
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
    get_instrument,
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


def create_api(
    settings: Settings,
    signal_service: SignalService,
    access_store: AccessStore,
    cache: SharedMarketCache,
) -> FastAPI:
    app = FastAPI(
        title="MENTAL TRADER Backend",
        version="2.1.0",
        description="Telegram bot + secure Telegram Mini App backend.",
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
            allow_methods=["GET", "OPTIONS"],
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
            "version": "2.1.0",
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
