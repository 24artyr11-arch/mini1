from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import secrets
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from html import escape
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np
import pandas as pd
from aiogram import Bot, Dispatcher, F, Router
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message, WebAppInfo
from dotenv import load_dotenv

load_dotenv()


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Settings:
    telegram_bot_token: str
    twelve_data_api_key: str
    twelve_data_base_url: str = "https://api.twelvedata.com"

    capital_identifier: str = ""
    capital_api_key: str = ""
    capital_api_password: str = ""
    capital_base_url: str = "https://demo-api-capital.backend-capital.com"

    api_host: str = "0.0.0.0"
    api_port: int = 3000
    backend_api_key: str = ""
    miniapp_url: str = ""
    miniapp_origin: str = ""
    miniapp_auth_max_age_seconds: int = 86400
    price_cache_seconds: int = 15
    capital_price_cache_seconds: int = 60
    candle_cache_grace_seconds: int = 4

    timeframe: str = "15min"
    timeframe_label: str = "M15"
    bars_count: int = 350
    request_timeout_seconds: int = 15

    # Adaptive decision threshold. The strategy targets about 70% directional
    # signals and 30% WAIT across recent M15 market states.
    signal_target_rate: float = 0.70
    adaptive_lookback: int = 96
    min_score: int = 40
    strong_score: int = 80

    # Trade-level construction. These are not position-size recommendations.
    min_stop_atr: float = 0.90
    max_stop_atr: float = 1.80
    tp1_rr: float = 1.50
    tp2_rr: float = 2.70

    # Manual access/payment review settings.
    admin_telegram_id: int = 0
    database_path: str = "bot.db"
    usdt_trc20_address: str = ""
    btc_address: str = ""
    monthly_usd: float = 50.0
    lifetime_usd: float = 180.0
    btc_usd_rate: float = 77000.0

    # Referral program. Available to active subscribers only.
    bot_username: str = "mental_traderbot"
    referral_default_percent: float = 30.0
    referral_min_payout_usdt: float = 20.0

    # NOWPayments: automated crypto checkout. Secrets stay server-side.
    nowpayments_api_key: str = ""
    nowpayments_ipn_secret: str = ""
    nowpayments_base_url: str = "https://api.nowpayments.io/v1"
    nowpayments_ipn_url: str = ""
    nowpayments_usdt_currency: str = "usdttrc20"
    nowpayments_btc_currency: str = "btc"


def _resolve_database_path() -> str:
    """
    Keep user/referral/payment/statistics data on persistent storage in production.

    DATABASE_PATH always wins when explicitly configured. On Bothost (DOMAIN is
    present), /app/data is used automatically even if DATABASE_PATH was omitted.
    Local development keeps using ./bot.db.
    """
    explicit = (os.getenv("DATABASE_PATH") or "").strip()
    if explicit:
        return explicit

    persistent_dir = Path("/app/data")
    production_host = bool((os.getenv("DOMAIN") or "").strip())

    if production_host or persistent_dir.exists():
        try:
            persistent_dir.mkdir(parents=True, exist_ok=True)
            return str(persistent_dir / "bot.db")
        except OSError:
            # Local/non-container environments may not permit /app/data.
            pass

    return "bot.db"


def get_settings() -> Settings:
    token = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "Telegram token is missing. Set BOT_TOKEN (recommended) or TELEGRAM_BOT_TOKEN."
        )

    api_key = os.getenv("TWELVE_DATA_API_KEY")
    if not api_key:
        raise RuntimeError("Twelve Data API key is missing. Set TWELVE_DATA_API_KEY.")

    capital_identifier = (os.getenv("CAPITAL_IDENTIFIER") or "").strip()
    capital_api_key = (os.getenv("CAPITAL_API_KEY") or "").strip()
    capital_api_password = (os.getenv("CAPITAL_API_PASSWORD") or "").strip()
    missing_capital = [
        name
        for name, value in (
            ("CAPITAL_IDENTIFIER", capital_identifier),
            ("CAPITAL_API_KEY", capital_api_key),
            ("CAPITAL_API_PASSWORD", capital_api_password),
        )
        if not value
    ]
    if missing_capital:
        raise RuntimeError(
            "Capital.com credentials are missing. Set " + ", ".join(missing_capital) + "."
        )

    return Settings(
        telegram_bot_token=token,
        twelve_data_api_key=api_key,
        twelve_data_base_url=os.getenv(
            "TWELVE_DATA_BASE_URL", "https://api.twelvedata.com"
        ).rstrip("/"),
        capital_identifier=capital_identifier,
        capital_api_key=capital_api_key,
        capital_api_password=capital_api_password,
        capital_base_url=os.getenv(
            "CAPITAL_BASE_URL", "https://demo-api-capital.backend-capital.com"
        ).rstrip("/"),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("PORT", os.getenv("API_PORT", "3000"))),
        backend_api_key=(os.getenv("BACKEND_API_KEY") or "").strip(),
        miniapp_url=(os.getenv("MINIAPP_URL") or "").strip().rstrip("/"),
        miniapp_origin=(os.getenv("MINIAPP_ORIGIN") or "").strip().rstrip("/"),
        miniapp_auth_max_age_seconds=int(os.getenv("MINIAPP_AUTH_MAX_AGE_SECONDS", "86400")),
        price_cache_seconds=int(os.getenv("PRICE_CACHE_SECONDS", "15")),
        capital_price_cache_seconds=int(os.getenv("CAPITAL_PRICE_CACHE_SECONDS", "60")),
        candle_cache_grace_seconds=int(os.getenv("CANDLE_CACHE_GRACE_SECONDS", "4")),
        bars_count=int(os.getenv("BARS_COUNT", "350")),
        request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "15")),
        signal_target_rate=float(os.getenv("TARGET_SIGNAL_RATE", "0.70")),
        adaptive_lookback=int(os.getenv("ADAPTIVE_LOOKBACK", "96")),
        min_score=int(os.getenv("MIN_SCORE", "40")),
        strong_score=int(os.getenv("STRONG_SCORE", "80")),
        admin_telegram_id=int(os.getenv("ADMIN_TELEGRAM_ID", "0")),
        database_path=_resolve_database_path(),
        usdt_trc20_address=os.getenv("USDT_TRC20_ADDRESS", "").strip(),
        btc_address=os.getenv("BTC_ADDRESS", "").strip(),
        monthly_usd=float(os.getenv("MONTHLY_USD", "50")),
        lifetime_usd=float(os.getenv("LIFETIME_USD", "180")),
        btc_usd_rate=float(os.getenv("BTC_USD_RATE", "77000")),
        bot_username=(os.getenv("BOT_USERNAME") or "mental_traderbot").strip().lstrip("@"),
        referral_default_percent=max(0.0, min(100.0, float(os.getenv("REFERRAL_DEFAULT_PERCENT", "30")))),
        referral_min_payout_usdt=max(0.0, float(os.getenv("REFERRAL_MIN_PAYOUT_USDT", "20"))),
        nowpayments_api_key=(os.getenv("NOWPAYMENTS_API_KEY") or "").strip(),
        nowpayments_ipn_secret=(os.getenv("NOWPAYMENTS_IPN_SECRET") or "").strip(),
        nowpayments_base_url=(os.getenv("NOWPAYMENTS_BASE_URL") or "https://api.nowpayments.io/v1").strip().rstrip("/"),
        nowpayments_ipn_url=(
            (os.getenv("NOWPAYMENTS_IPN_URL") or "").strip()
            or (
                ((os.getenv("TELEGRAM_WEBHOOK_URL") or "").strip().split("/api/telegram/webhook", 1)[0].rstrip("/")
                 + "/api/payments/nowpayments/ipn")
                if (os.getenv("TELEGRAM_WEBHOOK_URL") or "").strip()
                else (
                    f"https://{(os.getenv('DOMAIN') or '').strip().removeprefix('https://').removeprefix('http://').rstrip('/')}/api/payments/nowpayments/ipn"
                    if (os.getenv("DOMAIN") or "").strip()
                    else ""
                )
            )
        ),
        nowpayments_usdt_currency=(os.getenv("NOWPAYMENTS_USDT_CURRENCY") or "usdttrc20").strip().lower(),
        nowpayments_btc_currency=(os.getenv("NOWPAYMENTS_BTC_CURRENCY") or "btc").strip().lower(),
    )


# -----------------------------------------------------------------------------
# Models and instruments
# -----------------------------------------------------------------------------

class Market(StrEnum):
    FOREX = "forex"
    METALS = "metals"
    CRYPTO = "crypto"
    NASDAQ = "nasdaq"


class Direction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"


@dataclass(frozen=True, slots=True)
class Instrument:
    id: str
    market: Market
    label: str
    display_name: str
    symbol: str
    digits: int
    exchange: str | None = None


@dataclass(slots=True)
class LivePrice:
    symbol: str
    price: float
    digits: int
    fetched_at: datetime
    data_source: str = ""


@dataclass(slots=True)
class Signal:
    instrument: Instrument
    provider_symbol: str
    direction: Direction
    current_price: float
    digits: int
    timeframe: str
    score: int
    price_fetched_at: datetime
    data_source: str = ""
    entry: float | None = None
    stop_loss: float | None = None
    take_profit_1: float | None = None
    take_profit_2: float | None = None
    risk_reward: float | None = None
    atr: float | None = None
    rsi: float | None = None
    ema20: float | None = None
    ema50: float | None = None
    ema200: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    support: float | None = None
    resistance: float | None = None
    setup: str | None = None
    reasons: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)


INSTRUMENTS: dict[Market, tuple[Instrument, ...]] = {
    Market.FOREX: (
        Instrument("eurusd", Market.FOREX, "EUR/USD", "Euro / US Dollar", "EURUSD", 5),
        Instrument("gbpusd", Market.FOREX, "GBP/USD", "British Pound / US Dollar", "GBPUSD", 5),
        Instrument("usdjpy", Market.FOREX, "USD/JPY", "US Dollar / Japanese Yen", "USDJPY", 3),
        Instrument("usdchf", Market.FOREX, "USD/CHF", "US Dollar / Swiss Franc", "USDCHF", 5),
        Instrument("audusd", Market.FOREX, "AUD/USD", "Australian Dollar / US Dollar", "AUDUSD", 5),
        Instrument("usdcad", Market.FOREX, "USD/CAD", "US Dollar / Canadian Dollar", "USDCAD", 5),
        Instrument("nzdusd", Market.FOREX, "NZD/USD", "New Zealand Dollar / US Dollar", "NZDUSD", 5),
        Instrument("eurgbp", Market.FOREX, "EUR/GBP", "Euro / British Pound", "EURGBP", 5),
        Instrument("eurjpy", Market.FOREX, "EUR/JPY", "Euro / Japanese Yen", "EURJPY", 3),
        Instrument("gbpjpy", Market.FOREX, "GBP/JPY", "British Pound / Japanese Yen", "GBPJPY", 3),
    ),
    Market.METALS: (
        Instrument("xauusd", Market.METALS, "Gold / USD", "Gold / US Dollar", "XAUUSD", 3),
        Instrument("xagusd", Market.METALS, "Silver / USD", "Silver / US Dollar", "XAGUSD", 4),
        Instrument("xptusd", Market.METALS, "Platinum / USD", "Platinum / US Dollar", "XPTUSD", 2),
        Instrument("xpdusd", Market.METALS, "Palladium / USD", "Palladium / US Dollar", "XPDUSD", 2),
        Instrument("xaueur", Market.METALS, "Gold / EUR", "Gold / Euro", "XAUEUR", 3),
        Instrument("xaugbp", Market.METALS, "Gold / GBP", "Gold / British Pound", "XAUGBP", 3),
        Instrument("xageur", Market.METALS, "Silver / EUR", "Silver / Euro", "XAGEUR", 4),
        Instrument("xaggbp", Market.METALS, "Silver / GBP", "Silver / British Pound", "XAGGBP", 4),
        Instrument("xauaud", Market.METALS, "Gold / AUD", "Gold / Australian Dollar", "XAUAUD", 3),
        Instrument("xaujpy", Market.METALS, "Gold / JPY", "Gold / Japanese Yen", "XAUJPY", 2),
    ),
    Market.CRYPTO: (
        # Popular, liquid crypto pairs with broad Twelve Data coverage.
        Instrument("btcusd", Market.CRYPTO, "BTC/USD", "Bitcoin / US Dollar", "BTC/USD", 2),
        Instrument("ethusd", Market.CRYPTO, "ETH/USD", "Ethereum / US Dollar", "ETH/USD", 2),
        Instrument("bnbusd", Market.CRYPTO, "BNB/USD", "BNB / US Dollar", "BNB/USD", 2),
        Instrument("xrpusd", Market.CRYPTO, "XRP/USD", "XRP / US Dollar", "XRP/USD", 4),
        Instrument("solusd", Market.CRYPTO, "SOL/USD", "Solana / US Dollar", "SOL/USD", 2),
        Instrument("trxusd", Market.CRYPTO, "TRX/USD", "TRON / US Dollar", "TRX/USD", 5),
        Instrument("dogeusd", Market.CRYPTO, "DOGE/USD", "Dogecoin / US Dollar", "DOGE/USD", 5),
        Instrument("xmrusd", Market.CRYPTO, "XMR/USD", "Monero / US Dollar", "XMR/USD", 2),
        Instrument("linkusd", Market.CRYPTO, "LINK/USD", "Chainlink / US Dollar", "LINK/USD", 3),
        Instrument("adausd", Market.CRYPTO, "ADA/USD", "Cardano / US Dollar", "ADA/USD", 4),
    ),
    Market.NASDAQ: (
        # QQQ is used instead of a broker-specific NAS100 CFD symbol.
        Instrument("qqq", Market.NASDAQ, "QQQ", "Invesco QQQ (Nasdaq-100 tracker)", "QQQ", 2, "NASDAQ"),
        Instrument("aapl", Market.NASDAQ, "AAPL", "Apple", "AAPL", 2, "NASDAQ"),
        Instrument("msft", Market.NASDAQ, "MSFT", "Microsoft", "MSFT", 2, "NASDAQ"),
        Instrument("nvda", Market.NASDAQ, "NVDA", "NVIDIA", "NVDA", 2, "NASDAQ"),
        Instrument("amzn", Market.NASDAQ, "AMZN", "Amazon", "AMZN", 2, "NASDAQ"),
        Instrument("meta", Market.NASDAQ, "META", "Meta Platforms", "META", 2, "NASDAQ"),
        Instrument("googl", Market.NASDAQ, "GOOGL", "Alphabet", "GOOGL", 2, "NASDAQ"),
        Instrument("tsla", Market.NASDAQ, "TSLA", "Tesla", "TSLA", 2, "NASDAQ"),
        Instrument("avgo", Market.NASDAQ, "AVGO", "Broadcom", "AVGO", 2, "NASDAQ"),
        Instrument("nflx", Market.NASDAQ, "NFLX", "Netflix", "NFLX", 2, "NASDAQ"),
    ),
}

MARKET_TITLES = {
    Market.FOREX: "💱 Forex",
    Market.METALS: "🥇 Metals",
    Market.CRYPTO: "₿ Crypto",
    Market.NASDAQ: "📈 Nasdaq",
}


def get_instrument(market: Market, instrument_id: str) -> Instrument | None:
    return next((item for item in INSTRUMENTS[market] if item.id == instrument_id), None)


# -----------------------------------------------------------------------------
# Twelve Data connector
# -----------------------------------------------------------------------------

class MarketDataError(RuntimeError):
    pass


class MarketDataRateLimitError(MarketDataError):
    pass


class TwelveDataError(MarketDataError):
    pass


class TwelveDataRateLimitError(TwelveDataError, MarketDataRateLimitError):
    pass


RATE_LIMIT_MESSAGE = "The market-data request limit was reached. Please wait a minute."


class TwelveDataClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _request_json(self, endpoint: str, params: dict[str, object]) -> dict:
        query = {k: v for k, v in params.items() if v is not None}
        query["apikey"] = self.settings.twelve_data_api_key
        url = f"{self.settings.twelve_data_base_url}/{endpoint.lstrip('/')}?{urllib.parse.urlencode(query)}"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "telegram-market-signal-bot/1.0",
            },
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self.settings.request_timeout_seconds
            ) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429:
                raise TwelveDataRateLimitError(RATE_LIMIT_MESSAGE) from exc
            raise TwelveDataError(f"Twelve Data HTTP {exc.code}: {body[:300]}") from exc
        except urllib.error.URLError as exc:
            raise TwelveDataError(f"Could not reach Twelve Data: {exc.reason}") from exc
        except TimeoutError as exc:
            raise TwelveDataError("Twelve Data request timed out.") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TwelveDataError("Twelve Data returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise TwelveDataError("Unexpected Twelve Data response format.")

        if payload.get("status") == "error" or "code" in payload and payload.get("message"):
            code = payload.get("code", "API")
            message = payload.get("message", "Unknown Twelve Data error")
            if str(code) == "429":
                raise TwelveDataRateLimitError(RATE_LIMIT_MESSAGE)
            raise TwelveDataError(f"Twelve Data {code}: {message}")

        return payload

    def get_price(self, instrument: Instrument) -> LivePrice:
        params: dict[str, object] = {"symbol": instrument.symbol}
        if instrument.exchange:
            params["exchange"] = instrument.exchange

        payload = self._request_json("price", params)
        value = payload.get("price")
        try:
            price = float(value)
        except (TypeError, ValueError) as exc:
            raise TwelveDataError(
                f"No usable live price returned for {instrument.label}."
            ) from exc

        if price <= 0:
            raise TwelveDataError(f"Invalid live price returned for {instrument.label}.")

        return LivePrice(
            symbol=instrument.symbol,
            price=price,
            digits=instrument.digits,
            fetched_at=datetime.now(timezone.utc),
            data_source="Twelve Data",
        )

    def get_closed_bars(self, instrument: Instrument, count: int) -> pd.DataFrame:
        params: dict[str, object] = {
            "symbol": instrument.symbol,
            "interval": self.settings.timeframe,
            "outputsize": count + 2,
            "timezone": "UTC",
            "format": "JSON",
        }
        if instrument.exchange:
            params["exchange"] = instrument.exchange

        payload = self._request_json("time_series", params)
        values = payload.get("values")
        if not isinstance(values, list) or not values:
            raise TwelveDataError(
                f"No M15 candle data returned for {instrument.label}."
            )

        rows: list[dict[str, object]] = []
        for item in values:
            if not isinstance(item, dict):
                continue
            try:
                rows.append(
                    {
                        "time": pd.to_datetime(item["datetime"], utc=True),
                        "open": float(item["open"]),
                        "high": float(item["high"]),
                        "low": float(item["low"]),
                        "close": float(item["close"]),
                        "volume": float(item.get("volume", 0) or 0),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue

        if not rows:
            raise TwelveDataError(
                f"Twelve Data returned candle data in an unexpected format for {instrument.label}."
            )

        data = pd.DataFrame(rows).sort_values("time").drop_duplicates("time")
        data = data.reset_index(drop=True)

        # Twelve Data time-series can include the currently forming 15-minute bar.
        # Use it only after its 15-minute interval has elapsed.
        if len(data) > 0:
            last_start = data.iloc[-1]["time"]
            now = pd.Timestamp.now(tz="UTC")
            if last_start + pd.Timedelta(minutes=15) > now:
                data = data.iloc[:-1]

        data = data.tail(count).reset_index(drop=True)
        if len(data) < 220:
            raise TwelveDataError(
                f"Only {len(data)} closed M15 candles are available for {instrument.label}; at least 220 are required."
            )
        return data


# -----------------------------------------------------------------------------
# Indicators
# -----------------------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["ema20"] = data["close"].ewm(span=20, adjust=False).mean()
    data["ema50"] = data["close"].ewm(span=50, adjust=False).mean()
    data["ema200"] = data["close"].ewm(span=200, adjust=False).mean()

    delta = data["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["rsi"] = 100 - (100 / (1 + rs))
    data.loc[(avg_loss == 0) & (avg_gain > 0), "rsi"] = 100.0
    data.loc[(avg_loss == 0) & (avg_gain == 0), "rsi"] = 50.0

    ema12 = data["close"].ewm(span=12, adjust=False).mean()
    ema26 = data["close"].ewm(span=26, adjust=False).mean()
    data["macd"] = ema12 - ema26
    data["macd_signal"] = data["macd"].ewm(span=9, adjust=False).mean()
    data["macd_hist"] = data["macd"] - data["macd_signal"]

    previous_close = data["close"].shift(1)
    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["atr"] = true_range.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()
    return data


def swing_levels(
    df: pd.DataFrame, current_price: float, lookback: int = 80, wing: int = 2
) -> tuple[float | None, float | None]:
    recent = df.tail(lookback).reset_index(drop=True)
    highs: list[float] = []
    lows: list[float] = []

    for i in range(wing, len(recent) - wing):
        high = float(recent.loc[i, "high"])
        low = float(recent.loc[i, "low"])
        high_window = recent.loc[i - wing : i + wing, "high"]
        low_window = recent.loc[i - wing : i + wing, "low"]
        if high >= float(high_window.max()):
            highs.append(high)
        if low <= float(low_window.min()):
            lows.append(low)

    supports = [x for x in lows if x < current_price]
    resistances = [x for x in highs if x > current_price]
    return (
        max(supports) if supports else None,
        min(resistances) if resistances else None,
    )


# -----------------------------------------------------------------------------
# Capital.com connector and shared provider errors
# -----------------------------------------------------------------------------

class CapitalError(MarketDataError):
    pass


class CapitalRateLimitError(CapitalError, MarketDataRateLimitError):
    pass


class CapitalInstrumentUnavailable(CapitalError):
    pass


@dataclass(frozen=True, slots=True)
class CapitalSessionTokens:
    cst: str
    security_token: str


class _CapitalHttpError(RuntimeError):
    def __init__(self, status: int, error_code: str = ""):
        self.status = status
        self.error_code = error_code
        label = error_code or "unknown error"
        super().__init__(f"Capital.com HTTP {status}: {label}")


class CapitalClient:
    """Read-only Capital.com market-data adapter for Forex and Metals.

    Only session and market-data endpoints are implemented. The client has no
    order or position methods. Bid/ask values are normalized to midpoint OHLC
    data so the existing signal strategy remains provider-independent.
    """

    METAL_EPICS: dict[str, tuple[str, ...]] = {
        "xauusd": ("GOLD", "XAUUSD"),
        "xagusd": ("SILVER", "XAGUSD"),
        "xptusd": ("PLATINUM", "XPTUSD"),
        "xpdusd": ("PALLADIUM", "XPDUSD"),
    }
    METAL_SEARCH_TERMS: dict[str, str] = {
        "xauusd": "Gold",
        "xagusd": "Silver",
        "xptusd": "Platinum",
        "xpdusd": "Palladium",
    }

    # If Capital.com has no direct cross, build it from synchronized M15 data.
    DERIVED_METALS: dict[str, tuple[str, str, str]] = {
        "xaueur": ("xauusd", "EURUSD", "divide"),
        "xaugbp": ("xauusd", "GBPUSD", "divide"),
        "xageur": ("xagusd", "EURUSD", "divide"),
        "xaggbp": ("xagusd", "GBPUSD", "divide"),
        "xauaud": ("xauusd", "AUDUSD", "divide"),
        "xaujpy": ("xauusd", "USDJPY", "multiply"),
    }

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.capital_base_url.rstrip("/")
        parsed_base_url = urllib.parse.urlsplit(self.base_url)
        allowed_hosts = {
            "api-capital.backend-capital.com",
            "demo-api-capital.backend-capital.com",
        }
        if parsed_base_url.scheme != "https" or parsed_base_url.hostname not in allowed_hosts:
            raise CapitalError(
                "CAPITAL_BASE_URL must be an official HTTPS Capital.com API URL."
            )
        self.provider_label = (
            "Capital.com Demo" if "demo-api-capital" in self.base_url.lower() else "Capital.com"
        )
        self._session: CapitalSessionTokens | None = None
        self._session_last_used = 0.0
        self._last_login_attempt = 0.0
        self._session_lock = threading.RLock()
        self._rate_lock = threading.Lock()
        self._last_request_at = 0.0
        self._epic_lock = threading.Lock()
        self._epic_cache: dict[str, str] = {}

    @staticmethod
    def _error_code(raw: str) -> str:
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return ""
        if not isinstance(payload, dict):
            return ""
        return str(payload.get("errorCode") or payload.get("message") or "")[:160]

    def _throttle(self) -> None:
        # Capital.com documents a maximum of 10 REST requests per second.
        with self._rate_lock:
            now = time.monotonic()
            wait = 0.11 - (now - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = time.monotonic()

    def _http_json(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, object] | None = None,
        payload: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        url = f"{self.base_url}{path}"
        if query:
            clean_query = {key: value for key, value in query.items() if value is not None}
            url += "?" + urllib.parse.urlencode(clean_query)

        request_headers = {
            "Accept": "application/json",
            "User-Agent": "mental-trader-bot/3.5",
            **(headers or {}),
        }
        body = None
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            request_headers["Content-Type"] = "application/json"

        self._throttle()
        request = urllib.request.Request(
            url,
            data=body,
            headers=request_headers,
            method=method.upper(),
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.settings.request_timeout_seconds
            ) as response:
                raw = response.read().decode("utf-8", errors="replace")
                response_headers = {
                    key.lower(): value for key, value in response.headers.items()
                }
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise _CapitalHttpError(exc.code, self._error_code(raw)) from exc
        except urllib.error.URLError as exc:
            raise CapitalError(f"Could not reach Capital.com: {exc.reason}") from exc
        except TimeoutError as exc:
            raise CapitalError("Capital.com request timed out.") from exc

        if not raw.strip():
            return {}, response_headers
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CapitalError("Capital.com returned invalid JSON.") from exc
        if not isinstance(result, dict):
            raise CapitalError("Unexpected Capital.com response format.")
        error_code = str(result.get("errorCode") or "")
        if error_code:
            raise _CapitalHttpError(400, error_code[:160])
        return result, response_headers

    def _login_locked(self) -> None:
        # POST /session has a separate limit of one request per second.
        since_last_login = time.monotonic() - self._last_login_attempt
        if self._last_login_attempt and since_last_login < 1.05:
            time.sleep(1.05 - since_last_login)
        self._last_login_attempt = time.monotonic()
        try:
            _, headers = self._http_json(
                "POST",
                "/api/v1/session",
                payload={
                    "identifier": self.settings.capital_identifier,
                    "password": self.settings.capital_api_password,
                    "encryptedPassword": False,
                },
                headers={"X-CAP-API-KEY": self.settings.capital_api_key},
            )
        except _CapitalHttpError as exc:
            if exc.status == 429:
                raise CapitalRateLimitError(RATE_LIMIT_MESSAGE) from exc
            raise CapitalError(
                "Capital.com authentication failed. Check CAPITAL_IDENTIFIER, "
                "CAPITAL_API_KEY and CAPITAL_API_PASSWORD."
            ) from exc

        cst = (headers.get("cst") or "").strip()
        security_token = (headers.get("x-security-token") or "").strip()
        if not cst or not security_token:
            raise CapitalError(
                "Capital.com created a session without the required authentication tokens."
            )
        self._session = CapitalSessionTokens(cst=cst, security_token=security_token)
        self._session_last_used = time.monotonic()

    def _session_for_request(self) -> CapitalSessionTokens:
        with self._session_lock:
            idle = time.monotonic() - self._session_last_used
            if self._session is None or idle >= 540:
                self._login_locked()
            assert self._session is not None
            return self._session

    def _invalidate_session(self, used: CapitalSessionTokens) -> None:
        with self._session_lock:
            if self._session == used:
                self._session = None
                self._session_last_used = 0.0

    def _touch_session(self, used: CapitalSessionTokens) -> None:
        with self._session_lock:
            if self._session == used:
                self._session_last_used = time.monotonic()

    @staticmethod
    def _is_session_error(exc: _CapitalHttpError) -> bool:
        code = exc.error_code.lower()
        return exc.status in {401, 403} or (
            "security" in code and ("token" in code or "session" in code)
        )

    @staticmethod
    def _raise_provider_error(exc: _CapitalHttpError) -> None:
        code = exc.error_code.lower()
        if exc.status == 429 or "rate" in code or "too-many" in code:
            raise CapitalRateLimitError(RATE_LIMIT_MESSAGE) from exc
        if exc.status == 404 or "epic" in code or "market-not-found" in code:
            raise CapitalInstrumentUnavailable(
                "Capital.com did not return this instrument."
            ) from exc
        detail = exc.error_code or f"HTTP {exc.status}"
        raise CapitalError(f"Capital.com market-data request failed: {detail}.") from exc

    def _request_json(
        self,
        path: str,
        query: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        for attempt in range(2):
            session = self._session_for_request()
            try:
                payload, _ = self._http_json(
                    "GET",
                    path,
                    query=query,
                    headers={
                        "CST": session.cst,
                        "X-SECURITY-TOKEN": session.security_token,
                    },
                )
            except _CapitalHttpError as exc:
                if attempt == 0 and self._is_session_error(exc):
                    self._invalidate_session(session)
                    continue
                self._raise_provider_error(exc)
            self._touch_session(session)
            return payload
        raise CapitalError("Capital.com session could not be refreshed.")

    @staticmethod
    def _normalized(value: object) -> str:
        return "".join(character for character in str(value or "").upper() if character.isalnum())

    @staticmethod
    def _market_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
        markets = payload.get("markets")
        return [item for item in markets or [] if isinstance(item, dict)]

    def _candidate_epics(self, instrument: Instrument) -> tuple[str, ...]:
        if instrument.market == Market.METALS:
            return self.METAL_EPICS.get(instrument.id, (instrument.symbol,))
        return (instrument.symbol,)

    def _market_matches_type(self, item: dict[str, Any], instrument: Instrument) -> bool:
        actual = str(item.get("instrumentType") or "").upper()
        expected = "CURRENCIES" if instrument.market == Market.FOREX else "COMMODITIES"
        return not actual or actual == expected

    def _choose_market(
        self,
        markets: list[dict[str, Any]],
        instrument: Instrument,
        candidates: tuple[str, ...],
        *,
        exact_only: bool,
    ) -> dict[str, Any] | None:
        expected = {candidate.upper() for candidate in candidates}
        target = self._normalized(instrument.symbol)
        metal_term = self._normalized(self.METAL_SEARCH_TERMS.get(instrument.id, ""))
        ranked: list[tuple[int, str, dict[str, Any]]] = []

        for item in markets:
            if not self._market_matches_type(item, instrument):
                continue
            epic = str(item.get("epic") or "").strip()
            epic_upper = epic.upper()
            if not epic:
                continue
            if exact_only and epic_upper not in expected:
                continue
            text = self._normalized(
                " ".join(
                    str(item.get(field) or "")
                    for field in ("epic", "symbol", "instrumentName")
                )
            )
            score = 100 if epic_upper in expected else 0
            if target and target in text:
                score += 50
            if metal_term and metal_term in text:
                score += 30
            if str(item.get("marketStatus") or "").upper() == "TRADEABLE":
                score += 5
            if score > 0:
                ranked.append((score, epic_upper, item))

        if not ranked:
            return None
        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        return ranked[0][2]

    def _resolve_direct_epic(self, instrument: Instrument) -> str:
        with self._epic_lock:
            cached = self._epic_cache.get(instrument.id)
        if cached:
            return cached

        candidates = self._candidate_epics(instrument)
        payload = self._request_json(
            "/api/v1/markets",
            {"epics": ",".join(candidates)},
        )
        selected = self._choose_market(
            self._market_items(payload), instrument, candidates, exact_only=True
        )

        search_terms: tuple[str, ...] = ()
        if instrument.market == Market.FOREX:
            search_terms = (instrument.symbol, instrument.label)
        elif instrument.id in self.METAL_SEARCH_TERMS:
            search_terms = (self.METAL_SEARCH_TERMS[instrument.id],)
        for search_term in search_terms:
            if selected is not None:
                break
            payload = self._request_json(
                "/api/v1/markets",
                {"searchTerm": search_term},
            )
            selected = self._choose_market(
                self._market_items(payload), instrument, candidates, exact_only=False
            )

        epic = str((selected or {}).get("epic") or "").strip()
        if not epic:
            raise CapitalInstrumentUnavailable(
                f"{instrument.label} is unavailable on Capital.com."
            )
        with self._epic_lock:
            self._epic_cache[instrument.id] = epic
        return epic

    @staticmethod
    def _midpoint(value: object) -> float:
        if not isinstance(value, dict):
            raise CapitalError("Capital.com returned an unusable candle price.")
        try:
            bid_raw = value.get("bid")
            ask_raw = value.get("ask")
            if bid_raw is not None and ask_raw is not None:
                result = (float(bid_raw) + float(ask_raw)) / 2.0
            elif bid_raw is not None:
                result = float(bid_raw)
            elif ask_raw is not None:
                result = float(ask_raw)
            else:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise CapitalError("Capital.com returned an unusable candle price.") from exc
        if not math.isfinite(result) or result <= 0:
            raise CapitalError("Capital.com returned a non-positive candle price.")
        return result

    @staticmethod
    def _market_midpoint(item: dict[str, Any]) -> float:
        try:
            bid_raw = item.get("bid")
            offer_raw = item.get("offer")
            if bid_raw is not None and offer_raw is not None:
                result = (float(bid_raw) + float(offer_raw)) / 2.0
            elif bid_raw is not None:
                result = float(bid_raw)
            elif offer_raw is not None:
                result = float(offer_raw)
            else:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise CapitalError("Capital.com returned an unusable live price.") from exc
        if not math.isfinite(result) or result <= 0:
            raise CapitalError("Capital.com returned a non-positive live price.")
        return result

    def _direct_price(self, instrument: Instrument) -> tuple[float, str]:
        epic = self._resolve_direct_epic(instrument)
        payload = self._request_json("/api/v1/markets", {"epics": epic})
        selected = self._choose_market(
            self._market_items(payload), instrument, (epic,), exact_only=True
        )
        if selected is None:
            with self._epic_lock:
                self._epic_cache.pop(instrument.id, None)
            raise CapitalInstrumentUnavailable(
                f"Capital.com returned no live data for {instrument.label}."
            )
        return self._market_midpoint(selected), epic

    def _history_epic(self, epic: str, label: str, count: int) -> pd.DataFrame:
        payload = self._request_json(
            f"/api/v1/prices/{urllib.parse.quote(epic, safe='')}",
            {
                "resolution": "MINUTE_15",
                "max": min(max(count + 2, 222), 1000),
            },
        )
        prices = payload.get("prices")
        rows: list[dict[str, object]] = []
        for item in prices or []:
            if not isinstance(item, dict):
                continue
            try:
                rows.append(
                    {
                        "time": pd.to_datetime(
                            item.get("snapshotTimeUTC") or item["snapshotTime"],
                            utc=True,
                        ),
                        "open": self._midpoint(item.get("openPrice")),
                        "high": self._midpoint(item.get("highPrice")),
                        "low": self._midpoint(item.get("lowPrice")),
                        "close": self._midpoint(item.get("closePrice")),
                        "volume": float(item.get("lastTradedVolume", 0) or 0),
                    }
                )
            except (KeyError, TypeError, ValueError, OverflowError, CapitalError):
                continue
        if not rows:
            raise CapitalInstrumentUnavailable(
                f"Capital.com returned no M15 candles for {label}."
            )

        data = pd.DataFrame(rows).sort_values("time").drop_duplicates("time")
        data = data.reset_index(drop=True)
        if len(data) > 0:
            last_start = data.iloc[-1]["time"]
            now = pd.Timestamp.now(tz="UTC")
            if last_start + pd.Timedelta(minutes=15) > now:
                data = data.iloc[:-1]
        return data.tail(count).reset_index(drop=True)

    def _direct_bars(self, instrument: Instrument, count: int) -> pd.DataFrame:
        epic = self._resolve_direct_epic(instrument)
        return self._history_epic(epic, instrument.label, count)

    @staticmethod
    def _combine_price(metal_usd: float, fx: float, operation: str) -> float:
        if metal_usd <= 0 or fx <= 0:
            raise CapitalError("Cannot derive metal cross from non-positive prices.")
        if operation == "divide":
            return metal_usd / fx
        if operation == "multiply":
            return metal_usd * fx
        raise CapitalError("Unknown derived metal operation.")

    @staticmethod
    def _combine_bars(left: pd.DataFrame, right: pd.DataFrame, operation: str) -> pd.DataFrame:
        a = left.rename(columns={c: f"{c}_a" for c in ("open", "high", "low", "close", "volume")})
        b = right.rename(columns={c: f"{c}_b" for c in ("open", "high", "low", "close", "volume")})
        merged = a.merge(b, on="time", how="inner")
        if merged.empty:
            raise CapitalError("Unable to align M15 candles for a derived metal cross.")
        out = pd.DataFrame({"time": merged["time"]})
        if operation == "divide":
            out["open"] = merged["open_a"] / merged["open_b"]
            out["close"] = merged["close_a"] / merged["close_b"]
            out["high"] = merged["high_a"] / merged["low_b"]
            out["low"] = merged["low_a"] / merged["high_b"]
        elif operation == "multiply":
            out["open"] = merged["open_a"] * merged["open_b"]
            out["close"] = merged["close_a"] * merged["close_b"]
            out["high"] = merged["high_a"] * merged["high_b"]
            out["low"] = merged["low_a"] * merged["low_b"]
        else:
            raise CapitalError("Unknown derived metal operation.")
        out["volume"] = merged["volume_a"].fillna(0)
        return out.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    @staticmethod
    def _forex_instrument(symbol: str) -> Instrument:
        instrument = next(
            (item for item in INSTRUMENTS[Market.FOREX] if item.symbol == symbol),
            None,
        )
        if instrument is None:
            raise CapitalError(f"Internal Forex mapping is missing: {symbol}")
        return instrument

    def _derived_components(
        self, instrument: Instrument
    ) -> tuple[Instrument, Instrument, str] | None:
        config = self.DERIVED_METALS.get(instrument.id)
        if not config:
            return None
        base_id, fx_symbol, operation = config
        base = get_instrument(Market.METALS, base_id)
        if base is None:
            raise CapitalError(f"Internal metal mapping is missing: {base_id}")
        return base, self._forex_instrument(fx_symbol), operation

    def get_price(self, instrument: Instrument) -> LivePrice:
        provider_symbol = instrument.symbol
        try:
            price, provider_symbol = self._direct_price(instrument)
            source = self.provider_label
        except CapitalInstrumentUnavailable:
            derived = self._derived_components(instrument)
            if derived is None:
                raise
            base, fx_instrument, operation = derived
            metal_price, _ = self._direct_price(base)
            fx_price, _ = self._direct_price(fx_instrument)
            price = self._combine_price(metal_price, fx_price, operation)
            source = f"{self.provider_label} (derived cross)"
        return LivePrice(
            symbol=provider_symbol,
            price=price,
            digits=instrument.digits,
            fetched_at=datetime.now(timezone.utc),
            data_source=source,
        )

    def get_closed_bars(self, instrument: Instrument, count: int) -> pd.DataFrame:
        try:
            data = self._direct_bars(instrument, count)
        except CapitalInstrumentUnavailable:
            derived = self._derived_components(instrument)
            if derived is None:
                raise
            base, fx_instrument, operation = derived
            metal = self._direct_bars(base, count + 10)
            fx = self._direct_bars(fx_instrument, count + 10)
            data = self._combine_bars(metal, fx, operation).tail(count).reset_index(drop=True)
        if len(data) < 220:
            raise CapitalError(
                f"Only {len(data)} closed M15 candles are available for {instrument.label}; at least 220 are required."
            )
        return data

    def instrument_availability(
        self, instruments: list[Instrument]
    ) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for instrument in instruments:
            epic: str | None = None
            try:
                epic = self._resolve_direct_epic(instrument)
                mode = "direct"
            except CapitalInstrumentUnavailable:
                derived = self._derived_components(instrument)
                if derived is None:
                    mode = "unavailable"
                else:
                    base, fx_instrument, _ = derived
                    try:
                        self._resolve_direct_epic(base)
                        self._resolve_direct_epic(fx_instrument)
                        mode = "derived"
                    except CapitalInstrumentUnavailable:
                        mode = "unavailable"
            result.append(
                {
                    "market": instrument.market.value,
                    "id": instrument.id,
                    "label": instrument.label,
                    "symbol": instrument.symbol,
                    "provider_symbol": epic,
                    "available": mode != "unavailable",
                    "mode": mode,
                }
            )
        return result


# -----------------------------------------------------------------------------
# Strategy
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class SideAssessment:
    direction: Direction
    score: int
    reasons: list[str]
    blockers: list[str]
    setup: str | None


class M15SignalStrategy:
    """M15 directional strategy with an adaptive ~70/30 signal/WAIT target.

    The target controls how often the bot emits a direction, not how often that
    direction will be profitable. The threshold is calibrated from recent raw
    confirmation scores for the same instrument and therefore adapts to trend,
    volatility and asset class instead of using one fixed score everywhere.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def analyze(
        self,
        instrument: Instrument,
        live_price: LivePrice,
        raw_bars: pd.DataFrame,
    ) -> Signal:
        if len(raw_bars) < 220:
            raise ValueError("At least 220 closed M15 bars are required.")

        df = add_indicators(raw_bars)
        row = df.iloc[-1]
        current = live_price.price
        required = [
            "ema20",
            "ema50",
            "ema200",
            "rsi",
            "macd",
            "macd_signal",
            "macd_hist",
            "atr",
        ]
        if row[required].isna().any():
            raise ValueError("Not enough clean data to calculate indicators.")

        atr = float(row["atr"])
        support, resistance = swing_levels(df, current, lookback=80, wing=2)
        global_blockers = self._global_blockers(atr, current)

        buy = self._assess_side(Direction.BUY, df, current, atr, support, resistance)
        sell = self._assess_side(Direction.SELL, df, current, atr, support, resistance)
        assessment = buy if buy.score >= sell.score else sell
        score = assessment.score
        decision_strength = self._decision_strength(assessment, row, atr)

        dynamic_threshold = self._adaptive_threshold(df)

        if global_blockers:
            direction = Direction.WAIT
            blockers = global_blockers
        elif decision_strength >= dynamic_threshold and buy.score != sell.score:
            direction = assessment.direction
            blockers = assessment.blockers
        else:
            direction = Direction.WAIT
            blockers = assessment.blockers.copy()
            blockers.insert(
                0,
                f"Market confirmation is below the adaptive 70/30 decision threshold.",
            )

        signal = Signal(
            instrument=instrument,
            provider_symbol=live_price.symbol,
            direction=direction,
            current_price=current,
            digits=instrument.digits,
            timeframe=self.settings.timeframe_label,
            score=score,
            price_fetched_at=live_price.fetched_at,
            data_source=live_price.data_source,
            atr=atr,
            rsi=float(row["rsi"]),
            ema20=float(row["ema20"]),
            ema50=float(row["ema50"]),
            ema200=float(row["ema200"]),
            macd=float(row["macd"]),
            macd_signal=float(row["macd_signal"]),
            support=support,
            resistance=resistance,
            setup=assessment.setup,
            reasons=assessment.reasons,
            blockers=blockers,
        )

        if direction is not Direction.WAIT:
            self._attach_trade_levels(signal, df, atr, support, resistance)
        return signal

    @staticmethod
    def _global_blockers(atr: float, current: float) -> list[str]:
        blockers: list[str] = []
        if atr <= 0 or current <= 0 or not np.isfinite(atr) or not np.isfinite(current):
            blockers.append("Invalid ATR or current price.")
            return blockers
        # Only block truly abnormal data. Normal quiet/volatile conditions are
        # reflected in the score rather than automatically forcing WAIT.
        if atr / current > 0.15:
            blockers.append("Market data shows abnormally high M15 volatility.")
        return blockers

    @staticmethod
    def _decision_strength(
        assessment: SideAssessment,
        row: pd.Series,
        atr: float,
    ) -> float:
        """Continuous ranking value used only for adaptive 70/30 calibration."""
        if atr <= 0:
            return float(assessment.score)
        ema_push = (
            abs(float(row["ema20"]) - float(row["ema50"]))
            + 0.5 * abs(float(row["ema50"]) - float(row["ema200"]))
        ) / atr
        macd_push = abs(float(row["macd_hist"])) / atr
        rsi_push = abs(float(row["rsi"]) - 50.0) / 50.0
        return (
            float(assessment.score)
            + min(ema_push, 6.0) * 0.35
            + min(macd_push, 3.0) * 0.25
            + min(rsi_push, 1.0) * 0.20
        )

    def _adaptive_threshold(self, df: pd.DataFrame) -> float:
        """Calibrate a threshold targeting about 70% directional outcomes.

        For TARGET_SIGNAL_RATE=0.70, the bot uses the 30th percentile of the
        recent best-side decision strengths. This targets frequency, not win
        rate. Discrete market states and rare invalid-data blockers can make the
        realized ratio differ from exactly 70/30.
        """
        target = min(max(self.settings.signal_target_rate, 0.10), 0.95)
        lookback = max(30, self.settings.adaptive_lookback)
        start = max(210, len(df) - lookback)
        strengths: list[float] = []

        for i in range(start, len(df) - 1):
            sub = df.iloc[: i + 1]
            row = sub.iloc[-1]
            atr = float(row["atr"])
            current = float(row["close"])
            if not np.isfinite(atr) or atr <= 0 or current <= 0:
                continue
            support, resistance = swing_levels(sub, current, lookback=80, wing=2)
            buy = self._assess_side(Direction.BUY, sub, current, atr, support, resistance)
            sell = self._assess_side(Direction.SELL, sub, current, atr, support, resistance)
            assessment = buy if buy.score >= sell.score else sell
            strengths.append(self._decision_strength(assessment, row, atr))

        if len(strengths) < 20:
            return float(self.settings.min_score)

        wait_quantile = 1.0 - target
        threshold = float(np.quantile(np.asarray(strengths, dtype=float), wait_quantile))
        return max(float(self.settings.min_score), threshold)

    def _assess_side(
        self,
        direction: Direction,
        df: pd.DataFrame,
        current: float,
        atr: float,
        support: float | None,
        resistance: float | None,
    ) -> SideAssessment:
        row = df.iloc[-1]
        prev = df.iloc[-2]
        score = 0
        reasons: list[str] = []
        blockers: list[str] = []
        bullish = direction is Direction.BUY

        fast_structure = (
            row["ema20"] > row["ema50"] if bullish else row["ema20"] < row["ema50"]
        )
        long_structure = (
            row["ema50"] > row["ema200"] if bullish else row["ema50"] < row["ema200"]
        )
        price_side = current > row["ema20"] if bullish else current < row["ema20"]
        ema50_slope = row["ema50"] > prev["ema50"] if bullish else row["ema50"] < prev["ema50"]
        macd_ok = row["macd"] > row["macd_signal"] if bullish else row["macd"] < row["macd_signal"]
        hist_ok = row["macd_hist"] > 0 if bullish else row["macd_hist"] < 0
        hist_strengthening = row["macd_hist"] > prev["macd_hist"] if bullish else row["macd_hist"] < prev["macd_hist"]

        if fast_structure:
            score += 15
            reasons.append("EMA20/EMA50 direction agrees")
        else:
            blockers.append("EMA20/EMA50 do not confirm this side.")

        if long_structure:
            score += 10
            reasons.append("EMA50/EMA200 trend structure agrees")
        else:
            blockers.append("Longer EMA structure is mixed.")

        if price_side:
            score += 10
            reasons.append("Price is on the correct side of EMA20")

        if ema50_slope:
            score += 5
            reasons.append("EMA50 slope confirms direction")

        rsi = float(row["rsi"])
        if bullish:
            if 50 <= rsi <= 75:
                score += 15
                reasons.append(f"RSI supports bullish momentum ({rsi:.1f})")
            elif 45 <= rsi < 50:
                score += 7
                reasons.append(f"RSI is near bullish confirmation ({rsi:.1f})")
            else:
                blockers.append(f"RSI is weak for BUY ({rsi:.1f}).")
        else:
            if 25 <= rsi <= 50:
                score += 15
                reasons.append(f"RSI supports bearish momentum ({rsi:.1f})")
            elif 50 < rsi <= 55:
                score += 7
                reasons.append(f"RSI is near bearish confirmation ({rsi:.1f})")
            else:
                blockers.append(f"RSI is weak for SELL ({rsi:.1f}).")

        if macd_ok:
            score += 15
            reasons.append("MACD line confirms direction")
        else:
            blockers.append("MACD line does not confirm this side.")

        if hist_ok:
            score += 10
            reasons.append("MACD histogram confirms direction")
        if hist_strengthening:
            score += 5
            reasons.append("MACD momentum is strengthening")

        setup, setup_score = self._detect_setup(direction, df, atr, support, resistance)
        score += setup_score
        if setup:
            reasons.append(setup)

        if self._has_room(direction, current, atr, support, resistance):
            score += 10
            reasons.append("There is room before the nearest opposing level")
        else:
            blockers.append("A nearby support/resistance level reduces setup quality.")

        # Valid ATR is a small positive confirmation rather than a strict gate.
        score += 5
        reasons.append("ATR data is valid")
        return SideAssessment(direction, min(score, 100), reasons, blockers, setup)

    @staticmethod
    def _detect_setup(
        direction: Direction,
        df: pd.DataFrame,
        atr: float,
        support: float | None,
        resistance: float | None,
    ) -> tuple[str | None, int]:
        row = df.iloc[-1]
        prev = df.iloc[-2]
        bullish = direction is Direction.BUY

        touched_ema = (
            abs(float(row["low"] if bullish else row["high"]) - float(row["ema20"]))
            <= 0.35 * atr
        )
        candle_confirms = row["close"] > row["open"] if bullish else row["close"] < row["open"]
        closed_correct_side = row["close"] > row["ema20"] if bullish else row["close"] < row["ema20"]
        if touched_ema and candle_confirms and closed_correct_side:
            return "EMA20 pullback/rejection setup", 10

        if len(df) >= 42:
            if bullish:
                prior_res = float(df.iloc[-41:-1]["high"].max())
                if row["close"] > prior_res + 0.10 * atr and prev["close"] <= prior_res + 0.10 * atr:
                    return "Confirmed resistance breakout", 10
            else:
                prior_sup = float(df.iloc[-41:-1]["low"].min())
                if row["close"] < prior_sup - 0.10 * atr and prev["close"] >= prior_sup - 0.10 * atr:
                    return "Confirmed support breakout", 10
        return None, 0

    @staticmethod
    def _has_room(
        direction: Direction,
        current: float,
        atr: float,
        support: float | None,
        resistance: float | None,
    ) -> bool:
        if direction is Direction.BUY:
            return resistance is None or (resistance - current) >= 0.70 * atr
        return support is None or (current - support) >= 0.70 * atr

    def _attach_trade_levels(
        self,
        signal: Signal,
        df: pd.DataFrame,
        atr: float,
        support: float | None,
        resistance: float | None,
    ) -> None:
        """Build ATR/structure-aware levels without creating extra WAIT states."""
        bullish = signal.direction is Direction.BUY
        entry = signal.current_price
        recent_low = float(df.tail(12)["low"].min())
        recent_high = float(df.tail(12)["high"].max())

        if bullish:
            structural = support if support is not None else recent_low
            structural_distance = entry - (structural - 0.25 * atr)
        else:
            structural = resistance if resistance is not None else recent_high
            structural_distance = (structural + 0.25 * atr) - entry

        if not np.isfinite(structural_distance) or structural_distance <= 0:
            structural_distance = 1.20 * atr

        stop_distance = max(self.settings.min_stop_atr * atr, structural_distance)
        stop_distance = min(stop_distance, self.settings.max_stop_atr * atr)

        stop_loss = entry - stop_distance if bullish else entry + stop_distance
        tp1 = entry + self.settings.tp1_rr * stop_distance if bullish else entry - self.settings.tp1_rr * stop_distance
        tp2 = entry + self.settings.tp2_rr * stop_distance if bullish else entry - self.settings.tp2_rr * stop_distance

        signal.entry = entry
        signal.stop_loss = stop_loss
        signal.take_profit_1 = tp1
        signal.take_profit_2 = tp2
        signal.risk_reward = self.settings.tp2_rr
        signal.reasons.append(f"ATR/structure stop with {self.settings.tp2_rr:.2f}R TP2 target")


# -----------------------------------------------------------------------------
# Telegram formatting and keyboards
# -----------------------------------------------------------------------------

def _p(value: float | None, digits: int) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _utc_text(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def format_signal(signal: Signal) -> str:
    icon = {
        Direction.BUY: "🟢",
        Direction.SELL: "🔴",
        Direction.WAIT: "⚪",
    }[signal.direction]
    title = f"<b>{escape(signal.instrument.label)} · {escape(signal.timeframe)}</b>"
    price = _p(signal.current_price, signal.digits)

    if signal.direction is Direction.WAIT:
        blocker_text = "\n".join(
            f"• {escape(x)}" for x in signal.blockers[:4]
        ) or "• No valid setup right now."
        return (
            f"{title}\n\n{icon} <b>WAIT</b>\n\n"
            f"💵 <b>Current price:</b> <code>{price}</code>\n"
            f"📊 <b>Confirmation score:</b> {signal.score}/100\n"
            f"⏱ <b>Timeframe:</b> {escape(signal.timeframe)}\n"
            f"🕐 <b>Price fetched:</b> <code>{_utc_text(signal.price_fetched_at)}</code>\n\n"
            f"<b>Why no trade:</b>\n{blocker_text}\n\n"
            "<i>No valid entry is available right now. Recalculate after a new closed M15 candle.</i>\n\n"
            f"Data source: {escape(signal.data_source or 'Market data provider')}\n"
            "⚠️ Trading signals are informational and do not guarantee profit."
        )

    reasons = "\n".join(f"• {escape(x)}" for x in signal.reasons[:5])
    return (
        f"{title}\n\n{icon} <b>{signal.direction.value}</b>\n\n"
        f"💵 <b>Current price:</b> <code>{price}</code>\n\n"
        f"🎯 <b>ENTRY:</b> <code>{_p(signal.entry, signal.digits)}</code>\n"
        f"🛑 <b>STOP LOSS:</b> <code>{_p(signal.stop_loss, signal.digits)}</code>\n\n"
        f"💰 <b>TAKE PROFIT 1:</b> <code>{_p(signal.take_profit_1, signal.digits)}</code>\n"
        f"💰 <b>TAKE PROFIT 2:</b> <code>{_p(signal.take_profit_2, signal.digits)}</code>\n\n"
        f"⚖️ <b>Risk / Reward:</b> <code>1:{signal.risk_reward:.2f}</code>\n"
        f"📊 <b>Confirmation score:</b> {signal.score}/100\n"
        f"⏱ <b>Timeframe:</b> {escape(signal.timeframe)}\n"
        f"📈 <b>Setup:</b> {escape(signal.setup or 'Trend confirmation')}\n"
        f"🕐 <b>Price fetched:</b> <code>{_utc_text(signal.price_fetched_at)}</code>\n\n"
        f"<b>Confirmations:</b>\n{reasons}\n\n"
        f"Data source: {escape(signal.data_source or 'Market data provider')}\n"
        "⚠️ Trading signals are informational and do not guarantee profit."
    )


def format_price_update(signal: Signal, live: LivePrice) -> str:
    icon = {
        Direction.BUY: "🟢",
        Direction.SELL: "🔴",
        Direction.WAIT: "⚪",
    }[signal.direction]
    return (
        f"<b>{escape(signal.instrument.label)} · {escape(signal.timeframe)}</b>\n\n"
        f"{icon} <b>{signal.direction.value}</b>\n\n"
        f"💵 <b>Current price:</b> <code>{_p(live.price, live.digits)}</code>\n"
        f"Updated: <code>{_utc_text(live.fetched_at)}</code>\n\n"
        f"🎯 <b>ENTRY:</b> <code>{_p(signal.entry, signal.digits)}</code>\n"
        f"🛑 <b>STOP LOSS:</b> <code>{_p(signal.stop_loss, signal.digits)}</code>\n"
        f"💰 <b>TP1:</b> <code>{_p(signal.take_profit_1, signal.digits)}</code>\n"
        f"💰 <b>TP2:</b> <code>{_p(signal.take_profit_2, signal.digits)}</code>\n\n"
        "<i>Only the current price was refreshed. Signal levels were not recalculated.</i>\n\n"
        f"Data source: {escape(live.data_source or signal.data_source or 'Market data provider')}"
    )


def market_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=MARKET_TITLES[Market.FOREX], callback_data="market:forex")],
            [InlineKeyboardButton(text=MARKET_TITLES[Market.METALS], callback_data="market:metals")],
            [InlineKeyboardButton(text=MARKET_TITLES[Market.CRYPTO], callback_data="market:crypto")],
            [InlineKeyboardButton(text=MARKET_TITLES[Market.NASDAQ], callback_data="market:nasdaq")],
        ]
    )


def instruments_keyboard(market: Market) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []
    items = INSTRUMENTS[market]
    for i in range(0, len(items), 2):
        rows.append(
            [
                InlineKeyboardButton(
                    text=item.label,
                    callback_data=f"asset:{market.value}:{item.id}",
                )
                for item in items[i : i + 2]
            ]
        )
    rows.append([InlineKeyboardButton(text="⬅️ Back", callback_data="home")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def signal_keyboard(market: Market, instrument_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🔄 Refresh price",
                    callback_data=f"price:{market.value}:{instrument_id}",
                ),
                InlineKeyboardButton(
                    text="📊 Recalculate signal",
                    callback_data=f"signal:{market.value}:{instrument_id}",
                ),
            ],
            [
                InlineKeyboardButton(
                    text="⬅️ Instruments", callback_data=f"market:{market.value}"
                ),
                InlineKeyboardButton(text="🏠 Main menu", callback_data="home"),
            ],
        ]
    )


# -----------------------------------------------------------------------------
# Services and shared cache
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class SignalCache:
    """Per-user last rendered signal, used only by the Refresh price button."""
    items: dict[tuple[int, str, str], Signal] = field(default_factory=dict)

    def put(self, user_id: int, instrument: Instrument, signal: Signal) -> None:
        self.items[(user_id, instrument.market.value, instrument.id)] = signal

    def get(self, user_id: int, instrument: Instrument) -> Signal | None:
        return self.items.get((user_id, instrument.market.value, instrument.id))


@dataclass(slots=True)
class _CacheEntry:
    value: Any
    expires_at: float


class SharedMarketCache:
    """Process-wide cache shared by Telegram users and FastAPI requests."""
    def __init__(self) -> None:
        self._items: dict[str, _CacheEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self.hits = 0
        self.misses = 0

    def _fresh(self, key: str) -> Any | None:
        entry = self._items.get(key)
        if entry is None:
            return None
        if entry.expires_at <= time.monotonic():
            self._items.pop(key, None)
            return None
        return entry.value

    async def get_or_load(
        self,
        key: str,
        ttl_seconds: float,
        loader: Callable[[], Awaitable[Any]],
    ) -> tuple[Any, bool]:
        value = self._fresh(key)
        if value is not None:
            self.hits += 1
            return value, True
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            value = self._fresh(key)
            if value is not None:
                self.hits += 1
                return value, True
            self.misses += 1
            value = await loader()
            self._items[key] = _CacheEntry(value=value, expires_at=time.monotonic() + max(1.0, ttl_seconds))
            return value, False

    def clear(self) -> None:
        self._items.clear()

    def stats(self) -> dict[str, int]:
        now = time.monotonic()
        active = sum(1 for item in self._items.values() if item.expires_at > now)
        return {"active_entries": active, "hits": self.hits, "misses": self.misses}


def seconds_until_next_m15_close(grace_seconds: int = 4) -> float:
    now = datetime.now(timezone.utc)
    base = now.replace(second=0, microsecond=0)
    next_minute = ((now.minute // 15) + 1) * 15
    if next_minute >= 60:
        next_close = base.replace(minute=0) + timedelta(hours=1)
    else:
        next_close = base.replace(minute=next_minute)
    return max(1.0, (next_close - now).total_seconds() + grace_seconds)


class MarketDataRouter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.capital = CapitalClient(settings)
        self.twelve = TwelveDataClient(settings)

    def provider_name(self, instrument: Instrument) -> str:
        if instrument.market in {Market.FOREX, Market.METALS}:
            return self.capital.provider_label
        return "Twelve Data"

    def _client(self, instrument: Instrument):
        if instrument.market in {Market.FOREX, Market.METALS}:
            return self.capital
        return self.twelve

    async def price(self, instrument: Instrument) -> LivePrice:
        client = self._client(instrument)
        return await asyncio.to_thread(client.get_price, instrument)

    async def bars(self, instrument: Instrument, count: int) -> pd.DataFrame:
        client = self._client(instrument)
        return await asyncio.to_thread(client.get_closed_bars, instrument, count)


class SignalService:
    def __init__(self, settings: Settings, cache: SharedMarketCache | None = None):
        self.settings = settings
        self.strategy = M15SignalStrategy(settings)
        self.market_data = MarketDataRouter(settings)
        self.cache = cache or SharedMarketCache()

    async def _bars_with_meta(self, instrument: Instrument) -> tuple[pd.DataFrame, bool]:
        key = f"bars:{self.market_data.provider_name(instrument)}:{instrument.symbol}:M15"
        ttl = seconds_until_next_m15_close(self.settings.candle_cache_grace_seconds)
        return await self.cache.get_or_load(
            key, ttl, lambda: self.market_data.bars(instrument, self.settings.bars_count)
        )

    async def current_price_with_meta(self, instrument: Instrument) -> tuple[LivePrice, bool]:
        key = f"price:{self.market_data.provider_name(instrument)}:{instrument.symbol}"
        ttl = (
            float(self.settings.capital_price_cache_seconds)
            if instrument.market in {Market.FOREX, Market.METALS}
            else float(self.settings.price_cache_seconds)
        )
        return await self.cache.get_or_load(
            key, ttl, lambda: self.market_data.price(instrument)
        )

    async def calculate_with_meta(self, instrument: Instrument) -> tuple[Signal, bool]:
        key = f"signal:{instrument.market.value}:{instrument.id}:M15"
        ttl = seconds_until_next_m15_close(self.settings.candle_cache_grace_seconds)

        async def loader() -> Signal:
            bars, _ = await self._bars_with_meta(instrument)
            live, _ = await self.current_price_with_meta(instrument)
            return self.strategy.analyze(instrument, live, bars)

        return await self.cache.get_or_load(key, ttl, loader)

    async def calculate(self, instrument: Instrument) -> Signal:
        signal, _ = await self.calculate_with_meta(instrument)
        return signal

    async def current_price(self, instrument: Instrument) -> LivePrice:
        live, _ = await self.current_price_with_meta(instrument)
        return live


ADMIN_FILTER_NAMES = {
    "all": "Все",
    "active": "Активные",
    "monthly": "Monthly",
    "lifetime": "Lifetime",
    "expired": "Истёкшие",
    "cancelled": "Отменённые",
    "inactive": "Без подписки",
}


class AdminStates(StatesGroup):
    waiting_find_user = State()
    waiting_broadcast = State()
    broadcast_confirm = State()
    waiting_referral_percent = State()


def admin_panel_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="👥 Пользователи", callback_data="adm:users"),
                InlineKeyboardButton(text="📢 Рассылка", callback_data="adm:broadcast"),
            ],
            [InlineKeyboardButton(text="🔍 Найти пользователя", callback_data="adm:find")],
            [InlineKeyboardButton(text="💳 Ожидают оплаты", callback_data="adm:payments")],
            [
                InlineKeyboardButton(text="🤝 Рефералы", callback_data="adm:referrals"),
                InlineKeyboardButton(text="💸 Выплаты", callback_data="adm:payouts"),
            ],
            [
                InlineKeyboardButton(text="📊 Бизнес", callback_data="adm:stats"),
                InlineKeyboardButton(text="🏆 Лидерборд", callback_data="adm:leaderboard"),
            ],
            [InlineKeyboardButton(text="📈 Открыть сигналы", callback_data="access:open")],
        ]
    )


def admin_back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="⬅️ Админ-панель", callback_data="adm:panel")]
        ]
    )


def admin_users_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="👥 Все", callback_data="adm:ul:all:0"),
                InlineKeyboardButton(text="✅ Активные", callback_data="adm:ul:active:0"),
            ],
            [
                InlineKeyboardButton(text="📅 Monthly", callback_data="adm:ul:monthly:0"),
                InlineKeyboardButton(text="♾ Lifetime", callback_data="adm:ul:lifetime:0"),
            ],
            [
                InlineKeyboardButton(text="⌛ Истёкшие", callback_data="adm:ul:expired:0"),
                InlineKeyboardButton(text="🚫 Отменённые", callback_data="adm:ul:cancelled:0"),
            ],
            [InlineKeyboardButton(text="⚪ Без подписки", callback_data="adm:ul:inactive:0")],
            [InlineKeyboardButton(text="⬅️ Админ-панель", callback_data="adm:panel")],
        ]
    )


def admin_broadcast_segments_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="👥 Всем пользователям", callback_data="adm:bc:all")],
            [InlineKeyboardButton(text="✅ Активным подписчикам", callback_data="adm:bc:active")],
            [
                InlineKeyboardButton(text="📅 Monthly", callback_data="adm:bc:monthly"),
                InlineKeyboardButton(text="♾ Lifetime", callback_data="adm:bc:lifetime"),
            ],
            [InlineKeyboardButton(text="⬅️ Админ-панель", callback_data="adm:panel")],
        ]
    )


def _admin_status_label(row: sqlite3.Row) -> str:
    status = row["status"]
    plan = row["plan"]
    if status == "active" and plan == "lifetime":
        return "♾ Lifetime"
    if status == "active" and plan == "monthly":
        return "✅ Monthly"
    return {
        "inactive": "⚪ Без подписки",
        "expired": "⌛ Истекла",
        "cancelled": "🚫 Отменена",
    }.get(status, status or "—")


def _admin_user_name(row: sqlite3.Row) -> str:
    if row["username"]:
        return f"@{row['username']}"
    if row["first_name"]:
        return str(row["first_name"])
    return str(row["telegram_id"])


def admin_user_text(row: sqlite3.Row, default_referral_percent: float = 30.0) -> str:
    username = f"@{escape(row['username'])}" if row["username"] else "—"
    first_name = escape(row["first_name"] or "—")
    expires = row["expires_at"]
    if row["plan"] == "lifetime":
        expires_text = "Бессрочно"
    elif expires:
        dt = AccessStore._dt(expires)
        expires_text = dt.strftime("%d.%m.%Y %H:%M UTC") if dt else str(expires)
    else:
        expires_text = "—"
    override = row["referral_percent_override"] if "referral_percent_override" in row.keys() else None
    effective_percent = float(override) if override is not None else float(default_referral_percent)
    referral_mode = "персональный" if override is not None else "по умолчанию"
    return (
        "👤 <b>Пользователь</b>\n\n"
        f"Имя: <b>{first_name}</b>\n"
        f"Username: {username}\n"
        f"Telegram ID: <code>{row['telegram_id']}</code>\n\n"
        f"Статус: <b>{escape(_admin_status_label(row))}</b>\n"
        f"Тариф: <b>{escape(row['plan'] or '—')}</b>\n"
        f"Доступ до: <b>{escape(expires_text)}</b>\n\n"
        f"Реферальная ставка: <b>{effective_percent:g}%</b> ({referral_mode})"
    )


def admin_user_keyboard(row: sqlite3.Row) -> InlineKeyboardMarkup:
    uid = int(row["telegram_id"])
    rows = [
        [
            InlineKeyboardButton(
                text="➕ Выдать / продлить 30 дней",
                callback_data=f"adm:u:{uid}:monthly",
            )
        ],
        [InlineKeyboardButton(text="♾ Выдать Lifetime", callback_data=f"adm:u:{uid}:lifetime")],
        [InlineKeyboardButton(text="💰 Реферальный %", callback_data=f"adm:u:{uid}:refpct")],
    ]
    if row["status"] == "active":
        rows.append(
            [
                InlineKeyboardButton(
                    text="🚫 Отменить подписку",
                    callback_data=f"adm:u:{uid}:cancelask",
                )
            ]
        )
    rows.extend(
        [
            [InlineKeyboardButton(text="🔄 Обновить", callback_data=f"adm:u:{uid}:view")],
            [InlineKeyboardButton(text="⬅️ Пользователи", callback_data="adm:users")],
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def admin_cancel_confirm_keyboard(user_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="✅ Да, отменить",
                    callback_data=f"adm:u:{user_id}:cancel",
                )
            ],
            [InlineKeyboardButton(text="⬅️ Назад", callback_data=f"adm:u:{user_id}:view")],
        ]
    )

def admin_referral_percent_keyboard(user_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="10%", callback_data=f"adm:rp:{user_id}:10"),
                InlineKeyboardButton(text="20%", callback_data=f"adm:rp:{user_id}:20"),
                InlineKeyboardButton(text="30%", callback_data=f"adm:rp:{user_id}:30"),
            ],
            [
                InlineKeyboardButton(text="40%", callback_data=f"adm:rp:{user_id}:40"),
                InlineKeyboardButton(text="50%", callback_data=f"adm:rp:{user_id}:50"),
            ],
            [InlineKeyboardButton(text="✏️ Свой %", callback_data=f"adm:rp:{user_id}:custom")],
            [InlineKeyboardButton(text="♻️ По умолчанию", callback_data=f"adm:rp:{user_id}:reset")],
            [InlineKeyboardButton(text="⬅️ Пользователь", callback_data=f"adm:u:{user_id}:view")],
        ]
    )


def admin_payout_review_keyboard(payout_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Выплачено", callback_data=f"admin:payout:paid:{payout_id}"),
                InlineKeyboardButton(text="❌ Отклонить", callback_data=f"admin:payout:rejected:{payout_id}"),
            ]
        ]
    )

# -----------------------------------------------------------------------------
# Telegram handlers
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Access database and payment review
# -----------------------------------------------------------------------------

START_TEXT = """🤖 <b>Meet MENTAL-TRADER BOT — Your M15 Market Signal Engine</b>

Welcome! MENTAL-TRADER analyzes live market data and gives structured M15 trading signals you can use when trading manually in MT5 or another platform.

<b>What you get:</b>
⚡️ <b>Live market data:</b> Forex & Metals via Capital.com; Crypto & Nasdaq via Twelve Data.
📈 <b>Technical analysis:</b> EMA, RSI, MACD, ATR and support/resistance.
🎯 <b>Structured signals:</b> BUY / SELL / WAIT with Entry, Stop Loss and two Take Profit levels.
🛡️ <b>Risk-aware logic:</b> Every directional signal includes a defined stop and Risk/Reward target.

⚠️ Trading involves risk. Signals are analytical information, not guaranteed profits."""

PLAN_TEXT = """💎 <b>Unlock Full Access — Choose Your Plan</b>

📅 <b>Monthly Plan — ${monthly:g}</b>
· Full access for 30 days
· Forex, Metals, Crypto and Nasdaq signals
· Current price + M15 analysis
· Signal updates during your access period

♾️ <b>Lifetime Plan — ${lifetime:g}</b>
· One-time payment
· Lifetime access to the signal bot
· Forex, Metals, Crypto and Nasdaq signals
· No recurring access fee

⚠️ Trading involves risk. Past performance does not guarantee future results."""


@dataclass(frozen=True, slots=True)
class AccessInfo:
    active: bool
    plan: str | None = None
    expires_at: datetime | None = None


class AccessStore:
    """SQLite store for users, access, payments, referrals and business statistics."""

    def __init__(self, path: str):
        db_path = Path(path).expanduser()
        if db_path.parent != Path("."):
            db_path.parent.mkdir(parents=True, exist_ok=True)

        # Upgrade older deployments that stored bot.db in the application
        # directory. The migration only runs when the persistent DB does not
        # exist yet, so an existing /app/data/bot.db is never overwritten.
        self._migrate_legacy_database(db_path)

        self.path = str(db_path)
        self._lock = threading.RLock()
        self._init_db()

    @staticmethod
    def _migrate_legacy_database(target: Path) -> None:
        if target.exists():
            return

        candidates = (Path("bot.db"), Path("/app/bot.db"))
        log = logging.getLogger(__name__)

        for legacy in candidates:
            try:
                if not legacy.exists():
                    continue
                if legacy.resolve() == target.resolve():
                    continue

                target.parent.mkdir(parents=True, exist_ok=True)

                # sqlite3 backup() copies a consistent snapshot, including data
                # that may previously have lived in a WAL journal.
                with sqlite3.connect(str(legacy), timeout=15) as source:
                    with sqlite3.connect(str(target), timeout=15) as destination:
                        source.backup(destination)

                log.info(
                    "Migrated legacy SQLite database from %s to persistent path %s",
                    legacy,
                    target,
                )
                return
            except Exception:
                try:
                    target.unlink(missing_ok=True)
                except OSError:
                    pass
                log.exception(
                    "Could not migrate legacy SQLite database from %s to %s",
                    legacy,
                    target,
                )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            # WAL + NORMAL synchronous mode improves durability across process
            # restarts while keeping SQLite responsive for webhook/API traffic.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    telegram_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    plan TEXT,
                    status TEXT NOT NULL DEFAULT 'inactive',
                    activated_at TEXT,
                    expires_at TEXT,
                    updated_at TEXT NOT NULL,
                    created_at TEXT,
                    last_seen_at TEXT,
                    referral_code TEXT,
                    referrer_id INTEGER,
                    referral_percent_override REAL
                );

                CREATE TABLE IF NOT EXISTS payment_intents (
                    telegram_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    plan_key TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    amount_text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS payments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    telegram_id INTEGER NOT NULL,
                    username TEXT,
                    first_name TEXT,
                    plan_key TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    amount_text TEXT NOT NULL,
                    receipt_chat_id INTEGER,
                    receipt_message_id INTEGER,
                    receipt_file_id TEXT,
                    receipt_kind TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    reviewed_at TEXT,
                    reviewed_by INTEGER
                );

                CREATE TABLE IF NOT EXISTS referral_rewards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    referrer_id INTEGER NOT NULL,
                    referred_user_id INTEGER NOT NULL UNIQUE,
                    payment_id INTEGER NOT NULL UNIQUE,
                    payment_amount_usd REAL NOT NULL,
                    commission_percent REAL NOT NULL,
                    commission_amount_usd REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS referral_payouts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    telegram_id INTEGER NOT NULL,
                    amount_usdt REAL NOT NULL,
                    wallet_address TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    reviewed_at TEXT,
                    reviewed_by INTEGER
                );

                CREATE TABLE IF NOT EXISTS funnel_events (
                    telegram_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (telegram_id, event_type)
                );
                CREATE INDEX IF NOT EXISTS idx_funnel_events_type ON funnel_events(event_type, created_at);
                CREATE INDEX IF NOT EXISTS idx_referral_rewards_referrer ON referral_rewards(referrer_id);
                CREATE INDEX IF NOT EXISTS idx_referral_payouts_user ON referral_payouts(telegram_id, status);
                """
            )
            # Lightweight migration for databases created by earlier bot versions.
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
            if "created_at" not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN created_at TEXT")
            if "last_seen_at" not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN last_seen_at TEXT")
            if "referral_code" not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN referral_code TEXT")
            if "referrer_id" not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN referrer_id INTEGER")
            if "referral_percent_override" not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN referral_percent_override REAL")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_referral_code "
                "ON users(referral_code) WHERE referral_code IS NOT NULL"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_users_referrer_id ON users(referrer_id)"
            )

            payment_columns = {row["name"] for row in conn.execute("PRAGMA table_info(payments)").fetchall()}
            payment_migrations = {
                "provider": "TEXT NOT NULL DEFAULT 'manual'",
                "provider_payment_id": "TEXT",
                "provider_order_id": "TEXT",
                "price_amount_usd": "REAL",
                "pay_amount": "REAL",
                "pay_currency": "TEXT",
                "pay_address": "TEXT",
                "provider_status": "TEXT",
                "purchase_id": "TEXT",
                "updated_at": "TEXT",
            }
            for column_name, ddl in payment_migrations.items():
                if column_name not in payment_columns:
                    conn.execute(f"ALTER TABLE payments ADD COLUMN {column_name} {ddl}")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_payments_provider_payment "
                "ON payments(provider, provider_payment_id) WHERE provider_payment_id IS NOT NULL"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_payments_provider_order "
                "ON payments(provider, provider_order_id) WHERE provider_order_id IS NOT NULL"
            )

            now = self._now().isoformat()
            conn.execute(
                "UPDATE users SET created_at=COALESCE(created_at, updated_at, ?)",
                (now,),
            )
            conn.execute(
                "UPDATE users SET last_seen_at=COALESCE(last_seen_at, updated_at, created_at, ?)",
                (now,),
            )
            # Do not backfill Mini App opens from the users table: /start users are
            # also stored there and would inflate conversion analytics. Verified
            # Mini App opens are recorded only by the authenticated API path.
            conn.execute(
                "INSERT OR IGNORE INTO funnel_events (telegram_id, event_type, created_at) "
                "SELECT telegram_id, 'payment_started', MIN(created_at) FROM payments GROUP BY telegram_id"
            )
            conn.execute(
                "INSERT OR IGNORE INTO funnel_events (telegram_id, event_type, created_at) "
                "SELECT telegram_id, 'payment_paid', MIN(COALESCE(reviewed_at, created_at)) "
                "FROM payments WHERE status='approved' GROUP BY telegram_id"
            )

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _dt(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    @staticmethod
    def _record_funnel_event_locked(
        conn: sqlite3.Connection,
        telegram_id: int,
        event_type: str,
        created_at: str,
    ) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO funnel_events (telegram_id, event_type, created_at) VALUES (?, ?, ?)",
            (telegram_id, event_type, created_at),
        )

    def record_funnel_event(self, telegram_id: int, event_type: str) -> None:
        if event_type not in {"miniapp_opened", "payment_started", "payment_paid"}:
            raise ValueError("Unknown funnel event.")
        with self._lock, self._connect() as conn:
            self._record_funnel_event_locked(
                conn, telegram_id, event_type, self._now().isoformat()
            )

    def access_info(self, telegram_id: int) -> AccessInfo:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT plan, status, expires_at FROM users WHERE telegram_id = ?",
                (telegram_id,),
            ).fetchone()
            if not row or row["status"] != "active":
                return AccessInfo(False)
            plan = row["plan"]
            if plan == "lifetime":
                return AccessInfo(True, "lifetime", None)
            expires = self._dt(row["expires_at"])
            if plan == "monthly" and expires and expires > self._now():
                return AccessInfo(True, "monthly", expires)
            conn.execute(
                "UPDATE users SET status='expired', updated_at=? WHERE telegram_id=?",
                (self._now().isoformat(), telegram_id),
            )
            return AccessInfo(False, plan, expires)

    def set_intent(
        self,
        telegram_id: int,
        username: str | None,
        first_name: str | None,
        plan_key: str,
        currency: str,
        amount_text: str,
    ) -> None:
        now = self._now().isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO payment_intents
                    (telegram_id, username, first_name, plan_key, currency, amount_text, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(telegram_id) DO UPDATE SET
                    username=excluded.username,
                    first_name=excluded.first_name,
                    plan_key=excluded.plan_key,
                    currency=excluded.currency,
                    amount_text=excluded.amount_text,
                    created_at=excluded.created_at
                """,
                (telegram_id, username, first_name, plan_key, currency, amount_text, now),
            )
            self._record_funnel_event_locked(conn, telegram_id, "payment_started", now)

    def get_intent(self, telegram_id: int) -> sqlite3.Row | None:
        with self._lock, self._connect() as conn:
            return conn.execute(
                "SELECT * FROM payment_intents WHERE telegram_id=?",
                (telegram_id,),
            ).fetchone()

    def consume_intent_and_create_payment(
        self,
        telegram_id: int,
        receipt_chat_id: int,
        receipt_message_id: int,
        receipt_file_id: str,
        receipt_kind: str,
    ) -> int | None:
        with self._lock, self._connect() as conn:
            intent = conn.execute(
                "SELECT * FROM payment_intents WHERE telegram_id=?",
                (telegram_id,),
            ).fetchone()
            if not intent:
                return None
            cur = conn.execute(
                """
                INSERT INTO payments
                    (telegram_id, username, first_name, plan_key, currency, amount_text,
                     receipt_chat_id, receipt_message_id, receipt_file_id, receipt_kind,
                     status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    telegram_id,
                    intent["username"],
                    intent["first_name"],
                    intent["plan_key"],
                    intent["currency"],
                    intent["amount_text"],
                    receipt_chat_id,
                    receipt_message_id,
                    receipt_file_id,
                    receipt_kind,
                    self._now().isoformat(),
                ),
            )
            conn.execute("DELETE FROM payment_intents WHERE telegram_id=?", (telegram_id,))
            return int(cur.lastrowid)

    def get_payment(self, payment_id: int) -> sqlite3.Row | None:
        with self._lock, self._connect() as conn:
            return conn.execute("SELECT * FROM payments WHERE id=?", (payment_id,)).fetchone()

    def latest_payment(self, telegram_id: int) -> sqlite3.Row | None:
        """Return the newest submitted payment for a Mini App user."""
        with self._lock, self._connect() as conn:
            return conn.execute(
                "SELECT * FROM payments WHERE telegram_id=? ORDER BY id DESC LIMIT 1",
                (telegram_id,),
            ).fetchone()

    def approve(self, payment_id: int, admin_id: int) -> sqlite3.Row | None:
        now = self._now()
        with self._lock, self._connect() as conn:
            payment = conn.execute(
                "SELECT * FROM payments WHERE id=? AND status='pending'",
                (payment_id,),
            ).fetchone()
            if not payment:
                return None

            user = conn.execute(
                "SELECT * FROM users WHERE telegram_id=?",
                (payment["telegram_id"],),
            ).fetchone()

            plan = payment["plan_key"]
            expires_at: datetime | None = None
            if plan == "monthly":
                base = now
                if user and user["status"] == "active" and user["plan"] == "monthly":
                    existing = self._dt(user["expires_at"])
                    if existing and existing > now:
                        base = existing
                expires_at = base + timedelta(days=30)

            conn.execute(
                """
                INSERT INTO users
                    (telegram_id, username, first_name, plan, status, activated_at, expires_at, updated_at, created_at, last_seen_at)
                VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)
                ON CONFLICT(telegram_id) DO UPDATE SET
                    username=excluded.username,
                    first_name=excluded.first_name,
                    plan=excluded.plan,
                    status='active',
                    activated_at=excluded.activated_at,
                    expires_at=excluded.expires_at,
                    updated_at=excluded.updated_at,
                    created_at=COALESCE(users.created_at, excluded.created_at),
                    last_seen_at=COALESCE(users.last_seen_at, excluded.last_seen_at)
                """,
                (
                    payment["telegram_id"],
                    payment["username"],
                    payment["first_name"],
                    plan,
                    now.isoformat(),
                    expires_at.isoformat() if expires_at else None,
                    now.isoformat(),
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            conn.execute(
                """
                UPDATE payments
                SET status='approved', reviewed_at=?, reviewed_by=?, updated_at=?
                WHERE id=?
                """,
                (now.isoformat(), admin_id, now.isoformat(), payment_id),
            )
            self._record_funnel_event_locked(
                conn, int(payment["telegram_id"]), "payment_paid", now.isoformat()
            )
            return conn.execute("SELECT * FROM payments WHERE id=?", (payment_id,)).fetchone()

    def reject(self, payment_id: int, admin_id: int) -> sqlite3.Row | None:
        now = self._now().isoformat()
        with self._lock, self._connect() as conn:
            payment = conn.execute(
                "SELECT * FROM payments WHERE id=? AND status='pending'",
                (payment_id,),
            ).fetchone()
            if not payment:
                return None
            conn.execute(
                "UPDATE payments SET status='rejected', reviewed_at=?, reviewed_by=? WHERE id=?",
                (now, admin_id, payment_id),
            )
            return conn.execute("SELECT * FROM payments WHERE id=?", (payment_id,)).fetchone()



    def cancel_latest_pending_payment(self, telegram_id: int) -> sqlite3.Row | None:
        """
        Locally abandon the user's newest pending payment attempt.

        This does not and cannot reverse a blockchain transaction. If NOWPayments
        later reports the abandoned payment as finished, access is still activated
        so a user who actually sent funds is never left without service.
        """
        now = self._now().isoformat()
        with self._lock, self._connect() as conn:
            payment = conn.execute(
                """
                SELECT * FROM payments
                WHERE telegram_id=? AND status='pending'
                ORDER BY id DESC
                LIMIT 1
                """,
                (telegram_id,),
            ).fetchone()
            if payment is None:
                return None

            conn.execute(
                "UPDATE payments SET status='cancelled', updated_at=? WHERE id=?",
                (now, int(payment["id"])),
            )
            # Manual checkout intents are safe to clear too.
            conn.execute(
                "DELETE FROM payment_intents WHERE telegram_id=?",
                (telegram_id,),
            )
            return conn.execute(
                "SELECT * FROM payments WHERE id=?",
                (int(payment["id"]),),
            ).fetchone()


    def record_nowpayments_payment(
        self,
        telegram_id: int,
        username: str | None,
        first_name: str | None,
        plan_key: str,
        currency: str,
        response: dict[str, Any],
        order_id: str,
        price_amount_usd: float,
    ) -> sqlite3.Row:
        """Persist a NOWPayments payment immediately after provider creation."""
        now = self._now().isoformat()
        provider_payment_id = str(response.get("payment_id") or "").strip()
        if not provider_payment_id:
            raise ValueError("NOWPayments did not return a payment ID.")
        pay_amount = float(response.get("pay_amount") or 0.0)
        pay_currency = str(response.get("pay_currency") or "").strip().lower()
        pay_address = str(response.get("pay_address") or "").strip()
        provider_status = str(response.get("payment_status") or "waiting").strip().lower()
        purchase_id = str(response.get("purchase_id") or "").strip() or None
        amount_text = f"{pay_amount:.8f}".rstrip("0").rstrip(".") + f" {pay_currency.upper()}"
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO payments
                    (telegram_id, username, first_name, plan_key, currency, amount_text,
                     status, created_at, provider, provider_payment_id, provider_order_id,
                     price_amount_usd, pay_amount, pay_currency, pay_address, provider_status,
                     purchase_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, 'nowpayments', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    telegram_id, username, first_name, plan_key, currency, amount_text,
                    str(response.get("created_at") or now), provider_payment_id, order_id,
                    round(float(price_amount_usd), 2), pay_amount, pay_currency, pay_address,
                    provider_status, purchase_id, now,
                ),
            )
            self._record_funnel_event_locked(conn, telegram_id, "payment_started", now)
            return conn.execute("SELECT * FROM payments WHERE id=?", (int(cur.lastrowid),)).fetchone()

    def nowpayments_payment_by_provider_id(self, provider_payment_id: str) -> sqlite3.Row | None:
        with self._lock, self._connect() as conn:
            return conn.execute(
                "SELECT * FROM payments WHERE provider='nowpayments' AND provider_payment_id=?",
                (str(provider_payment_id),),
            ).fetchone()

    def update_nowpayments_payment(self, payload: dict[str, Any]) -> tuple[sqlite3.Row | None, bool]:
        """Apply provider status and activate access once when status becomes finished."""
        provider_payment_id = str(payload.get("payment_id") or "").strip()
        order_id = str(payload.get("order_id") or "").strip()
        provider_status = str(payload.get("payment_status") or "").strip().lower()
        if not provider_payment_id and not order_id:
            return None, False
        now = self._now()
        now_iso = now.isoformat()
        with self._lock, self._connect() as conn:
            if provider_payment_id:
                payment = conn.execute(
                    "SELECT * FROM payments WHERE provider='nowpayments' AND provider_payment_id=?",
                    (provider_payment_id,),
                ).fetchone()
            else:
                payment = None
            if payment is None and order_id:
                payment = conn.execute(
                    "SELECT * FROM payments WHERE provider='nowpayments' AND provider_order_id=?",
                    (order_id,),
                ).fetchone()
            if payment is None:
                return None, False

            terminal_failure = provider_status in {"failed", "expired", "refunded"}
            if provider_status == "finished":
                local_status = "approved"
            elif payment["status"] == "cancelled":
                # Keep an abandoned attempt cancelled while the provider still
                # reports non-final states. A later "finished" still activates access.
                local_status = "cancelled"
            else:
                local_status = provider_status if terminal_failure else "pending"
            conn.execute(
                """
                UPDATE payments SET
                    provider_payment_id=COALESCE(NULLIF(?, ''), provider_payment_id),
                    provider_status=?, status=?,
                    pay_amount=COALESCE(?, pay_amount),
                    pay_currency=COALESCE(NULLIF(?, ''), pay_currency),
                    pay_address=COALESCE(NULLIF(?, ''), pay_address),
                    purchase_id=COALESCE(NULLIF(?, ''), purchase_id),
                    updated_at=?
                WHERE id=?
                """,
                (
                    provider_payment_id, provider_status, local_status,
                    float(payload["pay_amount"]) if payload.get("pay_amount") is not None else None,
                    str(payload.get("pay_currency") or ""),
                    str(payload.get("pay_address") or ""),
                    str(payload.get("purchase_id") or ""),
                    now_iso, int(payment["id"]),
                ),
            )

            newly_activated = provider_status == "finished" and payment["status"] != "approved"
            if newly_activated:
                current_user = conn.execute(
                    "SELECT * FROM users WHERE telegram_id=?",
                    (payment["telegram_id"],),
                ).fetchone()
                expires_at: datetime | None = None
                if payment["plan_key"] == "monthly":
                    base = now
                    if current_user and current_user["status"] == "active" and current_user["plan"] == "monthly":
                        existing = self._dt(current_user["expires_at"])
                        if existing and existing > now:
                            base = existing
                    expires_at = base + timedelta(days=30)
                conn.execute(
                    """
                    INSERT INTO users
                        (telegram_id, username, first_name, plan, status, activated_at, expires_at, updated_at, created_at, last_seen_at)
                    VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)
                    ON CONFLICT(telegram_id) DO UPDATE SET
                        username=COALESCE(excluded.username, users.username),
                        first_name=COALESCE(excluded.first_name, users.first_name),
                        plan=excluded.plan, status='active', activated_at=excluded.activated_at,
                        expires_at=excluded.expires_at, updated_at=excluded.updated_at,
                        created_at=COALESCE(users.created_at, excluded.created_at),
                        last_seen_at=COALESCE(users.last_seen_at, excluded.last_seen_at)
                    """,
                    (
                        payment["telegram_id"], payment["username"], payment["first_name"],
                        payment["plan_key"], now_iso,
                        expires_at.isoformat() if expires_at else None, now_iso, now_iso, now_iso,
                    ),
                )
                conn.execute(
                    "UPDATE payments SET reviewed_at=?, reviewed_by=NULL, updated_at=? WHERE id=?",
                    (now_iso, now_iso, int(payment["id"])),
                )
                self._record_funnel_event_locked(
                    conn, int(payment["telegram_id"]), "payment_paid", now_iso
                )

            updated = conn.execute(
                "SELECT * FROM payments WHERE id=?", (int(payment["id"]),)
            ).fetchone()
            return updated, newly_activated


    def _ensure_referral_code_locked(
        self,
        conn: sqlite3.Connection,
        telegram_id: int,
    ) -> str:
        row = conn.execute(
            "SELECT referral_code FROM users WHERE telegram_id=?",
            (telegram_id,),
        ).fetchone()
        if row is None:
            raise ValueError("User does not exist.")
        if row["referral_code"]:
            return str(row["referral_code"])

        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        for _ in range(50):
            code = "".join(secrets.choice(alphabet) for _ in range(8))
            try:
                conn.execute(
                    "UPDATE users SET referral_code=? WHERE telegram_id=? AND referral_code IS NULL",
                    (code, telegram_id),
                )
                stored = conn.execute(
                    "SELECT referral_code FROM users WHERE telegram_id=?",
                    (telegram_id,),
                ).fetchone()
                if stored and stored["referral_code"]:
                    return str(stored["referral_code"])
            except sqlite3.IntegrityError:
                continue
        raise RuntimeError("Could not generate a unique referral code.")

    def touch_user(
        self,
        telegram_id: int,
        username: str | None,
        first_name: str | None,
    ) -> None:
        """Register/update a Telegram user without assuming the Mini App was opened."""
        now = self._now().isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO users
                    (telegram_id, username, first_name, status, updated_at, created_at, last_seen_at)
                VALUES (?, ?, ?, 'inactive', ?, ?, ?)
                ON CONFLICT(telegram_id) DO UPDATE SET
                    username=excluded.username,
                    first_name=excluded.first_name,
                    last_seen_at=excluded.last_seen_at
                """,
                (telegram_id, username, first_name, now, now, now),
            )
            self._ensure_referral_code_locked(conn, telegram_id)

    def record_miniapp_opened(self, telegram_id: int) -> None:
        """Record the first verified Mini App open for conversion analytics."""
        now = self._now().isoformat()
        with self._lock, self._connect() as conn:
            self._record_funnel_event_locked(conn, telegram_id, "miniapp_opened", now)

    def attach_referrer(self, telegram_id: int, start_param: str | None) -> bool:
        """Attach the first valid referrer from Telegram's signed start_param.

        The relation is immutable. Self-referrals and users who already had an
        approved purchase are ignored.
        """
        raw = (start_param or "").strip()
        if not raw.lower().startswith("ref_"):
            return False
        code = raw[4:].strip().upper()
        if len(code) != 8 or any(ch not in "ABCDEFGHJKLMNPQRSTUVWXYZ23456789" for ch in code):
            return False

        with self._lock, self._connect() as conn:
            user = conn.execute(
                "SELECT telegram_id, referrer_id FROM users WHERE telegram_id=?",
                (telegram_id,),
            ).fetchone()
            if not user or user["referrer_id"] is not None:
                return False
            if conn.execute(
                "SELECT 1 FROM payments WHERE telegram_id=? AND status='approved' LIMIT 1",
                (telegram_id,),
            ).fetchone():
                return False
            referrer = conn.execute(
                "SELECT telegram_id FROM users WHERE referral_code=?",
                (code,),
            ).fetchone()
            if not referrer:
                return False
            referrer_id = int(referrer["telegram_id"])
            if referrer_id == telegram_id:
                return False
            conn.execute(
                "UPDATE users SET referrer_id=?, updated_at=? WHERE telegram_id=? AND referrer_id IS NULL",
                (referrer_id, self._now().isoformat(), telegram_id),
            )
            return conn.total_changes > 0

    @staticmethod
    def _clamp_percent(value: float) -> float:
        return max(0.0, min(100.0, float(value)))

    def effective_referral_percent(self, telegram_id: int, default_percent: float) -> float:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT referral_percent_override FROM users WHERE telegram_id=?",
                (telegram_id,),
            ).fetchone()
            if row and row["referral_percent_override"] is not None:
                return self._clamp_percent(float(row["referral_percent_override"]))
            return self._clamp_percent(default_percent)

    def set_referral_percent_override(
        self,
        telegram_id: int,
        percent: float | None,
    ) -> sqlite3.Row | None:
        with self._lock, self._connect() as conn:
            if not conn.execute(
                "SELECT 1 FROM users WHERE telegram_id=?",
                (telegram_id,),
            ).fetchone():
                return None
            value = None if percent is None else self._clamp_percent(percent)
            conn.execute(
                "UPDATE users SET referral_percent_override=?, updated_at=? WHERE telegram_id=?",
                (value, self._now().isoformat(), telegram_id),
            )
            return conn.execute(
                "SELECT * FROM users WHERE telegram_id=?",
                (telegram_id,),
            ).fetchone()

    def create_referral_reward_for_payment(
        self,
        payment_id: int,
        default_percent: float,
        monthly_usd: float,
        lifetime_usd: float,
    ) -> sqlite3.Row | None:
        """Create at most one first-purchase referral reward per referred user."""
        now = self._now()
        with self._lock, self._connect() as conn:
            self._expire_due(conn)
            payment = conn.execute(
                "SELECT * FROM payments WHERE id=? AND status='approved'",
                (payment_id,),
            ).fetchone()
            if not payment:
                return None
            referred_user_id = int(payment["telegram_id"])
            referred = conn.execute(
                "SELECT referrer_id FROM users WHERE telegram_id=?",
                (referred_user_id,),
            ).fetchone()
            if not referred or referred["referrer_id"] is None:
                return None
            existing = conn.execute(
                "SELECT * FROM referral_rewards WHERE referred_user_id=?",
                (referred_user_id,),
            ).fetchone()
            if existing:
                return existing

            referrer_id = int(referred["referrer_id"])
            referrer = conn.execute(
                "SELECT * FROM users WHERE telegram_id=?",
                (referrer_id,),
            ).fetchone()
            if not referrer:
                return None

            override = referrer["referral_percent_override"]
            percent = self._clamp_percent(
                float(override) if override is not None else default_percent
            )
            referrer_active = (
                referrer["status"] == "active"
                and (
                    referrer["plan"] == "lifetime"
                    or (
                        referrer["plan"] == "monthly"
                        and self._dt(referrer["expires_at"]) is not None
                        and self._dt(referrer["expires_at"]) > now
                    )
                )
            )
            payment_amount = (
                float(payment["price_amount_usd"])
                if "price_amount_usd" in payment.keys() and payment["price_amount_usd"] is not None
                else (
                    float(monthly_usd)
                    if payment["plan_key"] == "monthly"
                    else float(lifetime_usd)
                )
            )
            status = "earned" if referrer_active else "missed_inactive"
            commission = round(payment_amount * percent / 100.0, 2) if referrer_active else 0.0
            try:
                conn.execute(
                    """
                    INSERT INTO referral_rewards
                        (referrer_id, referred_user_id, payment_id, payment_amount_usd,
                         commission_percent, commission_amount_usd, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        referrer_id,
                        referred_user_id,
                        payment_id,
                        round(payment_amount, 2),
                        round(percent, 2),
                        commission,
                        status,
                        now.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError:
                pass
            return conn.execute(
                "SELECT * FROM referral_rewards WHERE referred_user_id=?",
                (referred_user_id,),
            ).fetchone()

    def referral_dashboard(
        self,
        telegram_id: int,
        default_percent: float,
        min_payout_usdt: float,
    ) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            self._expire_due(conn)
            row = conn.execute(
                "SELECT * FROM users WHERE telegram_id=?",
                (telegram_id,),
            ).fetchone()
            if not row:
                raise ValueError("User not found.")
            code = self._ensure_referral_code_locked(conn, telegram_id)
            override = row["referral_percent_override"]
            percent = self._clamp_percent(
                float(override) if override is not None else default_percent
            )
            invited = int(conn.execute(
                "SELECT COUNT(*) FROM users WHERE referrer_id=?",
                (telegram_id,),
            ).fetchone()[0])
            paid_referrals = int(conn.execute(
                "SELECT COUNT(*) FROM referral_rewards WHERE referrer_id=?",
                (telegram_id,),
            ).fetchone()[0])
            earned = float(conn.execute(
                "SELECT COALESCE(SUM(commission_amount_usd),0) FROM referral_rewards "
                "WHERE referrer_id=? AND status='earned'",
                (telegram_id,),
            ).fetchone()[0] or 0.0)
            pending = float(conn.execute(
                "SELECT COALESCE(SUM(amount_usdt),0) FROM referral_payouts "
                "WHERE telegram_id=? AND status='pending'",
                (telegram_id,),
            ).fetchone()[0] or 0.0)
            paid_out = float(conn.execute(
                "SELECT COALESCE(SUM(amount_usdt),0) FROM referral_payouts "
                "WHERE telegram_id=? AND status='paid'",
                (telegram_id,),
            ).fetchone()[0] or 0.0)
            latest_payouts = [dict(item) for item in conn.execute(
                "SELECT id, amount_usdt, wallet_address, status, created_at, reviewed_at "
                "FROM referral_payouts WHERE telegram_id=? ORDER BY id DESC LIMIT 5",
                (telegram_id,),
            ).fetchall()]
            available = max(0.0, round(earned - pending - paid_out, 2))
            return {
                "referral_code": code,
                "commission_percent": round(percent, 2),
                "uses_default_percent": override is None,
                "invited": invited,
                "paid_referrals": paid_referrals,
                "total_earned_usdt": round(earned, 2),
                "available_balance_usdt": available,
                "pending_payout_usdt": round(pending, 2),
                "paid_out_usdt": round(paid_out, 2),
                "min_payout_usdt": round(float(min_payout_usdt), 2),
                "payouts": latest_payouts,
            }

    def create_payout_request(
        self,
        telegram_id: int,
        amount_usdt: float,
        wallet_address: str,
        min_payout_usdt: float,
    ) -> sqlite3.Row:
        amount = round(float(amount_usdt), 2)
        minimum = round(float(min_payout_usdt), 2)
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("Enter a valid payout amount.")
        if amount < minimum:
            raise ValueError(f"Minimum payout is {minimum:g} USDT.")
        with self._lock, self._connect() as conn:
            earned = float(conn.execute(
                "SELECT COALESCE(SUM(commission_amount_usd),0) FROM referral_rewards "
                "WHERE referrer_id=? AND status='earned'",
                (telegram_id,),
            ).fetchone()[0] or 0.0)
            reserved = float(conn.execute(
                "SELECT COALESCE(SUM(amount_usdt),0) FROM referral_payouts "
                "WHERE telegram_id=? AND status IN ('pending','paid')",
                (telegram_id,),
            ).fetchone()[0] or 0.0)
            available = round(earned - reserved, 2)
            if amount > available + 1e-9:
                raise ValueError(f"Available referral balance is {max(0.0, available):.2f} USDT.")
            cur = conn.execute(
                """
                INSERT INTO referral_payouts
                    (telegram_id, amount_usdt, wallet_address, status, created_at)
                VALUES (?, ?, ?, 'pending', ?)
                """,
                (telegram_id, amount, wallet_address, self._now().isoformat()),
            )
            return conn.execute(
                "SELECT * FROM referral_payouts WHERE id=?",
                (int(cur.lastrowid),),
            ).fetchone()

    def pending_payouts(self, limit: int = 20) -> list[sqlite3.Row]:
        with self._lock, self._connect() as conn:
            return list(conn.execute(
                "SELECT * FROM referral_payouts WHERE status='pending' ORDER BY created_at ASC LIMIT ?",
                (limit,),
            ).fetchall())

    def get_payout(self, payout_id: int) -> sqlite3.Row | None:
        with self._lock, self._connect() as conn:
            return conn.execute(
                "SELECT * FROM referral_payouts WHERE id=?",
                (payout_id,),
            ).fetchone()

    def review_payout(
        self,
        payout_id: int,
        status: str,
        admin_id: int,
    ) -> sqlite3.Row | None:
        if status not in {"paid", "rejected"}:
            raise ValueError("Invalid payout status.")
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM referral_payouts WHERE id=? AND status='pending'",
                (payout_id,),
            ).fetchone()
            if not row:
                return None
            conn.execute(
                "UPDATE referral_payouts SET status=?, reviewed_at=?, reviewed_by=? WHERE id=?",
                (status, self._now().isoformat(), admin_id, payout_id),
            )
            return conn.execute(
                "SELECT * FROM referral_payouts WHERE id=?",
                (payout_id,),
            ).fetchone()

    def referral_admin_stats(self) -> dict[str, float | int]:
        with self._lock, self._connect() as conn:
            linked = int(conn.execute(
                "SELECT COUNT(*) FROM users WHERE referrer_id IS NOT NULL"
            ).fetchone()[0])
            rewards = int(conn.execute(
                "SELECT COUNT(*) FROM referral_rewards"
            ).fetchone()[0])
            earned = float(conn.execute(
                "SELECT COALESCE(SUM(commission_amount_usd),0) FROM referral_rewards WHERE status='earned'"
            ).fetchone()[0] or 0.0)
            missed = int(conn.execute(
                "SELECT COUNT(*) FROM referral_rewards WHERE status='missed_inactive'"
            ).fetchone()[0])
            pending_count = int(conn.execute(
                "SELECT COUNT(*) FROM referral_payouts WHERE status='pending'"
            ).fetchone()[0])
            pending_amount = float(conn.execute(
                "SELECT COALESCE(SUM(amount_usdt),0) FROM referral_payouts WHERE status='pending'"
            ).fetchone()[0] or 0.0)
            paid_out = float(conn.execute(
                "SELECT COALESCE(SUM(amount_usdt),0) FROM referral_payouts WHERE status='paid'"
            ).fetchone()[0] or 0.0)
            return {
                "linked_users": linked,
                "paid_referrals": rewards,
                "earned_usdt": round(earned, 2),
                "missed_inactive": missed,
                "pending_payouts": pending_count,
                "pending_payout_usdt": round(pending_amount, 2),
                "paid_out_usdt": round(paid_out, 2),
            }

    def _expire_due(self, conn: sqlite3.Connection) -> None:
        now = self._now().isoformat()
        conn.execute(
            """
            UPDATE users
            SET status='expired', updated_at=?
            WHERE status='active' AND plan='monthly'
              AND expires_at IS NOT NULL AND expires_at <= ?
            """,
            (now, now),
        )

    def get_user(self, telegram_id: int) -> sqlite3.Row | None:
        with self._lock, self._connect() as conn:
            self._expire_due(conn)
            return conn.execute(
                "SELECT * FROM users WHERE telegram_id=?",
                (telegram_id,),
            ).fetchone()

    def find_user(self, query: str) -> sqlite3.Row | None:
        query = query.strip()
        with self._lock, self._connect() as conn:
            self._expire_due(conn)
            if query.lstrip("-").isdigit():
                return conn.execute(
                    "SELECT * FROM users WHERE telegram_id=?",
                    (int(query),),
                ).fetchone()
            username = query.lstrip("@").strip()
            if not username:
                return None
            return conn.execute(
                "SELECT * FROM users WHERE username = ? COLLATE NOCASE",
                (username,),
            ).fetchone()

    def cancel_access(self, telegram_id: int) -> sqlite3.Row | None:
        now = self._now().isoformat()
        with self._lock, self._connect() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE telegram_id=?",
                (telegram_id,),
            ).fetchone()
            if not user:
                return None
            conn.execute(
                "UPDATE users SET status='cancelled', updated_at=? WHERE telegram_id=?",
                (now, telegram_id),
            )
            return conn.execute(
                "SELECT * FROM users WHERE telegram_id=?",
                (telegram_id,),
            ).fetchone()

    def grant_monthly(self, telegram_id: int) -> sqlite3.Row | None:
        now = self._now()
        with self._lock, self._connect() as conn:
            self._expire_due(conn)
            user = conn.execute(
                "SELECT * FROM users WHERE telegram_id=?",
                (telegram_id,),
            ).fetchone()
            if not user:
                return None
            base = now
            existing = self._dt(user["expires_at"])
            if (
                user["status"] == "active"
                and user["plan"] == "monthly"
                and existing
                and existing > now
            ):
                base = existing
            expires = base + timedelta(days=30)
            activated = user["activated_at"] or now.isoformat()
            conn.execute(
                """
                UPDATE users
                SET plan='monthly', status='active', activated_at=?, expires_at=?, updated_at=?
                WHERE telegram_id=?
                """,
                (activated, expires.isoformat(), now.isoformat(), telegram_id),
            )
            return conn.execute(
                "SELECT * FROM users WHERE telegram_id=?",
                (telegram_id,),
            ).fetchone()

    def grant_lifetime(self, telegram_id: int) -> sqlite3.Row | None:
        now = self._now().isoformat()
        with self._lock, self._connect() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE telegram_id=?",
                (telegram_id,),
            ).fetchone()
            if not user:
                return None
            activated = user["activated_at"] or now
            conn.execute(
                """
                UPDATE users
                SET plan='lifetime', status='active', activated_at=?, expires_at=NULL, updated_at=?
                WHERE telegram_id=?
                """,
                (activated, now, telegram_id),
            )
            return conn.execute(
                "SELECT * FROM users WHERE telegram_id=?",
                (telegram_id,),
            ).fetchone()

    @staticmethod
    def _user_filter_sql(filter_key: str, now_iso: str) -> tuple[str, tuple]:
        if filter_key == "active":
            return (
                "status='active' AND (plan='lifetime' OR (plan='monthly' AND expires_at > ?))",
                (now_iso,),
            )
        if filter_key == "monthly":
            return (
                "status='active' AND plan='monthly' AND expires_at > ?",
                (now_iso,),
            )
        if filter_key == "lifetime":
            return ("status='active' AND plan='lifetime'", ())
        if filter_key == "expired":
            return ("status='expired'", ())
        if filter_key == "cancelled":
            return ("status='cancelled'", ())
        if filter_key == "inactive":
            return ("status='inactive'", ())
        return ("1=1", ())

    def list_users(
        self,
        filter_key: str = "all",
        limit: int = 8,
        offset: int = 0,
    ) -> tuple[list[sqlite3.Row], int]:
        now_iso = self._now().isoformat()
        with self._lock, self._connect() as conn:
            self._expire_due(conn)
            where, params = self._user_filter_sql(filter_key, now_iso)
            total = conn.execute(
                f"SELECT COUNT(*) FROM users WHERE {where}",
                params,
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT * FROM users WHERE {where} "
                "ORDER BY COALESCE(last_seen_at, created_at, updated_at) DESC "
                "LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
            return list(rows), int(total)

    def broadcast_user_ids(self, segment: str) -> list[int]:
        rows, _ = self.list_users(segment, limit=1_000_000, offset=0)
        return [int(row["telegram_id"]) for row in rows]

    def pending_payments(self, limit: int = 20) -> list[sqlite3.Row]:
        with self._lock, self._connect() as conn:
            return list(
                conn.execute(
                    "SELECT * FROM payments WHERE status='pending' AND COALESCE(provider,'manual')='manual' ORDER BY created_at ASC LIMIT ?",
                    (limit,),
                ).fetchall()
            )

    def business_statistics(self, monthly_usd: float, lifetime_usd: float) -> dict[str, Any]:
        now = self._now()
        now_iso = now.isoformat()
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        week_start = (now - timedelta(days=7)).isoformat()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
        with self._lock, self._connect() as conn:
            self._expire_due(conn)
            def scalar(query: str, params: tuple = ()):
                row = conn.execute(query, params).fetchone()
                return row[0] if row else 0
            active_monthly = int(scalar(
                "SELECT COUNT(*) FROM users WHERE status='active' AND plan='monthly' AND expires_at > ?",
                (now_iso,),
            ) or 0)
            active_lifetime = int(scalar(
                "SELECT COUNT(*) FROM users WHERE status='active' AND plan='lifetime'"
            ) or 0)
            opened = int(scalar("SELECT COUNT(*) FROM funnel_events WHERE event_type='miniapp_opened'") or 0)
            started = int(scalar("SELECT COUNT(*) FROM funnel_events WHERE event_type='payment_started'") or 0)
            paid = int(scalar("SELECT COUNT(*) FROM funnel_events WHERE event_type='payment_paid'") or 0)
            revenue = float(scalar(
                """SELECT COALESCE(SUM(CASE
                    WHEN price_amount_usd IS NOT NULL THEN price_amount_usd
                    WHEN plan_key='monthly' THEN ? ELSE ? END), 0)
                    FROM payments WHERE status='approved'""",
                (float(monthly_usd), float(lifetime_usd)),
            ) or 0.0)
            revenue_month = float(scalar(
                """SELECT COALESCE(SUM(CASE
                    WHEN price_amount_usd IS NOT NULL THEN price_amount_usd
                    WHEN plan_key='monthly' THEN ? ELSE ? END), 0)
                    FROM payments WHERE status='approved' AND COALESCE(reviewed_at, created_at) >= ?""",
                (float(monthly_usd), float(lifetime_usd), month_start),
            ) or 0.0)
            referral_turnover = float(scalar(
                "SELECT COALESCE(SUM(payment_amount_usd),0) FROM referral_rewards"
            ) or 0.0)
            referral_commission = float(scalar(
                "SELECT COALESCE(SUM(commission_amount_usd),0) FROM referral_rewards WHERE status='earned'"
            ) or 0.0)
            paid_out = float(scalar(
                "SELECT COALESCE(SUM(amount_usdt),0) FROM referral_payouts WHERE status='paid'"
            ) or 0.0)
            nowpayments_paid = int(scalar(
                "SELECT COUNT(*) FROM payments WHERE provider='nowpayments' AND status='approved'"
            ) or 0)
            nowpayments_pending = int(scalar(
                "SELECT COUNT(*) FROM payments WHERE provider='nowpayments' AND status='pending'"
            ) or 0)
            pct = lambda a, b: round((float(a) / float(b) * 100.0), 1) if b else 0.0
            return {
                "mrr_usd": round(active_monthly * float(monthly_usd), 2),
                "active_monthly": active_monthly,
                "active_lifetime": active_lifetime,
                "new_today": int(scalar("SELECT COUNT(*) FROM users WHERE created_at >= ?", (day_start,)) or 0),
                "new_7d": int(scalar("SELECT COUNT(*) FROM users WHERE created_at >= ?", (week_start,)) or 0),
                "new_month": int(scalar("SELECT COUNT(*) FROM users WHERE created_at >= ?", (month_start,)) or 0),
                "revenue_usd": round(revenue, 2),
                "revenue_month_usd": round(revenue_month, 2),
                "funnel_opened": opened,
                "funnel_started": started,
                "funnel_paid": paid,
                "conversion_open_to_start": pct(started, opened),
                "conversion_open_to_paid": pct(paid, opened),
                "conversion_start_to_paid": pct(paid, started),
                "referral_turnover_usd": round(referral_turnover, 2),
                "referral_commission_usd": round(referral_commission, 2),
                "referral_paid_out_usdt": round(paid_out, 2),
                "nowpayments_paid": nowpayments_paid,
                "nowpayments_pending": nowpayments_pending,
            }

    def referral_leaderboard(self, limit: int = 10) -> list[dict[str, Any]]:
        limit = max(1, min(50, int(limit)))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    u.telegram_id, u.username, u.first_name,
                    COUNT(rr.id) AS paid_referrals,
                    COALESCE(SUM(rr.payment_amount_usd), 0) AS turnover_usd,
                    COALESCE(SUM(CASE WHEN rr.status='earned' THEN rr.commission_amount_usd ELSE 0 END), 0) AS earned_usdt,
                    COALESCE((SELECT SUM(rp.amount_usdt) FROM referral_payouts rp
                              WHERE rp.telegram_id=u.telegram_id AND rp.status='paid'), 0) AS paid_out_usdt
                FROM users u
                JOIN referral_rewards rr ON rr.referrer_id=u.telegram_id
                GROUP BY u.telegram_id, u.username, u.first_name
                ORDER BY paid_referrals DESC, turnover_usd DESC, earned_usdt DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def statistics(self) -> dict[str, int]:
        now = self._now()
        now_iso = now.isoformat()
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        month_start = now.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        ).isoformat()
        with self._lock, self._connect() as conn:
            self._expire_due(conn)

            def count(query: str, params: tuple = ()) -> int:
                return int(conn.execute(query, params).fetchone()[0])

            return {
                "total": count("SELECT COUNT(*) FROM users"),
                "active": count(
                    "SELECT COUNT(*) FROM users WHERE status='active' "
                    "AND (plan='lifetime' OR (plan='monthly' AND expires_at > ?))",
                    (now_iso,),
                ),
                "monthly": count(
                    "SELECT COUNT(*) FROM users WHERE status='active' "
                    "AND plan='monthly' AND expires_at > ?",
                    (now_iso,),
                ),
                "lifetime": count(
                    "SELECT COUNT(*) FROM users WHERE status='active' AND plan='lifetime'"
                ),
                "inactive": count("SELECT COUNT(*) FROM users WHERE status='inactive'"),
                "expired": count("SELECT COUNT(*) FROM users WHERE status='expired'"),
                "cancelled": count("SELECT COUNT(*) FROM users WHERE status='cancelled'"),
                "new_today": count(
                    "SELECT COUNT(*) FROM users WHERE created_at >= ?", (day_start,)
                ),
                "new_month": count(
                    "SELECT COUNT(*) FROM users WHERE created_at >= ?", (month_start,)
                ),
                "pending": count("SELECT COUNT(*) FROM payments WHERE status='pending'"),
                "approved": count("SELECT COUNT(*) FROM payments WHERE status='approved'"),
                "rejected": count("SELECT COUNT(*) FROM payments WHERE status='rejected'"),
            }



# -----------------------------------------------------------------------------
# Telegram UI helpers
# -----------------------------------------------------------------------------

def green_button(text: str, callback_data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=callback_data, style="success")


def danger_button(text: str, callback_data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=callback_data, style="danger")


def admin_review_keyboard(payment_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [green_button("✅ Подтвердить", f"admin:approve:{payment_id}")],
            [danger_button("❌ Отклонить", f"admin:reject:{payment_id}")],
        ]
    )


def miniapp_keyboard(settings: Settings) -> InlineKeyboardMarkup | None:
    """Customer chat is only a launcher; all customer actions happen in Mini App."""
    if not settings.miniapp_url:
        return None
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🚀 Open MENTAL TRADER",
                    web_app=WebAppInfo(url=settings.miniapp_url),
                )
            ]
        ]
    )


def payment_amount(settings: Settings, currency: str, plan_key: str) -> str | None:
    """Server-authoritative payment amount used by the Mini App API."""
    if plan_key == "monthly":
        usd = settings.monthly_usd
    elif plan_key == "lifetime":
        usd = settings.lifetime_usd
    else:
        return None
    if currency == "usdt":
        return f"{usd:g} USDT"
    if currency == "btc":
        if settings.btc_usd_rate <= 0:
            return None
        return f"{usd / settings.btc_usd_rate:.8f} BTC"
    return None


# -----------------------------------------------------------------------------
# Telegram handlers
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)
router = Router(name=__name__)


def _parse_market(raw: str) -> Market | None:
    try:
        return Market(raw)
    except ValueError:
        return None


def has_access(user_id: int, store: AccessStore, settings: Settings) -> bool:
    return user_id == settings.admin_telegram_id or store.access_info(user_id).active


async def show_intro_message(message: Message, store: AccessStore, settings: Settings) -> None:
    """Customer chat is read-only apart from launching the Mini App."""
    user_id = message.from_user.id if message.from_user else message.chat.id
    active = has_access(user_id, store, settings)
    status = (
        "Your access is active."
        if active
        else "Open the Mini App to choose a plan and activate access."
    )
    await message.answer(
        "🤖 <b>MENTAL TRADER</b>\n\n"
        "All customer features — plans, payment, receipt upload, markets and trading signals — "
        "are available only inside the Telegram Mini App.\n\n"
        f"{status}",
        parse_mode=ParseMode.HTML,
        reply_markup=miniapp_keyboard(settings),
    )


@router.message(CommandStart())
async def start(message: Message, access_store: AccessStore, settings: Settings) -> None:
    if message.from_user:
        access_store.touch_user(
            message.from_user.id,
            message.from_user.username,
            message.from_user.first_name,
        )
    await show_intro_message(message, access_store, settings)


@router.message(Command("id"))
async def show_my_id(message: Message, settings: Settings) -> None:
    if message.from_user and _is_admin(message.from_user.id, settings):
        await message.answer(
            f"Telegram ID: <code>{message.from_user.id}</code>",
            parse_mode=ParseMode.HTML,
        )


async def _customer_callback_to_miniapp(
    callback: CallbackQuery,
    settings: Settings,
) -> None:
    await callback.answer("Open MENTAL TRADER Mini App.", show_alert=False)
    if callback.message:
        try:
            await callback.message.edit_text(
                "🤖 <b>MENTAL TRADER</b>\n\n"
                "This customer action is available only inside the Mini App.",
                parse_mode=ParseMode.HTML,
                reply_markup=miniapp_keyboard(settings),
            )
        except TelegramBadRequest:
            pass


@router.callback_query(
    (F.data == "intro")
    | (F.data == "get_bot")
    | (F.data == "plans")
    | (F.data == "access:open")
    | (F.data == "home")
)
async def legacy_customer_root_callback(
    callback: CallbackQuery,
    settings: Settings,
) -> None:
    await _customer_callback_to_miniapp(callback, settings)


@router.callback_query(
    F.data.startswith("plan:")
    | F.data.startswith("payment_methods:")
    | F.data.startswith("payment:")
    | F.data.startswith("address:")
    | F.data.startswith("market:")
    | F.data.startswith("asset:")
    | F.data.startswith("signal:")
    | F.data.startswith("price:")
)
async def legacy_customer_callback(
    callback: CallbackQuery,
    settings: Settings,
) -> None:
    await _customer_callback_to_miniapp(callback, settings)


def _is_admin(user_id: int, settings: Settings) -> bool:
    return bool(settings.admin_telegram_id and user_id == settings.admin_telegram_id)


async def _render_admin_panel(message: Message, access_store: AccessStore) -> None:
    stats = access_store.statistics()
    await message.edit_text(
        "🛠 <b>АДМИН-ПАНЕЛЬ</b>\n\n"
        f"Пользователей: <b>{stats['total']}</b>\n"
        f"Активных подписок: <b>{stats['active']}</b>\n"
        f"Ожидают проверки: <b>{stats['pending']}</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=admin_panel_keyboard(),
    )


@router.message(Command("admin"))
async def admin_command(
    message: Message,
    settings: Settings,
    access_store: AccessStore,
    state: FSMContext,
) -> None:
    if not message.from_user or not _is_admin(message.from_user.id, settings):
        return
    await state.clear()
    stats = access_store.statistics()
    await message.answer(
        "🛠 <b>АДМИН-ПАНЕЛЬ</b>\n\n"
        f"Пользователей: <b>{stats['total']}</b>\n"
        f"Активных подписок: <b>{stats['active']}</b>\n"
        f"Ожидают проверки: <b>{stats['pending']}</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=admin_panel_keyboard(),
    )


@router.callback_query(F.data == "adm:panel")
async def admin_panel_callback(
    callback: CallbackQuery,
    settings: Settings,
    access_store: AccessStore,
    state: FSMContext,
) -> None:
    if not _is_admin(callback.from_user.id, settings):
        await callback.answer("Недоступно.", show_alert=True)
        return
    await callback.answer()
    await state.clear()
    if callback.message:
        await _render_admin_panel(callback.message, access_store)


@router.callback_query(F.data == "adm:stats")
async def admin_stats(
    callback: CallbackQuery,
    settings: Settings,
    access_store: AccessStore,
) -> None:
    if not _is_admin(callback.from_user.id, settings):
        await callback.answer("Недоступно.", show_alert=True)
        return
    await callback.answer()
    if not callback.message:
        return
    s = access_store.statistics()
    b = access_store.business_statistics(settings.monthly_usd, settings.lifetime_usd)
    await callback.message.edit_text(
        "📊 <b>БИЗНЕС-СТАТИСТИКА</b>\n\n"
        f"💵 MRR: <b>${b['mrr_usd']:.2f}</b> (активные Monthly × цена)\n"
        f"💰 Выручка всего: <b>${b['revenue_usd']:.2f}</b>\n"
        f"🗓 Выручка в этом месяце: <b>${b['revenue_month_usd']:.2f}</b>\n\n"
        f"✅ Активных: <b>{s['active']}</b> · Monthly <b>{b['active_monthly']}</b> · Lifetime <b>{b['active_lifetime']}</b>\n"
        f"🆕 Новые: сегодня <b>{b['new_today']}</b> · 7 дней <b>{b['new_7d']}</b> · месяц <b>{b['new_month']}</b>\n\n"
        "<b>Воронка Mini App</b>\n"
        f"Открыли: <b>{b['funnel_opened']}</b>\n"
        f"Начали оплату: <b>{b['funnel_started']}</b> · <b>{b['conversion_open_to_start']:.1f}%</b> от открывших\n"
        f"Оплатили: <b>{b['funnel_paid']}</b> · <b>{b['conversion_open_to_paid']:.1f}%</b> от открывших · <b>{b['conversion_start_to_paid']:.1f}%</b> от начавших\n\n"
        "<b>Реферальная экономика</b>\n"
        f"Оборот: <b>${b['referral_turnover_usd']:.2f}</b>\n"
        f"Начислено комиссий: <b>{b['referral_commission_usd']:.2f} USDT</b>\n"
        f"Выплачено: <b>{b['referral_paid_out_usdt']:.2f} USDT</b>\n\n"
        f"NOWPayments: завершено <b>{b['nowpayments_paid']}</b> · в процессе <b>{b['nowpayments_pending']}</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=admin_back_keyboard(),
    )


@router.callback_query(F.data == "adm:leaderboard")
async def admin_referral_leaderboard(
    callback: CallbackQuery,
    settings: Settings,
    access_store: AccessStore,
) -> None:
    if not _is_admin(callback.from_user.id, settings):
        await callback.answer("Недоступно.", show_alert=True)
        return
    await callback.answer()
    if not callback.message:
        return
    rows = access_store.referral_leaderboard(10)
    if not rows:
        text = "🏆 <b>ЛИДЕРБОРД ПАРТНЁРОВ</b>\n\nПока нет оплаченных рефералов."
    else:
        lines = ["🏆 <b>ЛИДЕРБОРД ПАРТНЁРОВ</b>", ""]
        for index, row in enumerate(rows, 1):
            name = f"@{row['username']}" if row.get('username') else (row.get('first_name') or str(row['telegram_id']))
            lines.append(
                f"<b>{index}. {escape(str(name))}</b> — оплат <b>{int(row['paid_referrals'])}</b>\n"
                f"   Оборот: <b>${float(row['turnover_usd']):.2f}</b> · начислено <b>{float(row['earned_usdt']):.2f} USDT</b> · выплачено <b>{float(row['paid_out_usdt']):.2f} USDT</b>"
            )
        text = "\n".join(lines)
    await callback.message.edit_text(
        text,
        parse_mode=ParseMode.HTML,
        reply_markup=admin_back_keyboard(),
    )


@router.callback_query(F.data == "adm:referrals")
async def admin_referrals(
    callback: CallbackQuery,
    settings: Settings,
    access_store: AccessStore,
) -> None:
    if not _is_admin(callback.from_user.id, settings):
        await callback.answer("Недоступно.", show_alert=True)
        return
    await callback.answer()
    if not callback.message:
        return
    s = access_store.referral_admin_stats()
    await callback.message.edit_text(
        "🤝 <b>РЕФЕРАЛЬНАЯ СИСТЕМА</b>\n\n"
        f"Стандартная ставка: <b>{settings.referral_default_percent:g}%</b>\n"
        f"Минимальная выплата: <b>{settings.referral_min_payout_usdt:g} USDT</b>\n\n"
        f"Привязано рефералов: <b>{s['linked_users']}</b>\n"
        f"Первые покупки: <b>{s['paid_referrals']}</b>\n"
        f"Начислено комиссий: <b>{s['earned_usdt']:.2f} USDT</b>\n"
        f"Без комиссии из-за неактивной подписки: <b>{s['missed_inactive']}</b>\n\n"
        f"Ожидают выплат: <b>{s['pending_payouts']}</b> на <b>{s['pending_payout_usdt']:.2f} USDT</b>\n"
        f"Выплачено: <b>{s['paid_out_usdt']:.2f} USDT</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=admin_back_keyboard(),
    )


@router.callback_query(F.data == "adm:payouts")
async def admin_payouts(
    callback: CallbackQuery,
    settings: Settings,
    access_store: AccessStore,
) -> None:
    if not _is_admin(callback.from_user.id, settings):
        await callback.answer("Недоступно.", show_alert=True)
        return
    await callback.answer()
    if not callback.message:
        return
    rows = access_store.pending_payouts(20)
    buttons: list[list[InlineKeyboardButton]] = []
    for row in rows:
        user = access_store.get_user(int(row["telegram_id"]))
        name = _admin_user_name(user) if user else str(row["telegram_id"])
        buttons.append([
            InlineKeyboardButton(
                text=f"💸 #{row['id']} · {name[:16]} · {float(row['amount_usdt']):.2f} USDT",
                callback_data=f"adm:po:{row['id']}",
            )
        ])
    buttons.append([InlineKeyboardButton(text="⬅️ Админ-панель", callback_data="adm:panel")])
    await callback.message.edit_text(
        "💸 <b>ЗАПРОСЫ НА ВЫПЛАТУ</b>\n\n"
        + (f"Ожидают: <b>{len(rows)}</b>" if rows else "Новых запросов нет."),
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
    )


@router.callback_query(F.data.startswith("adm:po:"))
async def admin_payout_detail(
    callback: CallbackQuery,
    settings: Settings,
    access_store: AccessStore,
) -> None:
    if not _is_admin(callback.from_user.id, settings):
        await callback.answer("Недоступно.", show_alert=True)
        return
    try:
        payout_id = int(callback.data.rsplit(":", 1)[1])
    except ValueError:
        await callback.answer("Некорректный ID.", show_alert=True)
        return
    row = access_store.get_payout(payout_id)
    if not row:
        await callback.answer("Запрос не найден.", show_alert=True)
        return
    await callback.answer()
    if not callback.message:
        return
    user = access_store.get_user(int(row["telegram_id"]))
    username = f"@{user['username']}" if user and user["username"] else "—"
    keyboard = (
        admin_payout_review_keyboard(payout_id)
        if row["status"] == "pending"
        else admin_back_keyboard()
    )
    await callback.message.edit_text(
        "💸 <b>ВЫПЛАТА ПО РЕФЕРАЛКЕ</b>\n\n"
        f"ID: <code>{payout_id}</code>\n"
        f"Пользователь: {escape(username)}\n"
        f"Telegram ID: <code>{row['telegram_id']}</code>\n"
        f"Сумма: <b>{float(row['amount_usdt']):.2f} USDT</b>\n"
        f"Кошелёк TRC20: <code>{escape(row['wallet_address'])}</code>\n"
        f"Статус: <b>{escape(row['status'])}</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard,
    )


@router.callback_query(F.data == "adm:users")
async def admin_users(callback: CallbackQuery, settings: Settings) -> None:
    if not _is_admin(callback.from_user.id, settings):
        await callback.answer("Недоступно.", show_alert=True)
        return
    await callback.answer()
    if callback.message:
        await callback.message.edit_text(
            "👥 <b>ПОЛЬЗОВАТЕЛИ</b>\n\nВыбери категорию:",
            parse_mode=ParseMode.HTML,
            reply_markup=admin_users_menu_keyboard(),
        )


@router.callback_query(F.data.startswith("adm:ul:"))
async def admin_user_list(
    callback: CallbackQuery,
    settings: Settings,
    access_store: AccessStore,
) -> None:
    if not _is_admin(callback.from_user.id, settings):
        await callback.answer("Недоступно.", show_alert=True)
        return
    await callback.answer()
    if not callback.message:
        return
    parts = callback.data.split(":")
    if len(parts) != 4:
        return
    filter_key = parts[2]
    if filter_key not in ADMIN_FILTER_NAMES:
        filter_key = "all"
    try:
        page = max(0, int(parts[3]))
    except ValueError:
        page = 0

    per_page = 8
    rows, total = access_store.list_users(filter_key, per_page, page * per_page)
    max_page = max(0, (total - 1) // per_page)
    if page > max_page:
        page = max_page
        rows, total = access_store.list_users(filter_key, per_page, page * per_page)

    buttons: list[list[InlineKeyboardButton]] = []
    for row in rows:
        name = _admin_user_name(row)[:22]
        label = f"{name} · {_admin_status_label(row)}"
        buttons.append(
            [
                InlineKeyboardButton(
                    text=label,
                    callback_data=f"adm:u:{row['telegram_id']}:view",
                )
            ]
        )

    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(
            InlineKeyboardButton(
                text="⬅️",
                callback_data=f"adm:ul:{filter_key}:{page - 1}",
            )
        )
    nav.append(
        InlineKeyboardButton(
            text=f"{page + 1}/{max_page + 1}",
            callback_data="adm:none",
        )
    )
    if page < max_page:
        nav.append(
            InlineKeyboardButton(
                text="➡️",
                callback_data=f"adm:ul:{filter_key}:{page + 1}",
            )
        )
    buttons.append(nav)
    buttons.append([InlineKeyboardButton(text="⬅️ Категории", callback_data="adm:users")])

    await callback.message.edit_text(
        f"👥 <b>{escape(ADMIN_FILTER_NAMES[filter_key])}</b> — {total}\n\n"
        "Выбери пользователя:",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
    )


@router.callback_query(F.data == "adm:none")
async def admin_noop(callback: CallbackQuery, settings: Settings) -> None:
    if _is_admin(callback.from_user.id, settings):
        await callback.answer()


@router.callback_query(F.data == "adm:find")
async def admin_find_start(
    callback: CallbackQuery,
    settings: Settings,
    state: FSMContext,
) -> None:
    if not _is_admin(callback.from_user.id, settings):
        await callback.answer("Недоступно.", show_alert=True)
        return
    await callback.answer()
    await state.set_state(AdminStates.waiting_find_user)
    if callback.message:
        await callback.message.edit_text(
            "🔍 <b>ПОИСК ПОЛЬЗОВАТЕЛЯ</b>\n\n"
            "Отправь Telegram ID или @username.",
            parse_mode=ParseMode.HTML,
            reply_markup=admin_back_keyboard(),
        )


@router.message(AdminStates.waiting_find_user, F.text)
async def admin_find_message(
    message: Message,
    settings: Settings,
    access_store: AccessStore,
    state: FSMContext,
) -> None:
    if not message.from_user or not _is_admin(message.from_user.id, settings):
        return
    row = access_store.find_user(message.text or "")
    await state.clear()
    if not row:
        await message.answer(
            "❌ Пользователь не найден. Пользователь должен хотя бы один раз нажать /start.",
            reply_markup=admin_back_keyboard(),
        )
        return
    await message.answer(
        admin_user_text(row, settings.referral_default_percent),
        parse_mode=ParseMode.HTML,
        reply_markup=admin_user_keyboard(row),
    )


@router.callback_query(F.data.startswith("adm:u:"))
async def admin_user_action(
    callback: CallbackQuery,
    settings: Settings,
    access_store: AccessStore,
    bot: Bot,
    state: FSMContext,
) -> None:
    if not _is_admin(callback.from_user.id, settings):
        await callback.answer("Недоступно.", show_alert=True)
        return
    parts = callback.data.split(":")
    if len(parts) != 4:
        await callback.answer()
        return
    try:
        user_id = int(parts[2])
    except ValueError:
        await callback.answer("Некорректный ID.", show_alert=True)
        return
    action = parts[3]
    row = access_store.get_user(user_id)
    if not row:
        await callback.answer("Пользователь не найден.", show_alert=True)
        return
    if not callback.message:
        await callback.answer()
        return

    if action == "view":
        await callback.answer()
        await callback.message.edit_text(
            admin_user_text(row, settings.referral_default_percent),
            parse_mode=ParseMode.HTML,
            reply_markup=admin_user_keyboard(row),
        )
        return

    if action == "refpct":
        await callback.answer()
        override = row["referral_percent_override"]
        effective = (
            float(override)
            if override is not None
            else float(settings.referral_default_percent)
        )
        mode = "персональный" if override is not None else "по умолчанию"
        await callback.message.edit_text(
            "💰 <b>РЕФЕРАЛЬНАЯ СТАВКА</b>\n\n"
            f"Пользователь: <code>{user_id}</code>\n"
            f"Текущая ставка: <b>{effective:g}%</b> ({mode})\n"
            f"Глобальная ставка: <b>{settings.referral_default_percent:g}%</b>\n\n"
            "Выбери новое значение:",
            parse_mode=ParseMode.HTML,
            reply_markup=admin_referral_percent_keyboard(user_id),
        )
        return

    if action == "cancelask":
        await callback.answer()
        await callback.message.edit_text(
            admin_user_text(row, settings.referral_default_percent)
            + "\n\n⚠️ <b>Отменить доступ этому пользователю?</b>",
            parse_mode=ParseMode.HTML,
            reply_markup=admin_cancel_confirm_keyboard(user_id),
        )
        return

    if action == "cancel":
        updated = access_store.cancel_access(user_id)
        await callback.answer("Подписка отменена.")
        if updated:
            try:
                await bot.send_message(
                    user_id,
                    "🔒 <b>Your access has been disabled by the administrator.</b>\n\n"
                    "Please contact support if you believe this is a mistake.",
                    parse_mode=ParseMode.HTML,
                )
            except (TelegramForbiddenError, TelegramBadRequest):
                pass
            await callback.message.edit_text(
                admin_user_text(updated, settings.referral_default_percent),
                parse_mode=ParseMode.HTML,
                reply_markup=admin_user_keyboard(updated),
            )
        return

    if action == "monthly":
        updated = access_store.grant_monthly(user_id)
        await callback.answer("Добавлено 30 дней.")
        if updated:
            try:
                exp = AccessStore._dt(updated["expires_at"])
                exp_text = exp.strftime("%d %B %Y") if exp else "30 days"
                await bot.send_message(
                    user_id,
                    "✅ <b>Your access has been activated/extended for 30 days.</b>\n\n"
                    f"Expires: <b>{escape(exp_text)}</b>",
                    parse_mode=ParseMode.HTML,
                    reply_markup=miniapp_keyboard(settings),
                )
            except (TelegramForbiddenError, TelegramBadRequest):
                pass
            await callback.message.edit_text(
                admin_user_text(updated, settings.referral_default_percent),
                parse_mode=ParseMode.HTML,
                reply_markup=admin_user_keyboard(updated),
            )
        return

    if action == "lifetime":
        updated = access_store.grant_lifetime(user_id)
        await callback.answer("Lifetime выдан.")
        if updated:
            try:
                await bot.send_message(
                    user_id,
                    "✅ <b>Lifetime access has been activated.</b>\n\n"
                    "You now have permanent access to all trading signals.",
                    parse_mode=ParseMode.HTML,
                    reply_markup=miniapp_keyboard(settings),
                )
            except (TelegramForbiddenError, TelegramBadRequest):
                pass
            await callback.message.edit_text(
                admin_user_text(updated, settings.referral_default_percent),
                parse_mode=ParseMode.HTML,
                reply_markup=admin_user_keyboard(updated),
            )
        return

    await callback.answer()


@router.callback_query(F.data.startswith("adm:rp:"))
async def admin_referral_percent_action(
    callback: CallbackQuery,
    settings: Settings,
    access_store: AccessStore,
    state: FSMContext,
) -> None:
    if not _is_admin(callback.from_user.id, settings):
        await callback.answer("Недоступно.", show_alert=True)
        return
    parts = callback.data.split(":")
    if len(parts) != 4:
        await callback.answer("Некорректная команда.", show_alert=True)
        return
    try:
        user_id = int(parts[2])
    except ValueError:
        await callback.answer("Некорректный ID.", show_alert=True)
        return
    action = parts[3]
    if not access_store.get_user(user_id):
        await callback.answer("Пользователь не найден.", show_alert=True)
        return

    if action == "custom":
        await callback.answer()
        await state.set_state(AdminStates.waiting_referral_percent)
        await state.update_data(referral_user_id=user_id)
        if callback.message:
            await callback.message.edit_text(
                "✏️ <b>СВОЙ РЕФЕРАЛЬНЫЙ %</b>\n\n"
                "Отправь число от 0 до 100. Например: <code>35</code>",
                parse_mode=ParseMode.HTML,
                reply_markup=InlineKeyboardMarkup(
                    inline_keyboard=[
                        [InlineKeyboardButton(text="⬅️ Назад", callback_data=f"adm:u:{user_id}:refpct")]
                    ]
                ),
            )
        return

    if action == "reset":
        updated = access_store.set_referral_percent_override(user_id, None)
        await callback.answer("Возвращено значение по умолчанию.")
    else:
        try:
            percent = float(action)
        except ValueError:
            await callback.answer("Некорректный процент.", show_alert=True)
            return
        updated = access_store.set_referral_percent_override(user_id, percent)
        await callback.answer(f"Установлено {percent:g}%.")

    await state.clear()
    if callback.message and updated:
        await callback.message.edit_text(
            admin_user_text(updated, settings.referral_default_percent),
            parse_mode=ParseMode.HTML,
            reply_markup=admin_user_keyboard(updated),
        )


@router.message(AdminStates.waiting_referral_percent, F.text)
async def admin_referral_percent_custom(
    message: Message,
    settings: Settings,
    access_store: AccessStore,
    state: FSMContext,
) -> None:
    if not message.from_user or not _is_admin(message.from_user.id, settings):
        return
    data = await state.get_data()
    user_id = int(data.get("referral_user_id") or 0)
    try:
        percent = float((message.text or "").replace(",", ".").strip())
    except ValueError:
        await message.answer("Отправь число от 0 до 100.")
        return
    if not (0 <= percent <= 100):
        await message.answer("Процент должен быть от 0 до 100.")
        return
    updated = access_store.set_referral_percent_override(user_id, percent)
    await state.clear()
    if not updated:
        await message.answer("Пользователь не найден.", reply_markup=admin_back_keyboard())
        return
    await message.answer(
        admin_user_text(updated, settings.referral_default_percent),
        parse_mode=ParseMode.HTML,
        reply_markup=admin_user_keyboard(updated),
    )


@router.callback_query(F.data == "adm:payments")
async def admin_pending_payments(
    callback: CallbackQuery,
    settings: Settings,
    access_store: AccessStore,
) -> None:
    if not _is_admin(callback.from_user.id, settings):
        await callback.answer("Недоступно.", show_alert=True)
        return
    await callback.answer()
    if not callback.message:
        return
    rows = access_store.pending_payments(20)
    buttons: list[list[InlineKeyboardButton]] = []
    for row in rows:
        username = f"@{row['username']}" if row["username"] else str(row["telegram_id"])
        plan = "30 дней" if row["plan_key"] == "monthly" else "Lifetime"
        buttons.append(
            [
                InlineKeyboardButton(
                    text=f"🧾 #{row['id']} · {username[:18]} · {plan}",
                    callback_data=f"adm:pay:{row['id']}",
                )
            ]
        )
    buttons.append(
        [InlineKeyboardButton(text="⬅️ Админ-панель", callback_data="adm:panel")]
    )
    text = "💳 <b>ОЖИДАЮТ ПРОВЕРКИ</b>\n\n"
    text += f"Найдено: <b>{len(rows)}</b>" if rows else "Новых чеков нет."
    await callback.message.edit_text(
        text,
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
    )


@router.callback_query(F.data.startswith("adm:pay:"))
async def admin_payment_detail(
    callback: CallbackQuery,
    settings: Settings,
    access_store: AccessStore,
) -> None:
    if not _is_admin(callback.from_user.id, settings):
        await callback.answer("Недоступно.", show_alert=True)
        return
    await callback.answer()
    if not callback.message:
        return
    try:
        payment_id = int(callback.data.rsplit(":", 1)[1])
    except ValueError:
        return
    p = access_store.get_payment(payment_id)
    if not p:
        await callback.message.edit_text(
            "Платёж не найден.", reply_markup=admin_back_keyboard()
        )
        return
    username = f"@{escape(p['username'])}" if p["username"] else "—"
    plan = "30 Days" if p["plan_key"] == "monthly" else "Lifetime"
    method = "USDT TRC20" if p["currency"] == "usdt" else "BTC"
    await callback.message.edit_text(
        "🧾 <b>ПЛАТЁЖ НА ПРОВЕРКЕ</b>\n\n"
        f"ID: <code>{p['id']}</code>\n"
        f"Пользователь: {username}\n"
        f"Telegram ID: <code>{p['telegram_id']}</code>\n"
        f"Тариф: <b>{plan}</b>\n"
        f"Метод: <b>{method}</b>\n"
        f"Сумма: <b>{escape(p['amount_text'])}</b>\n\n"
        "Сам чек был переслан админу в момент отправки пользователем.",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="✅ Подтвердить",
                        callback_data=f"admin:approve:{payment_id}",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text="❌ Отклонить",
                        callback_data=f"admin:reject:{payment_id}",
                    )
                ],
                [
                    InlineKeyboardButton(
                        text="⬅️ Список платежей", callback_data="adm:payments"
                    )
                ],
            ]
        ),
    )


@router.callback_query(F.data == "adm:broadcast")
async def admin_broadcast_menu(
    callback: CallbackQuery,
    settings: Settings,
    state: FSMContext,
) -> None:
    if not _is_admin(callback.from_user.id, settings):
        await callback.answer("Недоступно.", show_alert=True)
        return
    await callback.answer()
    await state.clear()
    if callback.message:
        await callback.message.edit_text(
            "📢 <b>РАССЫЛКА</b>\n\nВыбери, кому отправить сообщение:",
            parse_mode=ParseMode.HTML,
            reply_markup=admin_broadcast_segments_keyboard(),
        )


@router.callback_query(F.data.startswith("adm:bc:"))
async def admin_broadcast_segment(
    callback: CallbackQuery,
    settings: Settings,
    access_store: AccessStore,
    state: FSMContext,
) -> None:
    if not _is_admin(callback.from_user.id, settings):
        await callback.answer("Недоступно.", show_alert=True)
        return
    segment = callback.data.split(":", 2)[2]
    if segment not in {"all", "active", "monthly", "lifetime"}:
        await callback.answer()
        return
    recipients = [
        uid
        for uid in access_store.broadcast_user_ids(segment)
        if uid != settings.admin_telegram_id
    ]
    await state.set_state(AdminStates.waiting_broadcast)
    await state.update_data(broadcast_segment=segment)
    await callback.answer()
    if callback.message:
        await callback.message.edit_text(
            "📢 <b>ПОДГОТОВКА РАССЫЛКИ</b>\n\n"
            f"Получателей сейчас: <b>{len(recipients)}</b>\n\n"
            "Отправь следующим сообщением текст, фото или видео, которое нужно разослать.",
            parse_mode=ParseMode.HTML,
            reply_markup=admin_back_keyboard(),
        )


@router.message(AdminStates.waiting_broadcast)
async def admin_broadcast_draft(
    message: Message,
    settings: Settings,
    access_store: AccessStore,
    state: FSMContext,
) -> None:
    if not message.from_user or not _is_admin(message.from_user.id, settings):
        return
    if not (message.text or message.photo or message.video):
        await message.answer(
            "Поддерживаются только текст, фото или видео. Отправь сообщение ещё раз."
        )
        return
    data = await state.get_data()
    segment = data.get("broadcast_segment", "all")
    count = len(
        [
            uid
            for uid in access_store.broadcast_user_ids(segment)
            if uid != settings.admin_telegram_id
        ]
    )
    await state.update_data(
        draft_chat_id=message.chat.id,
        draft_message_id=message.message_id,
    )
    await state.set_state(AdminStates.broadcast_confirm)
    segment_name = ADMIN_FILTER_NAMES.get(segment, segment)
    await message.answer(
        "📢 <b>ПОДТВЕРЖДЕНИЕ РАССЫЛКИ</b>\n\n"
        f"Аудитория: <b>{escape(segment_name)}</b>\n"
        f"Получателей: <b>{count}</b>\n\n"
        "Отправить это сообщение?",
        parse_mode=ParseMode.HTML,
        reply_markup=InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="✅ Отправить", callback_data="adm:bc_send")],
                [InlineKeyboardButton(text="❌ Отмена", callback_data="adm:broadcast")],
            ]
        ),
    )


async def _broadcast_copy(
    bot: Bot,
    user_id: int,
    source_chat_id: int,
    source_message_id: int,
) -> bool:
    for attempt in range(3):
        try:
            await bot.copy_message(
                chat_id=user_id,
                from_chat_id=source_chat_id,
                message_id=source_message_id,
            )
            return True
        except TelegramRetryAfter as exc:
            await asyncio.sleep(float(exc.retry_after) + 0.5)
        except (TelegramForbiddenError, TelegramBadRequest):
            return False
        except Exception:
            logger.exception("Broadcast failed for telegram_id=%s", user_id)
            if attempt < 2:
                await asyncio.sleep(1.0)
    return False


@router.callback_query(F.data == "adm:bc_send")
async def admin_broadcast_send(
    callback: CallbackQuery,
    settings: Settings,
    access_store: AccessStore,
    bot: Bot,
    state: FSMContext,
) -> None:
    if not _is_admin(callback.from_user.id, settings):
        await callback.answer("Недоступно.", show_alert=True)
        return
    data = await state.get_data()
    if not data.get("draft_chat_id") or not data.get("draft_message_id"):
        await callback.answer(
            "Черновик не найден. Создай рассылку заново.", show_alert=True
        )
        await state.clear()
        return
    segment = data.get("broadcast_segment", "all")
    source_chat_id = int(data["draft_chat_id"])
    source_message_id = int(data["draft_message_id"])
    recipients = [
        uid
        for uid in access_store.broadcast_user_ids(segment)
        if uid != settings.admin_telegram_id
    ]
    await state.clear()
    await callback.answer("Рассылка запущена.")
    if callback.message:
        await callback.message.edit_text(
            "📤 <b>Рассылка запущена</b>\n\n"
            f"Получателей: <b>{len(recipients)}</b>\n"
            "Бот продолжает работать во время отправки.",
            parse_mode=ParseMode.HTML,
        )

    sent = 0
    failed = 0
    for user_id in recipients:
        ok = await _broadcast_copy(
            bot, user_id, source_chat_id, source_message_id
        )
        if ok:
            sent += 1
        else:
            failed += 1
        # Conservative pacing (~18 messages/sec) to reduce flood-control hits.
        await asyncio.sleep(0.055)

    await bot.send_message(
        settings.admin_telegram_id,
        "✅ <b>РАССЫЛКА ЗАВЕРШЕНА</b>\n\n"
        f"Доставлено: <b>{sent}</b>\n"
        f"Не доставлено: <b>{failed}</b>\n"
        f"Всего обработано: <b>{sent + failed}</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=admin_panel_keyboard(),
    )


@router.message(F.photo | F.document)
async def customer_file_in_chat(
    message: Message,
    settings: Settings,
) -> None:
    """Receipts must be uploaded through authenticated Mini App checkout."""
    if not message.from_user or _is_admin(message.from_user.id, settings):
        return
    await message.answer(
        "📎 Payment receipts are accepted only inside the MENTAL TRADER Mini App.",
        reply_markup=miniapp_keyboard(settings),
    )


async def _safe_callback_answer(
    callback: CallbackQuery,
    text: str | None = None,
    *,
    show_alert: bool = False,
) -> None:
    """
    Telegram callback queries expire quickly. A stale callback must never stop
    payment approval/rejection from being completed.
    """
    try:
        await callback.answer(text=text, show_alert=show_alert)
    except TelegramBadRequest as exc:
        logger.warning("Callback query could not be answered (probably expired): %s", exc)


@router.callback_query(F.data.startswith("admin:payout:"))
async def admin_payout_review(
    callback: CallbackQuery,
    settings: Settings,
    access_store: AccessStore,
) -> None:
    if callback.from_user.id != settings.admin_telegram_id:
        await _safe_callback_answer(callback, "Только для администратора.", show_alert=True)
        return
    parts = callback.data.split(":")
    if len(parts) != 4 or parts[2] not in {"paid", "rejected"}:
        await _safe_callback_answer(callback, "Некорректная команда.", show_alert=True)
        return
    try:
        payout_id = int(parts[3])
    except ValueError:
        await _safe_callback_answer(callback, "Некорректный ID.", show_alert=True)
        return
    status = parts[2]
    row = access_store.review_payout(payout_id, status, callback.from_user.id)
    if not row:
        await _safe_callback_answer(callback, "Запрос уже обработан или не найден.", show_alert=True)
        return
    label = "✅ Выплата отмечена как выполненная." if status == "paid" else "❌ Выплата отклонена."
    await _safe_callback_answer(callback, label)
    if callback.message:
        try:
            base = callback.message.html_text or escape(callback.message.text or "")
            await callback.message.edit_text(
                base + f"\n\n<b>{escape(label)}</b>",
                parse_mode=ParseMode.HTML,
                reply_markup=None,
            )
        except Exception:
            logger.exception("Could not update referral payout admin message")


@router.callback_query(F.data.startswith("admin:approve:"))
async def admin_approve(
    callback: CallbackQuery,
    settings: Settings,
    access_store: AccessStore,
    bot: Bot,
) -> None:
    if callback.from_user.id != settings.admin_telegram_id:
        await _safe_callback_answer(callback, "Только для администратора.", show_alert=True)
        return

    try:
        payment_id = int(callback.data.rsplit(":", 1)[1])
    except (TypeError, ValueError, AttributeError):
        await _safe_callback_answer(callback, "Некорректный ID платежа.", show_alert=True)
        return

    try:
        payment = access_store.approve(payment_id, callback.from_user.id)
    except Exception:
        logger.exception("Admin approve failed for payment %s", payment_id)
        await _safe_callback_answer(callback, "Ошибка при подтверждении платежа.", show_alert=True)
        return

    if payment is None:
        await _safe_callback_answer(callback, "Платёж уже обработан или не найден.", show_alert=True)
        return

    # Give immediate visible feedback in Telegram before any notification/edit.
    await _safe_callback_answer(callback, "✅ Платёж подтверждён.")

    # Referral commission is calculated once, on the referred user's first
    # approved purchase. The referrer must have an active subscription at this
    # exact moment; otherwise this first-purchase commission is missed.
    try:
        access_store.create_referral_reward_for_payment(
            payment_id,
            settings.referral_default_percent,
            settings.monthly_usd,
            settings.lifetime_usd,
        )
    except Exception:
        logger.exception("Could not create referral reward for payment %s", payment_id)

    info = access_store.access_info(int(payment["telegram_id"]))
    if payment["plan_key"] == "lifetime":
        user_text = (
            "✅ <b>Payment approved!</b>\n\n"
            "Your <b>Lifetime</b> access has been activated.\n"
            "Open the Mini App and tap <b>Check Status</b>."
        )
        expiry_text = "Lifetime"
    else:
        expiry = info.expires_at
        expiry_text = expiry.strftime("%Y-%m-%d %H:%M UTC") if expiry else "30 days"
        user_text = (
            "✅ <b>Payment approved!</b>\n\n"
            "Your <b>30-day</b> access has been activated.\n"
            f"Expires: <b>{expiry_text}</b>\n\n"
            "Open the Mini App and tap <b>Check Status</b>."
        )

    try:
        await bot.send_message(
            int(payment["telegram_id"]),
            user_text,
            parse_mode=ParseMode.HTML,
            reply_markup=miniapp_keyboard(settings),
        )
    except Exception:
        logger.exception("Could not notify approved user")

    if callback.message:
        try:
            base_text = callback.message.html_text or escape(callback.message.text or "")
            await callback.message.edit_text(
                base_text
                + f"\n\n✅ <b>ПОДТВЕРЖДЕНО</b> администратором "
                  f"<code>{callback.from_user.id}</code>\n"
                  f"Доступ: <b>{escape(expiry_text)}</b>",
                parse_mode=ParseMode.HTML,
                reply_markup=None,
            )
        except Exception:
            logger.exception("Could not update admin approval message for payment %s", payment_id)


@router.callback_query(F.data.startswith("admin:reject:"))
async def admin_reject(
    callback: CallbackQuery,
    settings: Settings,
    access_store: AccessStore,
    bot: Bot,
) -> None:
    if callback.from_user.id != settings.admin_telegram_id:
        await _safe_callback_answer(callback, "Только для администратора.", show_alert=True)
        return

    try:
        payment_id = int(callback.data.rsplit(":", 1)[1])
    except (TypeError, ValueError, AttributeError):
        await _safe_callback_answer(callback, "Некорректный ID платежа.", show_alert=True)
        return

    try:
        payment = access_store.reject(payment_id, callback.from_user.id)
    except Exception:
        logger.exception("Admin reject failed for payment %s", payment_id)
        await _safe_callback_answer(callback, "Ошибка при отклонении платежа.", show_alert=True)
        return

    if payment is None:
        await _safe_callback_answer(callback, "Платёж уже обработан или не найден.", show_alert=True)
        return

    await _safe_callback_answer(callback, "❌ Платёж отклонён.")

    try:
        await bot.send_message(
            int(payment["telegram_id"]),
            "❌ <b>Payment could not be verified.</b>\n\n"
            "Please check your transaction details in the Mini App and submit a new receipt if needed.",
            parse_mode=ParseMode.HTML,
            reply_markup=miniapp_keyboard(settings),
        )
    except Exception:
        logger.exception("Could not notify rejected user")

    if callback.message:
        try:
            base_text = callback.message.html_text or escape(callback.message.text or "")
            await callback.message.edit_text(
                base_text
                + f"\n\n❌ <b>ОТКЛОНЕНО</b> администратором "
                  f"<code>{callback.from_user.id}</code>",
                parse_mode=ParseMode.HTML,
                reply_markup=None,
            )
        except Exception:
            logger.exception("Could not update admin rejection message for payment %s", payment_id)
