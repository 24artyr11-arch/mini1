from __future__ import annotations

import asyncio
import json
import logging
import os
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

    fcs_api_key: str = ""
    fcs_base_url: str = "https://api-v4.fcsapi.com"

    api_host: str = "0.0.0.0"
    api_port: int = 3000
    backend_api_key: str = ""
    miniapp_url: str = ""
    miniapp_origin: str = ""
    miniapp_auth_max_age_seconds: int = 86400
    price_cache_seconds: int = 15
    fcs_price_cache_seconds: int = 60
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


def get_settings() -> Settings:
    token = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "Telegram token is missing. Set BOT_TOKEN (recommended) or TELEGRAM_BOT_TOKEN."
        )

    api_key = os.getenv("TWELVE_DATA_API_KEY")
    if not api_key:
        raise RuntimeError("Twelve Data API key is missing. Set TWELVE_DATA_API_KEY.")

    fcs_api_key = (os.getenv("FCS_API_KEY") or os.getenv("FCS_ACCESS_KEY") or "").strip()
    if not fcs_api_key:
        raise RuntimeError(
            "FCS API key is missing. Set FCS_API_KEY."
        )

    return Settings(
        telegram_bot_token=token,
        twelve_data_api_key=api_key,
        twelve_data_base_url=os.getenv(
            "TWELVE_DATA_BASE_URL", "https://api.twelvedata.com"
        ).rstrip("/"),
        fcs_api_key=fcs_api_key,
        fcs_base_url=os.getenv(
            "FCS_BASE_URL", "https://api-v4.fcsapi.com"
        ).rstrip("/"),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("PORT", os.getenv("API_PORT", "3000"))),
        backend_api_key=(os.getenv("BACKEND_API_KEY") or "").strip(),
        miniapp_url=(os.getenv("MINIAPP_URL") or "").strip().rstrip("/"),
        miniapp_origin=(os.getenv("MINIAPP_ORIGIN") or "").strip().rstrip("/"),
        miniapp_auth_max_age_seconds=int(os.getenv("MINIAPP_AUTH_MAX_AGE_SECONDS", "86400")),
        price_cache_seconds=int(os.getenv("PRICE_CACHE_SECONDS", "15")),
        fcs_price_cache_seconds=int(os.getenv("FCS_PRICE_CACHE_SECONDS", "60")),
        candle_cache_grace_seconds=int(os.getenv("CANDLE_CACHE_GRACE_SECONDS", "4")),
        bars_count=int(os.getenv("BARS_COUNT", "350")),
        request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "15")),
        signal_target_rate=float(os.getenv("TARGET_SIGNAL_RATE", "0.70")),
        adaptive_lookback=int(os.getenv("ADAPTIVE_LOOKBACK", "96")),
        min_score=int(os.getenv("MIN_SCORE", "40")),
        strong_score=int(os.getenv("STRONG_SCORE", "80")),
        admin_telegram_id=int(os.getenv("ADMIN_TELEGRAM_ID", "0")),
        database_path=os.getenv("DATABASE_PATH", "bot.db"),
        usdt_trc20_address=os.getenv("USDT_TRC20_ADDRESS", "").strip(),
        btc_address=os.getenv("BTC_ADDRESS", "").strip(),
        monthly_usd=float(os.getenv("MONTHLY_USD", "50")),
        lifetime_usd=float(os.getenv("LIFETIME_USD", "180")),
        btc_usd_rate=float(os.getenv("BTC_USD_RATE", "77000")),
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


RATE_LIMIT_MESSAGE = "Please wait a minute, the tokens have run out."


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
# FCS API connector and shared provider errors
# -----------------------------------------------------------------------------

class FcsError(MarketDataError):
    pass


class FcsRateLimitError(FcsError, MarketDataRateLimitError):
    pass


class FcsInstrumentUnavailable(FcsError):
    pass


class FcsClient:
    """FCS API adapter for Forex and Metals.

    The rest of the application receives one normalized OHLC format regardless
    of provider. Direct FCS symbols are preferred. For several metal crosses,
    the adapter can derive the cross from a USD metal quote and a Forex leg when
    the direct commodity symbol is not available.
    """

    METAL_ALIASES: dict[str, tuple[str, ...]] = {
        "xauusd": ("XAUUSD", "GOLD"),
        "xagusd": ("XAGUSD", "SILVER"),
        "xptusd": ("XPTUSD", "PLATINUM"),
        "xpdusd": ("XPDUSD", "PALLADIUM"),
    }

    # Direct symbols are attempted first. If FCS does not expose the direct
    # cross, derive it from synchronized M15 candles.
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
        self.base_url = settings.fcs_base_url.rstrip("/")

    def _request_json(self, endpoint: str, params: dict[str, object]) -> dict:
        query = {k: v for k, v in params.items() if v is not None}
        query["access_key"] = self.settings.fcs_api_key
        url = f"{self.base_url}/{endpoint.lstrip('/')}?{urllib.parse.urlencode(query)}"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "mental-trader-bot/2.1",
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
                raise FcsRateLimitError(RATE_LIMIT_MESSAGE) from exc
            if exc.code in {401, 403}:
                raise FcsError("FCS API authentication failed. Check FCS_API_KEY and plan access.") from exc
            if exc.code in {400, 404}:
                raise FcsInstrumentUnavailable("FCS API did not return this instrument.") from exc
            raise FcsError(f"FCS API HTTP {exc.code}: {body[:220]}") from exc
        except urllib.error.URLError as exc:
            raise FcsError(f"Could not reach FCS API: {exc.reason}") from exc
        except TimeoutError as exc:
            raise FcsError("FCS API request timed out.") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FcsError("FCS API returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise FcsError("Unexpected FCS API response format.")

        status = payload.get("status")
        code = payload.get("code")
        if status is False or (code not in {None, 200, "200"} and not payload.get("response")):
            message = str(payload.get("msg") or payload.get("message") or "Unknown FCS API error")
            if str(code) == "429" or "limit" in message.lower() or "credit" in message.lower():
                raise FcsRateLimitError(RATE_LIMIT_MESSAGE)
            if str(code) in {"401", "403"}:
                raise FcsError(f"FCS API authentication/plan error: {message}")
            raise FcsInstrumentUnavailable(message)
        return payload

    @staticmethod
    def _response_items(payload: dict) -> list[dict]:
        response = payload.get("response")
        if isinstance(response, list):
            return [item for item in response if isinstance(item, dict)]
        if isinstance(response, dict):
            # History is documented both as a timestamp-keyed object and as a
            # list depending on endpoint/format. Normalize both variants.
            if any(key in response for key in ("o", "h", "l", "c", "active")):
                return [response]
            return [item for item in response.values() if isinstance(item, dict)]
        return []

    @staticmethod
    def _active_price(item: dict) -> float:
        active = item.get("active") if isinstance(item.get("active"), dict) else item
        try:
            ask = active.get("a")
            bid = active.get("b")
            if ask is not None and bid is not None:
                price = (float(ask) + float(bid)) / 2.0
            else:
                price = float(active["c"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FcsError("FCS API returned an unusable live price.") from exc
        if price <= 0:
            raise FcsError("FCS API returned a non-positive live price.")
        return price

    def _latest_direct(self, symbol: str, *, commodity: bool) -> float:
        params: dict[str, object] = {
            "symbol": symbol,
            "period": "15m",
            "type": "commodity" if commodity else "forex",
        }
        payload = self._request_json("forex/latest", params)
        items = self._response_items(payload)
        if not items:
            raise FcsInstrumentUnavailable(f"FCS API returned no live data for {symbol}.")
        return self._active_price(items[0])

    def _history_direct(self, symbol: str, count: int, *, commodity: bool) -> pd.DataFrame:
        params: dict[str, object] = {
            "symbol": symbol,
            "period": "15m",
            "length": min(max(count + 2, 222), 10000),
            "is_chart": 0,
        }
        params["type"] = "commodity" if commodity else "forex"
        payload = self._request_json("forex/history", params)
        items = self._response_items(payload)
        rows: list[dict[str, object]] = []
        for item in items:
            try:
                timestamp = item.get("t")
                if timestamp is not None:
                    dt = pd.to_datetime(float(timestamp), unit="s", utc=True)
                else:
                    dt = pd.to_datetime(item["tm"], utc=True)
                rows.append({
                    "time": dt,
                    "open": float(item["o"]),
                    "high": float(item["h"]),
                    "low": float(item["l"]),
                    "close": float(item["c"]),
                    "volume": float(item.get("v", 0) or 0),
                })
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        if not rows:
            raise FcsInstrumentUnavailable(f"FCS API returned no M15 candles for {symbol}.")
        data = pd.DataFrame(rows).sort_values("time").drop_duplicates("time").reset_index(drop=True)

        # Do not analyze a still-forming M15 candle.
        if len(data) > 0:
            last_start = data.iloc[-1]["time"]
            now = pd.Timestamp.now(tz="UTC")
            if last_start + pd.Timedelta(minutes=15) > now:
                data = data.iloc[:-1]
        return data.tail(count).reset_index(drop=True)

    def _metal_aliases(self, instrument: Instrument) -> tuple[str, ...]:
        return self.METAL_ALIASES.get(instrument.id, (instrument.symbol,))

    def _direct_price(self, instrument: Instrument) -> float:
        if instrument.market == Market.FOREX:
            return self._latest_direct(instrument.symbol, commodity=False)
        last_error: Exception | None = None
        for symbol in self._metal_aliases(instrument):
            try:
                return self._latest_direct(symbol, commodity=True)
            except FcsInstrumentUnavailable as exc:
                last_error = exc
        if last_error:
            raise FcsInstrumentUnavailable(str(last_error))
        raise FcsInstrumentUnavailable(f"{instrument.label} is unavailable on FCS API.")

    def _direct_bars(self, instrument: Instrument, count: int) -> pd.DataFrame:
        if instrument.market == Market.FOREX:
            return self._history_direct(instrument.symbol, count, commodity=False)
        last_error: Exception | None = None
        for symbol in self._metal_aliases(instrument):
            try:
                return self._history_direct(symbol, count, commodity=True)
            except FcsInstrumentUnavailable as exc:
                last_error = exc
        if last_error:
            raise FcsInstrumentUnavailable(str(last_error))
        raise FcsInstrumentUnavailable(f"{instrument.label} is unavailable on FCS API.")

    @staticmethod
    def _combine_price(metal_usd: float, fx: float, operation: str) -> float:
        if metal_usd <= 0 or fx <= 0:
            raise FcsError("Cannot derive metal cross from non-positive prices.")
        if operation == "divide":
            return metal_usd / fx
        if operation == "multiply":
            return metal_usd * fx
        raise FcsError("Unknown derived metal operation.")

    @staticmethod
    def _combine_bars(left: pd.DataFrame, right: pd.DataFrame, operation: str) -> pd.DataFrame:
        a = left.rename(columns={c: f"{c}_a" for c in ("open", "high", "low", "close", "volume")})
        b = right.rename(columns={c: f"{c}_b" for c in ("open", "high", "low", "close", "volume")})
        merged = a.merge(b, on="time", how="inner")
        if merged.empty:
            raise FcsError("Unable to align M15 candles for a derived metal cross.")
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
            raise FcsError("Unknown derived metal operation.")
        out["volume"] = merged["volume_a"].fillna(0)
        return out.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    def _derived_components(self, instrument: Instrument) -> tuple[Instrument, str, str] | None:
        config = self.DERIVED_METALS.get(instrument.id)
        if not config:
            return None
        base_id, fx_symbol, operation = config
        base = get_instrument(Market.METALS, base_id)
        if base is None:
            raise FcsError(f"Internal metal mapping is missing: {base_id}")
        return base, fx_symbol, operation

    def get_price(self, instrument: Instrument) -> LivePrice:
        try:
            price = self._direct_price(instrument)
            source = "FCS API"
        except FcsInstrumentUnavailable:
            derived = self._derived_components(instrument)
            if derived is None:
                raise
            base, fx_symbol, operation = derived
            metal_price = self._direct_price(base)
            fx_price = self._latest_direct(fx_symbol, commodity=False)
            price = self._combine_price(metal_price, fx_price, operation)
            source = "FCS API (derived cross)"
        return LivePrice(
            symbol=instrument.symbol,
            price=price,
            digits=instrument.digits,
            fetched_at=datetime.now(timezone.utc),
            data_source=source,
        )

    def get_closed_bars(self, instrument: Instrument, count: int) -> pd.DataFrame:
        try:
            data = self._direct_bars(instrument, count)
        except FcsInstrumentUnavailable:
            derived = self._derived_components(instrument)
            if derived is None:
                raise
            base, fx_symbol, operation = derived
            metal = self._direct_bars(base, count + 10)
            fx = self._history_direct(fx_symbol, count + 10, commodity=False)
            data = self._combine_bars(metal, fx, operation).tail(count).reset_index(drop=True)
        if len(data) < 220:
            raise FcsError(
                f"Only {len(data)} closed M15 candles are available for {instrument.label}; at least 220 are required."
            )
        return data

    def symbol_availability(self) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {"forex": set(), "commodity": set()}
        for kind in ("forex", "commodity"):
            payload = self._request_json("forex/list", {"type": kind, "per_page": 5000})
            for item in self._response_items(payload):
                ticker = str(item.get("ticker") or "")
                profile = item.get("profile") if isinstance(item.get("profile"), dict) else {}
                symbol = str(profile.get("symbol") or ticker.split(":")[-1] or "").upper()
                if symbol:
                    result[kind].add(symbol)
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
            provider_symbol=instrument.symbol,
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
        self.fcs = FcsClient(settings)
        self.twelve = TwelveDataClient(settings)

    def provider_name(self, instrument: Instrument) -> str:
        return "FCS API" if instrument.market in {Market.FOREX, Market.METALS} else "Twelve Data"

    def _client(self, instrument: Instrument):
        if instrument.market in {Market.FOREX, Market.METALS}:
            return self.fcs
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
            float(self.settings.fcs_price_cache_seconds)
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


def admin_panel_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="👥 Пользователи", callback_data="adm:users"),
                InlineKeyboardButton(text="📢 Рассылка", callback_data="adm:broadcast"),
            ],
            [InlineKeyboardButton(text="🔍 Найти пользователя", callback_data="adm:find")],
            [InlineKeyboardButton(text="💳 Ожидают оплаты", callback_data="adm:payments")],
            [InlineKeyboardButton(text="📊 Статистика", callback_data="adm:stats")],
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


def admin_user_text(row: sqlite3.Row) -> str:
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
    return (
        "👤 <b>Пользователь</b>\n\n"
        f"Имя: <b>{first_name}</b>\n"
        f"Username: {username}\n"
        f"Telegram ID: <code>{row['telegram_id']}</code>\n\n"
        f"Статус: <b>{escape(_admin_status_label(row))}</b>\n"
        f"Тариф: <b>{escape(row['plan'] or '—')}</b>\n"
        f"Доступ до: <b>{escape(expires_text)}</b>"
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

# -----------------------------------------------------------------------------
# Telegram handlers
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Access database and payment review
# -----------------------------------------------------------------------------

START_TEXT = """🤖 <b>Meet MENTAL-TRADER BOT — Your M15 Market Signal Engine</b>

Welcome! MENTAL-TRADER analyzes live market data and gives structured M15 trading signals you can use when trading manually in MT5 or another platform.

<b>What you get:</b>
⚡️ <b>Live market data:</b> Forex & Metals via FCS API; Crypto & Nasdaq via Twelve Data.
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
    """Small SQLite store for access, payment intents and admin reviews."""

    def __init__(self, path: str):
        db_path = Path(path)
        if db_path.parent != Path('.'):
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(db_path)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
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
                    last_seen_at TEXT
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
                """
            )
            # Lightweight migration for databases created by earlier bot versions.
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
            if "created_at" not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN created_at TEXT")
            if "last_seen_at" not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN last_seen_at TEXT")
            now = self._now().isoformat()
            conn.execute(
                "UPDATE users SET created_at=COALESCE(created_at, updated_at, ?)",
                (now,),
            )
            conn.execute(
                "UPDATE users SET last_seen_at=COALESCE(last_seen_at, updated_at, created_at, ?)",
                (now,),
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
                SET status='approved', reviewed_at=?, reviewed_by=?
                WHERE id=?
                """,
                (now.isoformat(), admin_id, payment_id),
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


    def touch_user(
        self,
        telegram_id: int,
        username: str | None,
        first_name: str | None,
    ) -> None:
        """Register every /start user so broadcasts can reach non-subscribers too."""
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
                    "SELECT * FROM payments WHERE status='pending' ORDER BY created_at ASC LIMIT ?",
                    (limit,),
                ).fetchall()
            )

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
    await callback.message.edit_text(
        "📊 <b>СТАТИСТИКА</b>\n\n"
        f"👥 Всего пользователей: <b>{s['total']}</b>\n"
        f"✅ Активных: <b>{s['active']}</b>\n"
        f"📅 Monthly: <b>{s['monthly']}</b>\n"
        f"♾ Lifetime: <b>{s['lifetime']}</b>\n"
        f"⚪ Без подписки: <b>{s['inactive']}</b>\n"
        f"⌛ Истёкших: <b>{s['expired']}</b>\n"
        f"🚫 Отменённых: <b>{s['cancelled']}</b>\n\n"
        f"🆕 Новых сегодня (UTC): <b>{s['new_today']}</b>\n"
        f"🗓 Новых в этом месяце (UTC): <b>{s['new_month']}</b>\n\n"
        f"💳 Платежи: подтверждено <b>{s['approved']}</b> · "
        f"отклонено <b>{s['rejected']}</b> · ожидает <b>{s['pending']}</b>",
        parse_mode=ParseMode.HTML,
        reply_markup=admin_back_keyboard(),
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
        admin_user_text(row),
        parse_mode=ParseMode.HTML,
        reply_markup=admin_user_keyboard(row),
    )


@router.callback_query(F.data.startswith("adm:u:"))
async def admin_user_action(
    callback: CallbackQuery,
    settings: Settings,
    access_store: AccessStore,
    bot: Bot,
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
            admin_user_text(row),
            parse_mode=ParseMode.HTML,
            reply_markup=admin_user_keyboard(row),
        )
        return

    if action == "cancelask":
        await callback.answer()
        await callback.message.edit_text(
            admin_user_text(row)
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
                admin_user_text(updated),
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
                admin_user_text(updated),
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
                admin_user_text(updated),
                parse_mode=ParseMode.HTML,
                reply_markup=admin_user_keyboard(updated),
            )
        return

    await callback.answer()


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


@router.callback_query(F.data.startswith("admin:approve:"))
async def admin_approve(
    callback: CallbackQuery,
    settings: Settings,
    access_store: AccessStore,
    bot: Bot,
) -> None:
    if callback.from_user.id != settings.admin_telegram_id:
        await callback.answer("Только для администратора.", show_alert=True)
        return

    try:
        payment_id = int(callback.data.rsplit(":", 1)[1])
    except (TypeError, ValueError, AttributeError):
        await callback.answer("Некорректный ID платежа.", show_alert=True)
        return

    try:
        payment = access_store.approve(payment_id, callback.from_user.id)
    except Exception:
        logger.exception("Admin approve failed for payment %s", payment_id)
        await callback.answer("Ошибка при подтверждении платежа.", show_alert=True)
        return

    if payment is None:
        await callback.answer("Платёж уже обработан или не найден.", show_alert=True)
        return

    # Give immediate visible feedback in Telegram before any notification/edit.
    await callback.answer("✅ Платёж подтверждён.")

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
        await callback.answer("Только для администратора.", show_alert=True)
        return

    try:
        payment_id = int(callback.data.rsplit(":", 1)[1])
    except (TypeError, ValueError, AttributeError):
        await callback.answer("Некорректный ID платежа.", show_alert=True)
        return

    try:
        payment = access_store.reject(payment_id, callback.from_user.id)
    except Exception:
        logger.exception("Admin reject failed for payment %s", payment_id)
        await callback.answer("Ошибка при отклонении платежа.", show_alert=True)
        return

    if payment is None:
        await callback.answer("Платёж уже обработан или не найден.", show_alert=True)
        return

    await callback.answer("❌ Платёж отклонён.")

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


