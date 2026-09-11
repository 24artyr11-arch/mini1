# MENTAL TRADER backend v3.2

FastAPI + aiogram backend for the MENTAL TRADER Telegram Mini App.

## Architecture

Customer actions are **Mini App only**:

`Mini App -> FastAPI -> SQLite / market providers -> Telegram admin review`

The Telegram chat is no longer a customer UI. `/start` only shows the **Open MENTAL TRADER** Web App button. Legacy customer buttons are neutralized and redirect users to the Mini App. The Russian `/admin` panel remains in Telegram chat.

## Mini App customer flow

- View subscription status
- Choose Monthly ($50) or Lifetime ($180)
- Choose USDT TRC20 or BTC
- Receive amount and wallet address from the backend
- Upload receipt directly from Mini App (image/PDF, max 8 MB)
- See pending/rejected/approved state inside Mini App
- Browse Forex, Metals, Crypto and Nasdaq
- View M15 BUY / SELL / WAIT signals and refresh price/signal
- View profile/subscription information

Payment amounts and wallet addresses are server-side. The browser never decides the amount and never receives `BOT_TOKEN`, provider keys or `BACKEND_API_KEY`.

## Telegram admin flow

`/admin` remains Russian-only and includes users, search, manual Monthly/Lifetime grants, cancellation, pending payments, approval/rejection, statistics and broadcasts.

When a receipt is uploaded in Mini App, the backend sends the actual file to `ADMIN_TELEGRAM_ID`, then sends the admin review card with **Подтвердить / Отклонить** buttons.

## Required environment variables

```env
BOT_TOKEN=...
ADMIN_TELEGRAM_ID=...
MINIAPP_URL=https://mini-app.24artyr11.workers.dev
MINIAPP_ORIGIN=https://mini-app.24artyr11.workers.dev
MINIAPP_AUTH_MAX_AGE_SECONDS=86400
BACKEND_API_KEY=long-random-server-secret

FCS_API_KEY=...
FCS_BASE_URL=https://api-v4.fcsapi.com
TWELVE_DATA_API_KEY=...

API_HOST=0.0.0.0
API_PORT=3000
DATABASE_PATH=bot.db

PRICE_CACHE_SECONDS=15
FCS_PRICE_CACHE_SECONDS=60
CANDLE_CACHE_GRACE_SECONDS=4
TARGET_SIGNAL_RATE=0.70
ADAPTIVE_LOOKBACK=96
MIN_SCORE=40
STRONG_SCORE=80
BARS_COUNT=350
REQUEST_TIMEOUT_SECONDS=15

USDT_TRC20_ADDRESS=...
BTC_ADDRESS=...
MONTHLY_USD=50
LIFETIME_USD=180
BTC_USD_RATE=77000
```

If your host injects `PORT`, it overrides `API_PORT` automatically.

## Bothost

Use:

```text
Main file: main.py
Web application port: 3000
```

The root `main.py` exposes one FastAPI/Uvicorn server. On Bothost, Telegram admin updates use a webhook built automatically from the platform `DOMAIN` variable. This avoids `getUpdates` conflicts and makes admin approval/rejection buttons reliable. Local development falls back to polling when no public domain is available.

Test after deployment:

```text
https://YOUR-BACKEND-DOMAIN/health
```

Expected `version` is `3.2.0`.

## Mini App payment API

- `GET /api/payment/options`
- `POST /api/payment/intent`
- `POST /api/payment/receipt`
- `GET /api/payment/status`

All four authenticate the user with Telegram `initData` in `X-Telegram-Init-Data`.

## Other Mini App API

- `GET /api/me`
- `GET /api/markets`
- `GET /api/instruments/{market}`
- `GET /api/price/{market}/{instrument_id}`
- `GET /api/signal/{market}/{instrument_id}`

Signal and price endpoints require an active subscription (administrator is exempt).

## Security notes

Do not commit real secrets to GitHub. `.env.example` is only a template. Never store wallet private keys or seed phrases on the server; only public receiving addresses belong in the environment variables.


## v3.1 receipt upload fix
Receipt uploads now include plan/currency and the API reconstructs a lost temporary payment intent after a host restart. Amounts are always recalculated server-side.

## v3.2 admin callback reliability

- Production uses Telegram webhooks instead of long polling when `DOMAIN` is available.
- `/admin`, Approve and Reject callback queries are delivered through `/api/telegram/webhook`.
- Approve/Reject now show an immediate Telegram confirmation toast and remove the review buttons after processing.
- `TELEGRAM_WEBHOOK_URL` and `TELEGRAM_WEBHOOK_SECRET` are optional overrides; Bothost normally needs neither.
