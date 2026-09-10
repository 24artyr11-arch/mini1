# MENTAL TRADER Backend v2

Telegram bot + FastAPI backend for the MENTAL TRADER Telegram Mini App.

## Architecture

- Telegram bot: payments, receipts, subscription approval, Russian `/admin` panel, broadcasts
- FastAPI: secure Mini App API
- OANDA: Forex + Metals
- Twelve Data: Crypto + Nasdaq
- Shared in-process cache: price, M15 candles and signal results
- SQLite: users, subscriptions and payment review

## Mini App security

The Mini App never sends a Telegram ID supplied by JavaScript as proof of identity. It sends the raw `Telegram.WebApp.initData` string in the `X-Telegram-Init-Data` header.

The backend:

1. verifies the Telegram HMAC using `BOT_TOKEN` and `WebAppData`;
2. validates `auth_date` freshness;
3. extracts the signed Telegram user;
4. checks that user's subscription in SQLite;
5. only then returns protected price/signal data.

Do **not** put `BOT_TOKEN`, `BACKEND_API_KEY`, OANDA/Twelve Data keys, wallet private keys or seed phrases in the Mini App frontend.

## Environment variables

Copy `.env.example` to `.env` for local/self-hosted use, or create the same variables in your hosting dashboard.

Required:

```env
BOT_TOKEN=
ADMIN_TELEGRAM_ID=

OANDA_API_TOKEN=
OANDA_ACCOUNT_ID=
OANDA_ENV=practice

TWELVE_DATA_API_KEY=

BACKEND_API_KEY=choose-a-long-random-server-side-secret
```

After the Mini App is deployed, add:

```env
MINIAPP_URL=https://your-miniapp.pages.dev
MINIAPP_ORIGIN=https://your-miniapp.pages.dev
MINIAPP_AUTH_MAX_AGE_SECONDS=86400
```

`MINIAPP_ORIGIN` must be the exact browser origin allowed to call FastAPI. Do not use `*` in production.

## Run

```bash
pip install -r requirements.txt
python run.py
```

The process starts Telegram polling and FastAPI together.

Check:

```text
https://YOUR-BACKEND/health
https://YOUR-BACKEND/docs
```

The backend itself must also be reachable via public HTTPS for a production Mini App.

## Mini App endpoints

Authenticated with `X-Telegram-Init-Data`:

```text
GET /api/me
GET /api/markets
GET /api/instruments/{market}
GET /api/price/{market}/{instrument_id}
GET /api/signal/{market}/{instrument_id}
```

`/api/price` and `/api/signal` additionally require an active Monthly/Lifetime subscription (admin is allowed automatically).

Server-side admin/testing endpoints use `X-Backend-Key`:

```text
GET /api/admin/oanda/availability
GET /api/admin/cache
```

The normal Mini App frontend must never know `BACKEND_API_KEY`.

## Cache behavior

- current price: `PRICE_CACHE_SECONDS` (default 15 seconds)
- M15 candles: until next M15 close + grace period
- calculated signal: until next M15 close + grace period
- identical simultaneous requests share the same in-flight request

So many users requesting the same instrument do not create one provider request per user.

## Telegram menu button

When `MINIAPP_URL` is configured, the bot attempts to set a global **Open Trader** menu button automatically at startup.

For the prominent **Launch app / Open App** button on the bot profile, also configure the Main Mini App once in `@BotFather` and paste the same HTTPS Mini App URL.

## Deployment order

1. Deploy this backend and confirm `/health` works over HTTPS.
2. Deploy the separate Mini App frontend to Cloudflare Pages.
3. Put the frontend URL into backend `MINIAPP_URL` and `MINIAPP_ORIGIN`, then redeploy/restart backend.
4. Put the backend HTTPS URL and bot username into frontend `config.js`, redeploy frontend.
5. Configure Main Mini App in `@BotFather` with the frontend URL.
6. Open the Mini App from Telegram and test `/api/me`, one OANDA signal and one Twelve Data signal.
