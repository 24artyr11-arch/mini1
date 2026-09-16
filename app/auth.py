from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from urllib.parse import parse_qsl


class MiniAppAuthError(ValueError):
    """Raised when Telegram Mini App initData cannot be trusted."""


@dataclass(frozen=True, slots=True)
class MiniAppUser:
    telegram_id: int
    first_name: str
    last_name: str | None
    username: str | None
    language_code: str | None
    is_premium: bool
    auth_date: int


def _parse_unique_pairs(init_data: str) -> dict[str, str]:
    pairs = parse_qsl(init_data, keep_blank_values=True, strict_parsing=False)
    if not pairs:
        raise MiniAppAuthError("Telegram initData is empty.")
    result: dict[str, str] = {}
    for key, value in pairs:
        if key in result:
            raise MiniAppAuthError("Telegram initData contains duplicate fields.")
        result[key] = value
    return result


def _calculate_hash(bot_token: str, params: dict[str, str], *, include_signature: bool) -> str:
    filtered = {
        key: value
        for key, value in params.items()
        if key != "hash" and (include_signature or key != "signature")
    }
    data_check_string = "\n".join(f"{key}={filtered[key]}" for key in sorted(filtered))
    secret_key = hmac.new(
        key=b"WebAppData",
        msg=bot_token.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    return hmac.new(
        key=secret_key,
        msg=data_check_string.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).hexdigest()


def validate_telegram_init_data(
    init_data: str,
    bot_token: str,
    max_age_seconds: int = 86_400,
    now: int | None = None,
) -> MiniAppUser:
    """Validate Telegram.WebApp.initData and return the authenticated user.

    The HMAC follows Telegram's WebAppData algorithm. Newer Telegram clients may
    include the separate Ed25519 `signature` field. We validate the documented
    modern form (excluding `hash` and `signature`) first, and retain a legacy
    compatibility check that excludes only `hash`. Both paths still require a
    valid HMAC derived from the private bot token.
    """
    if not init_data or len(init_data) > 16_384:
        raise MiniAppAuthError("Telegram initData is missing or too large.")
    if not bot_token:
        raise MiniAppAuthError("Bot token is not configured.")

    params = _parse_unique_pairs(init_data)
    received_hash = params.get("hash")
    if not received_hash:
        raise MiniAppAuthError("Telegram initData hash is missing.")

    modern_hash = _calculate_hash(bot_token, params, include_signature=False)
    valid = hmac.compare_digest(received_hash, modern_hash)

    # Compatibility with older client/library behavior when `signature` exists.
    if not valid and "signature" in params:
        legacy_hash = _calculate_hash(bot_token, params, include_signature=True)
        valid = hmac.compare_digest(received_hash, legacy_hash)

    if not valid:
        raise MiniAppAuthError("Telegram initData signature is invalid.")

    try:
        auth_date = int(params["auth_date"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MiniAppAuthError("Telegram initData auth_date is invalid.") from exc

    current = int(time.time()) if now is None else int(now)
    if auth_date > current + 60:
        raise MiniAppAuthError("Telegram initData auth_date is in the future.")
    if max_age_seconds > 0 and current - auth_date > max_age_seconds:
        raise MiniAppAuthError("Telegram initData has expired. Reopen the Mini App.")

    raw_user = params.get("user")
    if not raw_user:
        raise MiniAppAuthError("Telegram initData does not contain a user.")
    try:
        user = json.loads(raw_user)
    except json.JSONDecodeError as exc:
        raise MiniAppAuthError("Telegram user data is invalid JSON.") from exc
    if not isinstance(user, dict):
        raise MiniAppAuthError("Telegram user data is invalid.")

    try:
        telegram_id = int(user["id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MiniAppAuthError("Telegram user id is invalid.") from exc
    if telegram_id <= 0 or user.get("is_bot") is True:
        raise MiniAppAuthError("Telegram user is invalid.")

    first_name = str(user.get("first_name") or "Telegram User")[:128]
    last_name = str(user["last_name"])[:128] if user.get("last_name") else None
    username = str(user["username"])[:64] if user.get("username") else None
    language_code = str(user["language_code"])[:16] if user.get("language_code") else None

    return MiniAppUser(
        telegram_id=telegram_id,
        first_name=first_name,
        last_name=last_name,
        username=username,
        language_code=language_code,
        is_premium=bool(user.get("is_premium", False)),
        auth_date=auth_date,
    )
