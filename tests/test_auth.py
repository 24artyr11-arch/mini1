import hashlib
import hmac
import json
from urllib.parse import urlencode

from app.auth import MiniAppAuthError, validate_telegram_init_data


def sign(params: dict[str, str], token: str, include_signature: bool = False) -> str:
    working = dict(params)
    if not include_signature:
        working.pop("signature", None)
    data_check = "\n".join(f"{k}={working[k]}" for k in sorted(working))
    secret = hmac.new(b"WebAppData", token.encode(), hashlib.sha256).digest()
    return hmac.new(secret, data_check.encode(), hashlib.sha256).hexdigest()


def test_official_telegram_example_validates():
    token = "5768337691:AAH5YkoiEuPk8-FZa32hStHTqXiLPtAEhx8"
    init_data = (
        "query_id=AAHdF6IQAAAAAN0XohDhrOrc&"
        "user=%7B%22id%22%3A279058397%2C%22first_name%22%3A%22Vladislav%22%2C%22last_name%22%3A%22Kibenko%22%2C%22username%22%3A%22vdkfrost%22%2C%22language_code%22%3A%22ru%22%2C%22is_premium%22%3Atrue%7D&"
        "auth_date=1662771648&"
        "hash=c501b71e775f74ce10e377dea85a7ea24ecd640b223ea86dfe453e0eaed2e2b2"
    )
    user = validate_telegram_init_data(
        init_data, token, max_age_seconds=0, now=1662771648
    )
    assert user.telegram_id == 279058397
    assert user.username == "vdkfrost"


def test_modern_signature_field_is_supported():
    token = "123456:TEST_TOKEN"
    params = {
        "auth_date": "2000000000",
        "query_id": "abc",
        "signature": "telegram-ed25519-placeholder",
        "user": json.dumps({"id": 42, "first_name": "Test"}, separators=(",", ":")),
    }
    params["hash"] = sign(params, token, include_signature=False)
    user = validate_telegram_init_data(
        urlencode(params), token, max_age_seconds=60, now=2000000000
    )
    assert user.telegram_id == 42


def test_tampering_is_rejected():
    token = "123456:TEST_TOKEN"
    params = {
        "auth_date": "2000000000",
        "user": json.dumps({"id": 42, "first_name": "Test"}, separators=(",", ":")),
    }
    params["hash"] = sign(params, token)
    init_data = urlencode(params).replace("%2242%22", "%2299%22")
    # If the replace did not hit because id is numeric, tamper the encoded first name.
    init_data = init_data.replace("Test", "Mallory")
    try:
        validate_telegram_init_data(init_data, token, max_age_seconds=60, now=2000000000)
    except MiniAppAuthError:
        return
    raise AssertionError("Tampered initData must be rejected")
