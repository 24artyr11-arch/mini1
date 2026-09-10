from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]


def test_python_files_compile():
    for path in [
        ROOT / "main.py",
        ROOT / "run.py",
        ROOT / "app" / "core.py",
        ROOT / "app" / "auth.py",
        ROOT / "app" / "api.py",
        ROOT / "app" / "main.py",
    ]:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_four_markets_are_present():
    text = (ROOT / "app" / "core.py").read_text(encoding="utf-8")
    for name in [
        'FOREX = "forex"',
        'METALS = "metals"',
        'CRYPTO = "crypto"',
        'NASDAQ = "nasdaq"',
    ]:
        assert name in text


def test_provider_routing_and_cache_present():
    text = (ROOT / "app" / "core.py").read_text(encoding="utf-8")
    assert "class FcsClient" in text
    assert "class TwelveDataClient" in text
    assert "class SharedMarketCache" in text
    assert "Market.FOREX, Market.METALS" in text
    assert "seconds_until_next_m15_close" in text


def test_miniapp_auth_and_endpoints_present():
    api = (ROOT / "app" / "api.py").read_text(encoding="utf-8")
    auth = (ROOT / "app" / "auth.py").read_text(encoding="utf-8")
    for endpoint in [
        '/health',
        '/api/me',
        '/api/markets',
        '/api/instruments/{market_name}',
        '/api/price/{market_name}/{instrument_id}',
        '/api/signal/{market_name}/{instrument_id}',
        '/api/payment/options',
        '/api/payment/intent',
        '/api/payment/receipt',
        '/api/payment/status',
        '/api/admin/fcs/availability',
        '/api/admin/cache',
    ]:
        assert endpoint in api
    assert "X-Telegram-Init-Data" in api
    assert 'key=b"WebAppData"' in auth
    assert 'params.get("hash")' in auth
    assert 'params.get("user")' in auth
    assert 'params["auth_date"]' in auth


def test_customer_chat_is_miniapp_only_and_admin_remains():
    core = (ROOT / "app" / "core.py").read_text(encoding="utf-8")
    assert "All customer features" in core
    assert "Payment receipts are accepted only inside" in core
    assert 'Command("admin")' in core
    assert "admin:approve:" in core
    assert "admin:reject:" in core
    # Old chat execution handlers are intentionally removed.
    assert 'async def asset_selected(' not in core
    assert 'async def recalculate_signal(' not in core
    assert 'async def refresh_price(' not in core


def test_miniapp_menu_and_lifespan_startup():
    main = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
    core = (ROOT / "app" / "core.py").read_text(encoding="utf-8")
    assert "MenuButtonWebApp" in main
    assert "Open Trader" in main
    assert "@asynccontextmanager" in main
    assert "lifespan=lifespan" in main
    assert "MINIAPP_URL" in core


def test_receipt_upload_dependency_is_declared():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "python-multipart" in requirements
