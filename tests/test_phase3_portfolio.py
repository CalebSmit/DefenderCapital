"""
Phase 3 tests — portfolio_store (save / load / Excel round-trip)
Runs against isolated SQLite — no Neon required.
"""
import os
import sys
import tempfile
from pathlib import Path
import pytest

os.environ.pop("NEON_DATABASE_URL", None)
sys.path.insert(0, str(Path(__file__).parent.parent))

import auth.database as db_mod
from auth.database import init_db
from auth.auth_manager import register_user, login_user
from auth.portfolio_store import (
    save_portfolio,
    load_portfolio,
    has_portfolio,
    holdings_to_excel_bytes,
    excel_path_to_holdings,
)

SAMPLE_HOLDINGS = [
    {"ticker": "AAPL", "shares_held": 10.0, "cost_basis": 150.0},
    {"ticker": "MSFT", "shares_held": 5.0,  "cost_basis": 280.0},
    {"ticker": "NVDA", "shares_held": 3.0,  "cost_basis": 400.0},
]
SAMPLE_SETTINGS = {"covariance_mode": "EWMA", "lookback_years": 3}


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("db") / "port_test.db"
    db_mod._sqlite_path = lambda: tmp
    init_db()
    yield


@pytest.fixture(scope="module")
def user(db):
    register_user("portuser", "port@example.com", "portpass1", "Port User")
    return login_user("portuser", "portpass1")


@pytest.fixture(scope="module")
def user2(db):
    register_user("portuser2", "port2@example.com", "portpass2", "Port User 2")
    return login_user("portuser2", "portpass2")


# ── save & load ───────────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_no_portfolio_initially(self, user):
        assert has_portfolio(user["id"]) is False
        assert load_portfolio(user["id"]) is None

    def test_save_portfolio_success(self, user):
        ok, msg = save_portfolio(
            user["id"], "Test Fund", "TST", SAMPLE_HOLDINGS, SAMPLE_SETTINGS
        )
        assert ok is True
        assert "Test Fund" in msg

    def test_has_portfolio_after_save(self, user):
        assert has_portfolio(user["id"]) is True

    def test_load_returns_correct_name(self, user):
        p = load_portfolio(user["id"])
        assert p is not None
        assert p["portfolio_name"] == "Test Fund"
        assert p["short_name"] == "TST"

    def test_load_returns_correct_holdings(self, user):
        p = load_portfolio(user["id"])
        assert len(p["holdings"]) == 3
        tickers = {h["ticker"] for h in p["holdings"]}
        assert tickers == {"AAPL", "MSFT", "NVDA"}

    def test_load_returns_correct_settings(self, user):
        p = load_portfolio(user["id"])
        assert p["settings"].get("covariance_mode") == "EWMA"

    def test_load_has_updated_at(self, user):
        p = load_portfolio(user["id"])
        assert p["updated_at"] != ""


class TestUpsert:
    def test_save_overwrites_existing(self, user):
        new_holdings = [{"ticker": "GOOG", "shares_held": 2.0, "cost_basis": 130.0}]
        ok, msg = save_portfolio(user["id"], "Updated Fund", "UPD", new_holdings)
        assert ok is True

        p = load_portfolio(user["id"])
        assert p["portfolio_name"] == "Updated Fund"
        assert len(p["holdings"]) == 1
        assert p["holdings"][0]["ticker"] == "GOOG"

    def test_second_save_does_not_duplicate(self, db, user):
        from auth.database import fetchall, ph
        rows = fetchall(
            f"SELECT id FROM dcm_portfolios WHERE user_id = {ph()}", (user["id"],)
        )
        assert len(rows) == 1


class TestIsolation:
    def test_portfolios_isolated_per_user(self, user, user2):
        save_portfolio(user["id"],  "Fund A", "FA", [{"ticker": "AAPL", "shares_held": 1, "cost_basis": 100}])
        save_portfolio(user2["id"], "Fund B", "FB", [{"ticker": "TSLA", "shares_held": 2, "cost_basis": 200}])

        p1 = load_portfolio(user["id"])
        p2 = load_portfolio(user2["id"])

        assert p1["portfolio_name"] == "Fund A"
        assert p2["portfolio_name"] == "Fund B"
        assert p1["holdings"][0]["ticker"] == "AAPL"
        assert p2["holdings"][0]["ticker"] == "TSLA"


class TestValidation:
    def test_empty_name_rejected(self, user):
        ok, msg = save_portfolio(user["id"], "", "XX", SAMPLE_HOLDINGS)
        assert ok is False

    def test_non_list_holdings_rejected(self, user):
        ok, msg = save_portfolio(user["id"], "Fund", "F", {"ticker": "AAPL"})
        assert ok is False


# ── Excel round-trip ─────────────────────────────────────────────────────────

class TestExcelRoundTrip:
    def test_holdings_to_excel_bytes_returns_bytes(self):
        data = holdings_to_excel_bytes("My Fund", "MF", SAMPLE_HOLDINGS, SAMPLE_SETTINGS)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_excel_bytes_is_valid_xlsx(self):
        import openpyxl, io
        data = holdings_to_excel_bytes("My Fund", "MF", SAMPLE_HOLDINGS)
        wb = openpyxl.load_workbook(io.BytesIO(data))
        assert "Holdings" in wb.sheetnames

    def test_excel_path_to_holdings_round_trip(self, tmp_path):
        # Write Excel to temp file
        data = holdings_to_excel_bytes("Round Trip Fund", "RTF", SAMPLE_HOLDINGS, SAMPLE_SETTINGS)
        p = tmp_path / "test_portfolio.xlsx"
        p.write_bytes(data)

        holdings, name, short, settings = excel_path_to_holdings(str(p))

        assert name  == "Round Trip Fund"
        assert short == "RTF"
        assert len(holdings) == 3
        tickers = {h["ticker"] for h in holdings}
        assert tickers == {"AAPL", "MSFT", "NVDA"}

    def test_empty_holdings_excel(self, tmp_path):
        data = holdings_to_excel_bytes("Empty", "EMP", [])
        p = tmp_path / "empty.xlsx"
        p.write_bytes(data)
        holdings, name, short, _ = excel_path_to_holdings(str(p))
        assert holdings == []
        assert name == "Empty"
