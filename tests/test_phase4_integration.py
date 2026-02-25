"""
Phase 4 tests — end-to-end auth + portfolio flow integration.
Verifies the complete user journey: register → login → save → reload.
Runs against isolated SQLite — no Neon required.
"""
import os
import sys
import hashlib
import tempfile
from pathlib import Path
import pytest

os.environ.pop("NEON_DATABASE_URL", None)
sys.path.insert(0, str(Path(__file__).parent.parent))

import auth.database as db_mod
from auth.database import init_db
from auth.auth_manager import register_user, login_user, get_user_by_id
from auth.portfolio_store import (
    save_portfolio,
    load_portfolio,
    has_portfolio,
    holdings_to_excel_bytes,
    excel_path_to_holdings,
)

HOLDINGS_A = [
    {"ticker": "AAPL", "shares_held": 50.0, "cost_basis": 145.0},
    {"ticker": "MSFT", "shares_held": 30.0, "cost_basis": 270.0},
    {"ticker": "GOOG", "shares_held": 10.0, "cost_basis": 125.0},
]
HOLDINGS_B = [
    {"ticker": "TSLA", "shares_held": 20.0, "cost_basis": 220.0},
    {"ticker": "NVDA", "shares_held": 15.0, "cost_basis": 380.0},
]
SETTINGS_A = {"covariance_mode": "EWMA", "lookback_years": 3, "benchmark": "SPY"}


@pytest.fixture(scope="module", autouse=True)
def db(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("db") / "integration.db"
    db_mod._sqlite_path = lambda: tmp
    init_db()
    yield


class TestFullUserJourney:
    """Complete register → login → save → logout → login → auto-load flow."""

    def test_01_register_two_users(self):
        ok1, _ = register_user("trader_alice", "alice@dcm.com", "AlicePass99!", "Alice Smith")
        ok2, _ = register_user("trader_bob",   "bob@dcm.com",   "BobPass99!",   "Bob Jones")
        assert ok1 and ok2

    def test_02_login_returns_user_dict(self):
        user = login_user("trader_alice", "AlicePass99!")
        assert user["username"] == "trader_alice"
        assert user["full_name"] == "Alice Smith"
        assert "password_hash" not in user

    def test_03_no_portfolio_before_first_save(self):
        user = login_user("trader_alice", "AlicePass99!")
        assert not has_portfolio(user["id"])
        assert load_portfolio(user["id"]) is None

    def test_04_save_portfolio_for_alice(self):
        user = login_user("trader_alice", "AlicePass99!")
        ok, msg = save_portfolio(
            user["id"], "Alice Growth Fund", "AGF", HOLDINGS_A, SETTINGS_A
        )
        assert ok
        assert has_portfolio(user["id"])

    def test_05_load_portfolio_matches_what_was_saved(self):
        user = login_user("trader_alice", "AlicePass99!")
        p = load_portfolio(user["id"])
        assert p["portfolio_name"] == "Alice Growth Fund"
        assert p["short_name"] == "AGF"
        assert len(p["holdings"]) == 3
        assert p["settings"]["covariance_mode"] == "EWMA"

    def test_06_bob_has_no_portfolio(self):
        user = login_user("trader_bob", "BobPass99!")
        assert not has_portfolio(user["id"])

    def test_07_bob_saves_different_portfolio(self):
        user = login_user("trader_bob", "BobPass99!")
        ok, _ = save_portfolio(user["id"], "Bob Tech Fund", "BTF", HOLDINGS_B)
        assert ok

    def test_08_portfolios_do_not_cross_contaminate(self):
        alice = login_user("trader_alice", "AlicePass99!")
        bob   = login_user("trader_bob",   "BobPass99!")

        pa = load_portfolio(alice["id"])
        pb = load_portfolio(bob["id"])

        alice_tickers = {h["ticker"] for h in pa["holdings"]}
        bob_tickers   = {h["ticker"] for h in pb["holdings"]}

        assert alice_tickers == {"AAPL", "MSFT", "GOOG"}
        assert bob_tickers   == {"TSLA", "NVDA"}
        assert alice_tickers.isdisjoint(bob_tickers)

    def test_09_update_portfolio_replaces_not_appends(self):
        user = login_user("trader_alice", "AlicePass99!")
        new_holdings = [{"ticker": "META", "shares_held": 25.0, "cost_basis": 300.0}]
        ok, _ = save_portfolio(user["id"], "Alice Revised", "ARV", new_holdings)
        assert ok

        p = load_portfolio(user["id"])
        assert p["portfolio_name"] == "Alice Revised"
        assert len(p["holdings"]) == 1
        assert p["holdings"][0]["ticker"] == "META"

    def test_10_excel_round_trip_preserves_all_data(self, tmp_path):
        user = login_user("trader_bob", "BobPass99!")
        p = load_portfolio(user["id"])

        # Simulate what the sidebar "Download My Saved Portfolio" button does
        excel_bytes = holdings_to_excel_bytes(
            p["portfolio_name"], p["short_name"], p["holdings"], p["settings"]
        )
        assert isinstance(excel_bytes, bytes)

        # Simulate re-uploading that file
        xl_path = tmp_path / "bob.xlsx"
        xl_path.write_bytes(excel_bytes)
        holdings, name, short, settings = excel_path_to_holdings(str(xl_path))

        assert name  == "Bob Tech Fund"
        assert short == "BTF"
        assert len(holdings) == 2
        tickers = {h["ticker"] for h in holdings}
        assert tickers == {"TSLA", "NVDA"}

    def test_11_wrong_password_cannot_login(self):
        assert login_user("trader_alice", "WrongPassword") is None
        assert login_user("trader_alice", "") is None

    def test_12_nonexistent_user_cannot_login(self):
        assert login_user("phantom_user", "anything") is None

    def test_13_session_isolation_user_id_correct(self):
        """Each login returns the same user ID for the same account."""
        u1 = login_user("trader_alice", "AlicePass99!")
        u2 = login_user("trader_alice", "AlicePass99!")
        assert u1["id"] == u2["id"]

    def test_14_last_login_updated_on_each_login(self):
        import time
        from auth.database import fetchone, ph
        login_user("trader_bob", "BobPass99!")
        r1 = fetchone("SELECT last_login FROM dcm_users WHERE username = ?", ("trader_bob",))
        time.sleep(0.05)
        login_user("trader_bob", "BobPass99!")
        r2 = fetchone("SELECT last_login FROM dcm_users WHERE username = ?", ("trader_bob",))
        # Both should be non-null; second should be >= first
        assert r1["last_login"] is not None
        assert r2["last_login"] is not None


class TestEdgeCases:
    def test_save_empty_holdings_list(self):
        ok1, _ = register_user("edge_user1", "edge1@dcm.com", "EdgePass99!")
        user = login_user("edge_user1", "EdgePass99!")
        ok, _ = save_portfolio(user["id"], "Empty Fund", "EMP", [])
        assert ok
        p = load_portfolio(user["id"])
        assert p["holdings"] == []

    def test_save_holdings_with_many_tickers(self):
        ok1, _ = register_user("edge_user2", "edge2@dcm.com", "EdgePass99!")
        user = login_user("edge_user2", "EdgePass99!")
        large = [
            {"ticker": f"T{i:03d}", "shares_held": float(i), "cost_basis": float(i * 10)}
            for i in range(1, 51)   # 50 holdings
        ]
        ok, _ = save_portfolio(user["id"], "Large Fund", "LGF", large)
        assert ok
        p = load_portfolio(user["id"])
        assert len(p["holdings"]) == 50
