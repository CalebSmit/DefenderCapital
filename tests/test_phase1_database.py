"""
Phase 1 tests — database module
Runs against local SQLite (no Neon required).
"""
import json
import os
import sys
import pytest

# Ensure we run against SQLite (no DATABASE_URL set)
os.environ.pop("NEON_DATABASE_URL", None)

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from auth.database import (
    init_db,
    get_connection,
    fetchone,
    fetchall,
    execute,
    execute_returning,
    json_encode,
    json_decode,
    ph,
    backend_name,
    _sqlite_path,
)


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def fresh_db(tmp_path_factory):
    """Point SQLite at a temp file so tests don't pollute real data."""
    import auth.database as db_mod
    tmp = tmp_path_factory.mktemp("db") / "test.db"
    db_mod._sqlite_path = lambda: tmp   # monkey-patch
    init_db()
    yield
    # cleanup handled by tmp_path_factory


# ── tests ────────────────────────────────────────────────────────────────────

class TestBackend:
    def test_backend_is_sqlite(self):
        assert backend_name() == "sqlite"

    def test_placeholder_is_question_mark(self):
        assert ph() == "?"
        assert ph(3) == "?, ?, ?"


class TestSchema:
    def test_tables_created(self):
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row["name"] for row in cur.fetchall()}
        assert "dcm_users" in tables
        assert "dcm_portfolios" in tables

    def test_init_db_idempotent(self):
        """Calling init_db twice should not raise."""
        init_db()
        init_db()


class TestCRUD:
    def test_execute_insert(self):
        rows = execute(
            f"INSERT INTO dcm_users (username, email, password_hash) VALUES ({ph(3)})",
            ("alice", "alice@example.com", "hashed_pw_1"),
        )
        assert rows == 1

    def test_fetchone_found(self):
        execute(
            f"INSERT INTO dcm_users (username, email, password_hash) VALUES ({ph(3)})",
            ("bob", "bob@example.com", "hashed_pw_2"),
        )
        row = fetchone(f"SELECT * FROM dcm_users WHERE username = {ph()}", ("bob",))
        assert row is not None
        assert row["username"] == "bob"
        assert row["email"] == "bob@example.com"

    def test_fetchone_missing(self):
        row = fetchone(f"SELECT * FROM dcm_users WHERE username = {ph()}", ("nobody",))
        assert row is None

    def test_fetchall(self):
        rows = fetchall("SELECT * FROM dcm_users")
        assert len(rows) >= 2
        usernames = [r["username"] for r in rows]
        assert "alice" in usernames
        assert "bob" in usernames

    def test_execute_returning_insert(self):
        mark = ph()
        result = execute_returning(
            f"INSERT INTO dcm_users (username, email, password_hash) VALUES ({ph(3)}) RETURNING id",
            ("carol", "carol@example.com", "hashed_pw_3"),
        )
        assert result is not None
        assert "id" in result
        assert isinstance(result["id"], int)

    def test_unique_constraint(self):
        with pytest.raises(Exception):
            execute(
                f"INSERT INTO dcm_users (username, email, password_hash) VALUES ({ph(3)})",
                ("alice", "different@example.com", "pw"),
            )


class TestJSON:
    def test_json_encode_decode_list(self):
        data = [{"ticker": "AAPL", "shares": 10}, {"ticker": "MSFT", "shares": 5}]
        encoded = json_encode(data)
        assert isinstance(encoded, str)
        decoded = json_decode(encoded)
        assert decoded == data

    def test_json_encode_decode_dict(self):
        data = {"key": "value", "num": 42, "nested": {"a": 1}}
        assert json_decode(json_encode(data)) == data

    def test_json_decode_none(self):
        assert json_decode(None) is None

    def test_json_decode_already_parsed(self):
        # Postgres JSONB returns dicts/lists directly
        data = [1, 2, 3]
        assert json_decode(data) == data


class TestPortfolioTable:
    def test_insert_portfolio(self):
        user = fetchone(f"SELECT id FROM dcm_users WHERE username = {ph()}", ("alice",))
        holdings = json_encode([{"ticker": "AAPL", "shares": 10, "cost_basis": 150.0}])
        settings = json_encode({"covariance_mode": "EWMA"})
        rows = execute(
            f"INSERT INTO dcm_portfolios (user_id, portfolio_name, short_name, holdings, settings) "
            f"VALUES ({ph(5)})",
            (user["id"], "Alice Fund", "ALF", holdings, settings),
        )
        assert rows == 1

    def test_load_portfolio(self):
        user = fetchone(f"SELECT id FROM dcm_users WHERE username = {ph()}", ("alice",))
        row = fetchone(
            f"SELECT * FROM dcm_portfolios WHERE user_id = {ph()}",
            (user["id"],),
        )
        assert row is not None
        assert row["portfolio_name"] == "Alice Fund"
        holdings = json_decode(row["holdings"])
        assert isinstance(holdings, list)
        assert holdings[0]["ticker"] == "AAPL"

    def test_one_portfolio_per_user(self):
        user = fetchone(f"SELECT id FROM dcm_users WHERE username = {ph()}", ("alice",))
        with pytest.raises(Exception):
            execute(
                f"INSERT INTO dcm_portfolios (user_id, portfolio_name, holdings) VALUES ({ph(3)})",
                (user["id"], "Duplicate", "[]"),
            )
