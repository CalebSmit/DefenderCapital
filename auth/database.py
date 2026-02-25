"""
auth/database.py
─────────────────────────────────────────────────────────────────────────────
Database abstraction layer supporting:
  • PostgreSQL / Neon  (when NEON_DATABASE_URL is set in st.secrets or env)
  • SQLite             (automatic fallback for local dev and testing)

All public functions use the same interface regardless of backend.
"""

from __future__ import annotations

import json
import os
import sqlite3
import contextlib
from pathlib import Path
from typing import Any, Generator

# ── backend detection ──────────────────────────────────────────────────────

def _get_database_url() -> str | None:
    """Return the Neon/Postgres connection string, or None for SQLite."""
    # 1. Streamlit secrets (production on Streamlit Cloud)
    try:
        import streamlit as st
        url = st.secrets.get("NEON_DATABASE_URL", None)
        if url:
            return url
    except Exception:
        pass

    # 2. Environment variable (CI / Docker / local override)
    return os.environ.get("NEON_DATABASE_URL", None)


def _sqlite_path() -> Path:
    """Path to the local SQLite fallback database."""
    base = Path(__file__).parent.parent / "data"
    base.mkdir(exist_ok=True)
    return base / "dcm_users.db"


def _is_postgres() -> bool:
    return _get_database_url() is not None


# ── connection context manager ─────────────────────────────────────────────

@contextlib.contextmanager
def get_connection() -> Generator[Any, None, None]:
    """
    Yield a database connection.

    Usage:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(...)
            conn.commit()
    """
    if _is_postgres():
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(
            _get_database_url(),
            cursor_factory=psycopg2.extras.RealDictCursor,
        )
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        conn = sqlite3.connect(str(_sqlite_path()))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")   # better concurrent reads
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# ── schema init ────────────────────────────────────────────────────────────

_SCHEMA_POSTGRES = """
CREATE TABLE IF NOT EXISTS dcm_users (
    id            SERIAL PRIMARY KEY,
    username      VARCHAR(50)  UNIQUE NOT NULL,
    email         VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name     VARCHAR(100),
    is_admin      BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    last_login    TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS dcm_portfolios (
    id             SERIAL PRIMARY KEY,
    user_id        INTEGER NOT NULL REFERENCES dcm_users(id) ON DELETE CASCADE,
    portfolio_name VARCHAR(100) NOT NULL DEFAULT 'My Portfolio',
    short_name     VARCHAR(20)           DEFAULT 'PORT',
    holdings       JSONB        NOT NULL DEFAULT '[]',
    settings       JSONB                 DEFAULT '{}',
    updated_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (user_id)
);

CREATE TABLE IF NOT EXISTS dcm_portfolio_history (
    id             SERIAL PRIMARY KEY,
    user_id        INTEGER NOT NULL REFERENCES dcm_users(id) ON DELETE CASCADE,
    portfolio_name VARCHAR(100),
    holdings_json  JSONB,
    settings_json  JSONB,
    saved_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

_SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS dcm_users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    full_name     TEXT,
    is_admin      INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    last_login    TEXT
);

CREATE TABLE IF NOT EXISTS dcm_portfolios (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL REFERENCES dcm_users(id) ON DELETE CASCADE,
    portfolio_name TEXT    NOT NULL DEFAULT 'My Portfolio',
    short_name     TEXT             DEFAULT 'PORT',
    holdings       TEXT    NOT NULL DEFAULT '[]',
    settings       TEXT             DEFAULT '{}',
    updated_at     TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE (user_id)
);

CREATE TABLE IF NOT EXISTS dcm_portfolio_history (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL REFERENCES dcm_users(id) ON DELETE CASCADE,
    portfolio_name TEXT,
    holdings_json  TEXT,
    settings_json  TEXT,
    saved_at       TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    schema = _SCHEMA_POSTGRES if _is_postgres() else _SCHEMA_SQLITE
    with get_connection() as conn:
        cur = conn.cursor()
        # Split on semicolons so we can run each statement individually
        for stmt in [s.strip() for s in schema.split(";") if s.strip()]:
            cur.execute(stmt)


# ── generic query helpers ──────────────────────────────────────────────────

def fetchone(sql: str, params: tuple = ()) -> dict | None:
    """Return the first matching row as a plain dict, or None."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        row = cur.fetchone()
        if row is None:
            return None
        return dict(row)


def fetchall(sql: str, params: tuple = ()) -> list[dict]:
    """Return all matching rows as a list of plain dicts."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def execute(sql: str, params: tuple = ()) -> int:
    """Execute a write statement and return rowcount."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        return cur.rowcount


def execute_returning(sql: str, params: tuple = ()) -> dict | None:
    """
    Execute a write statement that includes RETURNING (Postgres) or
    simulate it on SQLite by fetching lastrowid.
    """
    if _is_postgres():
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None
    else:
        # SQLite: strip RETURNING clause, run, fetch by lastrowid
        import re
        base_sql = re.sub(r'\s+RETURNING\s+.*$', '', sql, flags=re.IGNORECASE | re.DOTALL)
        table_match = re.search(r'INSERT\s+INTO\s+(\w+)', base_sql, re.IGNORECASE)
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(base_sql, params)
            if table_match and cur.lastrowid:
                table = table_match.group(1)
                cur.execute(f"SELECT * FROM {table} WHERE id = ?", (cur.lastrowid,))
                row = cur.fetchone()
                return dict(row) if row else None
            return None


# ── JSON helpers (SQLite stores JSON as TEXT) ──────────────────────────────

def json_encode(obj: Any) -> Any:
    """Encode Python object to JSON string (SQLite) or pass through (Postgres)."""
    if _is_postgres():
        return json.dumps(obj)   # psycopg2 accepts dict/list natively but string is safe
    return json.dumps(obj)


def json_decode(value: Any) -> Any:
    """Decode a JSON value from either backend."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value            # Postgres JSONB already parsed
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


# ── placeholder helper ─────────────────────────────────────────────────────

def ph(n: int = 1) -> str:
    """
    Return the right placeholder for the current backend.
    Postgres uses %s, SQLite uses ?.

    ph()  → '%s' or '?'
    ph(3) → '%s, %s, %s' or '?, ?, ?'
    """
    mark = "%s" if _is_postgres() else "?"
    return ", ".join([mark] * n)


def backend_name() -> str:
    return "postgresql" if _is_postgres() else "sqlite"


# ── MED-7 FIX: Portfolio history functions ─────────────────────────────────

def save_portfolio_snapshot(
    user_id: int,
    portfolio_name: str,
    holdings_json: str,
    settings_json: str,
) -> int:
    """
    Save a snapshot of the portfolio to history.

    Parameters
    ----------
    user_id : int
        User ID
    portfolio_name : str
        Name of the portfolio
    holdings_json : str
        JSON-serialized holdings list
    settings_json : str
        JSON-serialized settings dict

    Returns
    -------
    int
        ID of the inserted history record
    """
    sql = f"""
    INSERT INTO dcm_portfolio_history (user_id, portfolio_name, holdings_json, settings_json)
    VALUES ({ph(4)})
    RETURNING id
    """
    result = execute_returning(sql, (user_id, portfolio_name, holdings_json, settings_json))
    return result.get("id") if result else None


def get_portfolio_history(user_id: int, limit: int = 100) -> list[dict]:
    """
    Retrieve portfolio history snapshots for a user.

    Parameters
    ----------
    user_id : int
        User ID
    limit : int
        Maximum number of records to return (default 100)

    Returns
    -------
    list[dict]
        List of portfolio history records, ordered by saved_at descending
    """
    sql = f"""
    SELECT id, user_id, portfolio_name, holdings_json, settings_json, saved_at
    FROM dcm_portfolio_history
    WHERE user_id = {ph()}
    ORDER BY saved_at DESC
    LIMIT {ph()}
    """
    return fetchall(sql, (user_id, limit))


# ─────────────────────────────────────────────────────────────────────────────
# LOW-6: Login Rate Limiting
# ─────────────────────────────────────────────────────────────────────────────

import time

# In-memory tracking of login attempts per username
_login_attempts: dict = {}


def check_and_record_login_attempt(username: str, success: bool) -> dict:
    """
    Track login attempts and enforce rate limiting.

    Parameters
    ----------
    username : str
        Username of the account
    success : bool
        True if authentication succeeded, False if it failed

    Returns
    -------
    dict
        {
            "locked": bool,
            "remaining_attempts": int (0-5),
            "lockout_seconds": float (0 if not locked)
        }

    Logic
    -----
    - On failure: increment count; if count >= 5, set 15-min lockout
    - On success: reset counter to 0
    """
    current_time = time.time()

    if username not in _login_attempts:
        _login_attempts[username] = {"count": 0, "lockout_until": None}

    record = _login_attempts[username]

    # Check if lockout has expired
    if record["lockout_until"] is not None and current_time >= record["lockout_until"]:
        record["lockout_until"] = None
        record["count"] = 0

    if success:
        # Success: reset counter
        record["count"] = 0
        record["lockout_until"] = None
        remaining = 5
        locked = False
        lockout_sec = 0.0
    else:
        # Failure: increment count
        record["count"] += 1
        if record["count"] >= 5:
            # Lock out for 15 minutes (900 seconds)
            record["lockout_until"] = current_time + 900
            locked = True
            lockout_sec = 900.0
            remaining = 0
        else:
            locked = False
            lockout_sec = 0.0
            remaining = 5 - record["count"]

    return {
        "locked": locked,
        "remaining_attempts": remaining,
        "lockout_seconds": lockout_sec,
    }


def is_account_locked(username: str) -> dict:
    """
    Check if an account is currently locked out.

    Parameters
    ----------
    username : str
        Username to check

    Returns
    -------
    dict
        {
            "locked": bool,
            "lockout_seconds_remaining": float
        }
    """
    current_time = time.time()

    if username not in _login_attempts:
        return {"locked": False, "lockout_seconds_remaining": 0.0}

    record = _login_attempts[username]

    # Check if lockout has expired
    if record["lockout_until"] is not None and current_time >= record["lockout_until"]:
        record["lockout_until"] = None
        record["count"] = 0

    if record["lockout_until"] is None:
        return {"locked": False, "lockout_seconds_remaining": 0.0}

    remaining = record["lockout_until"] - current_time
    return {
        "locked": True,
        "lockout_seconds_remaining": max(remaining, 0.0),
    }
