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
