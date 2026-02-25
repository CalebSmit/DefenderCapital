"""
auth/auth_manager.py
─────────────────────────────────────────────────────────────────────────────
User registration, login, and session helpers.
Passwords are hashed with bcrypt (never stored in plain text).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

import bcrypt

from auth.database import execute, execute_returning, fetchone, ph


# ── password helpers ───────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ── validation ─────────────────────────────────────────────────────────────

def _validate_username(username: str) -> str | None:
    """Return error message or None if valid."""
    if len(username) < 3:
        return "Username must be at least 3 characters."
    if len(username) > 50:
        return "Username must be 50 characters or fewer."
    if not re.match(r'^[A-Za-z0-9_.\-]+$', username):
        return "Username may only contain letters, numbers, underscores, hyphens, and dots."
    return None


def _validate_password(password: str) -> str | None:
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if len(password) > 128:
        return "Password must be 128 characters or fewer."
    return None


def _validate_email(email: str) -> str | None:
    if not re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email):
        return "Please enter a valid email address."
    if len(email) > 255:
        return "Email address is too long."
    return None


# ── public API ─────────────────────────────────────────────────────────────

def register_user(
    username: str,
    email: str,
    password: str,
    full_name: str = "",
    is_admin: bool = False,
) -> tuple[bool, str]:
    """
    Create a new user account.
    Returns (success: bool, message: str).
    """
    username = username.strip()
    email    = email.strip().lower()
    full_name = full_name.strip()

    err = _validate_username(username)
    if err:
        return False, err

    err = _validate_email(email)
    if err:
        return False, err

    err = _validate_password(password)
    if err:
        return False, err

    # Check duplicates
    if fetchone(f"SELECT id FROM dcm_users WHERE username = {ph()}", (username,)):
        return False, "Username is already taken."
    if fetchone(f"SELECT id FROM dcm_users WHERE email = {ph()}", (email,)):
        return False, "An account with that email already exists."

    pw_hash = hash_password(password)
    admin_val = True if is_admin else False   # bool for Postgres; int 1/0 for SQLite

    try:
        execute(
            f"INSERT INTO dcm_users (username, email, password_hash, full_name, is_admin) "
            f"VALUES ({ph(5)})",
            (username, email, pw_hash, full_name or None, admin_val),
        )
        return True, f"Account created for {username}."
    except Exception as exc:
        return False, f"Registration failed: {exc}"


def login_user(username: str, password: str) -> Optional[dict]:
    """
    Verify credentials.  *username* can be either a username or an email.
    Returns user dict on success, None on failure.
    """
    identifier = username.strip()
    # Try username first, then email
    row = fetchone(
        f"SELECT * FROM dcm_users WHERE username = {ph()}",
        (identifier,),
    )
    if row is None:
        row = fetchone(
            f"SELECT * FROM dcm_users WHERE email = {ph()}",
            (identifier.lower(),),
        )
    if row is None:
        return None
    if not verify_password(password, row["password_hash"]):
        return None

    # Update last_login timestamp
    try:
        now = datetime.now(timezone.utc).isoformat()
        execute(
            f"UPDATE dcm_users SET last_login = {ph()} WHERE id = {ph()}",
            (now, row["id"]),
        )
    except Exception:
        pass   # non-critical

    return {
        "id":        row["id"],
        "username":  row["username"],
        "email":     row["email"],
        "full_name": row.get("full_name") or "",
        "is_admin":  bool(row["is_admin"]),
    }


def get_user_by_id(user_id: int) -> Optional[dict]:
    row = fetchone(f"SELECT * FROM dcm_users WHERE id = {ph()}", (user_id,))
    if not row:
        return None
    return {
        "id":        row["id"],
        "username":  row["username"],
        "email":     row["email"],
        "full_name": row.get("full_name") or "",
        "is_admin":  bool(row["is_admin"]),
    }


def get_all_users() -> list[dict]:
    """Admin helper — list all users without exposing password hashes."""
    from auth.database import fetchall
    rows = fetchall(
        "SELECT id, username, email, full_name, is_admin, created_at, last_login "
        "FROM dcm_users ORDER BY created_at DESC"
    )
    return rows


def delete_user(user_id: int) -> bool:
    rows = execute(f"DELETE FROM dcm_users WHERE id = {ph()}", (user_id,))
    return rows > 0


def change_password(user_id: int, new_password: str) -> tuple[bool, str]:
    err = _validate_password(new_password)
    if err:
        return False, err
    pw_hash = hash_password(new_password)
    rows = execute(
        f"UPDATE dcm_users SET password_hash = {ph()} WHERE id = {ph()}",
        (pw_hash, user_id),
    )
    return (rows > 0), ("Password updated." if rows > 0 else "User not found.")
