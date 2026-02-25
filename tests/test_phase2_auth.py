"""
Phase 2 tests — auth_manager (register / login / password)
Runs against isolated SQLite — no Neon required.
"""
import os
import sys
import pytest

os.environ.pop("NEON_DATABASE_URL", None)
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import auth.database as db_mod
from auth.database import init_db
from auth.auth_manager import (
    register_user,
    login_user,
    get_user_by_id,
    get_all_users,
    delete_user,
    change_password,
    hash_password,
    verify_password,
)


@pytest.fixture(scope="module", autouse=True)
def fresh_db(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("db") / "auth_test.db"
    db_mod._sqlite_path = lambda: tmp
    init_db()
    yield


# ── password helpers ─────────────────────────────────────────────────────────

class TestPasswordHashing:
    def test_hash_is_different_from_plain(self):
        h = hash_password("secret123")
        assert h != "secret123"
        assert h.startswith("$2b$")

    def test_verify_correct_password(self):
        h = hash_password("mypassword")
        assert verify_password("mypassword", h) is True

    def test_verify_wrong_password(self):
        h = hash_password("mypassword")
        assert verify_password("wrongpassword", h) is False

    def test_two_hashes_of_same_password_differ(self):
        """bcrypt uses random salt — same input → different hashes."""
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2
        # but both still verify correctly
        assert verify_password("same", h1)
        assert verify_password("same", h2)


# ── registration ─────────────────────────────────────────────────────────────

class TestRegister:
    def test_successful_registration(self):
        ok, msg = register_user("testuser1", "test1@example.com", "password123", "Test User")
        assert ok is True
        assert "testuser1" in msg

    def test_duplicate_username_rejected(self):
        register_user("dupuser", "dup1@example.com", "password123")
        ok, msg = register_user("dupuser", "dup2@example.com", "password123")
        assert ok is False
        assert "taken" in msg.lower()

    def test_duplicate_email_rejected(self):
        register_user("userA", "shared@example.com", "password123")
        ok, msg = register_user("userB", "shared@example.com", "password123")
        assert ok is False
        assert "email" in msg.lower()

    def test_short_username_rejected(self):
        ok, msg = register_user("ab", "short@example.com", "password123")
        assert ok is False
        assert "3 characters" in msg

    def test_short_password_rejected(self):
        ok, msg = register_user("validuser99", "v99@example.com", "short")
        assert ok is False
        assert "8 characters" in msg

    def test_invalid_email_rejected(self):
        ok, msg = register_user("emailtest", "notanemail", "password123")
        assert ok is False
        assert "email" in msg.lower()

    def test_invalid_username_chars(self):
        ok, msg = register_user("bad user!", "bu@example.com", "password123")
        assert ok is False

    def test_admin_flag(self):
        ok, msg = register_user("adminuser", "admin@example.com", "adminpass1", is_admin=True)
        assert ok is True
        from auth.database import fetchone, ph
        row = fetchone("SELECT is_admin FROM dcm_users WHERE username = ?", ("adminuser",))
        assert bool(row["is_admin"]) is True


# ── login ────────────────────────────────────────────────────────────────────

class TestLogin:
    @pytest.fixture(autouse=True)
    def create_user(self):
        register_user("loginuser", "login@example.com", "correctpass1")

    def test_login_correct_credentials(self):
        user = login_user("loginuser", "correctpass1")
        assert user is not None
        assert user["username"] == "loginuser"

    def test_login_wrong_password(self):
        user = login_user("loginuser", "wrongpassword")
        assert user is None

    def test_login_nonexistent_user(self):
        user = login_user("ghostuser", "anypassword")
        assert user is None

    def test_login_returns_expected_keys(self):
        user = login_user("loginuser", "correctpass1")
        assert user is not None
        for key in ("id", "username", "email", "full_name", "is_admin"):
            assert key in user

    def test_password_hash_not_in_login_result(self):
        user = login_user("loginuser", "correctpass1")
        assert "password_hash" not in user

    def test_login_updates_last_login(self):
        from auth.database import fetchone, ph
        login_user("loginuser", "correctpass1")
        row = fetchone("SELECT last_login FROM dcm_users WHERE username = ?", ("loginuser",))
        assert row["last_login"] is not None


# ── user management ───────────────────────────────────────────────────────────

class TestUserManagement:
    @pytest.fixture(autouse=True)
    def create_user(self):
        register_user("mgmtuser", "mgmt@example.com", "mgmtpass1", "Mgmt User")

    def test_get_user_by_id(self):
        user = login_user("mgmtuser", "mgmtpass1")
        fetched = get_user_by_id(user["id"])
        assert fetched is not None
        assert fetched["username"] == "mgmtuser"

    def test_get_user_by_invalid_id(self):
        assert get_user_by_id(999999) is None

    def test_get_all_users_returns_list(self):
        users = get_all_users()
        assert isinstance(users, list)
        assert len(users) >= 1

    def test_change_password(self):
        user = login_user("mgmtuser", "mgmtpass1")
        ok, msg = change_password(user["id"], "newpassword99")
        assert ok is True
        assert login_user("mgmtuser", "newpassword99") is not None
        assert login_user("mgmtuser", "mgmtpass1") is None

    def test_change_password_too_short(self):
        user = login_user("mgmtuser", "newpassword99")
        ok, msg = change_password(user["id"], "short")
        assert ok is False

    def test_delete_user(self):
        register_user("deleteuser", "del@example.com", "deletepass1")
        user = login_user("deleteuser", "deletepass1")
        ok = delete_user(user["id"])
        assert ok is True
        assert get_user_by_id(user["id"]) is None
