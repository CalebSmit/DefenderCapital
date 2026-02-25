"""
dashboard/login_page.py
─────────────────────────────────────────────────────────────────────────────
Renders the login / registration gate.
Call render_auth_gate() at the very top of app.py before any other content.
It returns True when the user is authenticated, False otherwise (page stops).
"""

from __future__ import annotations

import streamlit as st
from auth.database import init_db
from auth.auth_manager import login_user, register_user

# ── DCM design tokens (match main app palette) ─────────────────────────────
_GOLD      = "#c9a84c"
_INK       = "#1a1a1a"
_CANVAS    = "#faf8f4"
_SURFACE   = "#f5f2eb"
_MUTED     = "#8a7f6e"
_BORDER    = "#e8e0cc"

_LOGIN_CSS = f"""
<style>
/* ── hide default Streamlit chrome on the auth page ── */
[data-testid="stToolbar"],
[data-testid="stDecoration"],
footer {{ display: none !important; }}

/* ── narrow the main block so the form sits centered at ~420px ── */
.main .block-container {{
    max-width: 460px !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}}

/* ── full-page auth layout ── */
.dcm-auth-outer {{
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: {_CANVAS};
    padding: 40px 16px;
}}
.dcm-auth-card {{
    background: #ffffff;
    border: 1px solid {_BORDER};
    border-radius: 16px;
    padding: 48px 44px 40px;
    width: 100%;
    max-width: 420px;
    box-shadow: 0 4px 32px rgba(0,0,0,0.07);
}}
.dcm-auth-logo {{
    font-family: 'Cinzel', 'Georgia', serif;
    font-size: 22px;
    font-weight: 700;
    color: {_GOLD};
    letter-spacing: 0.12em;
    text-transform: uppercase;
    text-align: center;
    margin-bottom: 4px;
}}
.dcm-auth-sub {{
    font-size: 12px;
    color: {_MUTED};
    text-align: center;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 32px;
}}
.dcm-auth-divider {{
    height: 1px;
    background: {_BORDER};
    margin: 24px 0;
}}
</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&display=swap" rel="stylesheet">
"""


def _logo() -> None:
    st.markdown(
        '<div class="dcm-auth-logo">Defender Capital</div>'
        '<div class="dcm-auth-sub">Risk Analytics Platform</div>',
        unsafe_allow_html=True,
    )


def _render_login() -> None:
    """Login form tab."""
    from auth.database import is_account_locked, check_and_record_login_attempt
    
    with st.form("dcm_login_form", clear_on_submit=False):
        identifier = st.text_input(
            "Username or Email",
            placeholder="your username or email",
        )
        password = st.text_input("Password", type="password", placeholder="••••••••")
        submitted = st.form_submit_button(
            "Sign In",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        if not identifier or not password:
            st.error("Please enter your username/email and password.")
            return

        # Check if account is locked before attempting auth
        lock_status = is_account_locked(identifier.strip())
        if lock_status["locked"]:
            mins_remaining = int(lock_status["lockout_seconds_remaining"] / 60) + 1
            st.error(f"Account locked due to too many failed attempts. Try again in {mins_remaining} minutes.")
            return

        user = login_user(identifier.strip(), password)
        if user:
            # Successful login: reset the counter
            check_and_record_login_attempt(identifier.strip(), success=True)
            st.session_state["dcm_user"]          = user
            st.session_state["dcm_authenticated"] = True
            st.rerun()
        else:
            # Failed login: record the attempt
            attempt_status = check_and_record_login_attempt(identifier.strip(), success=False)
            if attempt_status["locked"]:
                mins = int(attempt_status["lockout_seconds"] / 60)
                st.error(f"Incorrect username/email or password. Account locked for {mins} minutes.")
            else:
                remaining = attempt_status["remaining_attempts"]
                st.error(f"Incorrect username/email or password. {remaining} attempts remaining before lockout.")


def _render_register() -> None:
    """Registration form tab."""
    with st.form("dcm_register_form", clear_on_submit=True):
        full_name = st.text_input("Full Name", placeholder="Jane Smith")
        username  = st.text_input("Username", placeholder="jsmith")
        email     = st.text_input("Email", placeholder="jane@example.com")
        password  = st.text_input(
            "Password",
            type="password",
            placeholder="At least 8 characters",
        )
        password2 = st.text_input(
            "Confirm Password",
            type="password",
            placeholder="Repeat password",
        )
        submitted = st.form_submit_button(
            "Create Account",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        if not all([full_name, username, email, password, password2]):
            st.error("Please fill in all fields.")
            return
        if password != password2:
            st.error("Passwords do not match.")
            return

        ok, msg = register_user(
            username=username,
            email=email,
            password=password,
            full_name=full_name,
        )
        if ok:
            # Auto-login after successful registration
            user = login_user(username.strip(), password)
            if user:
                st.session_state["dcm_user"]          = user
                st.session_state["dcm_authenticated"] = True
                st.rerun()
            else:
                # Fallback: account created but auto-login failed for some reason
                st.success(f"{msg} You can now sign in.")
        else:
            st.error(msg)


def render_auth_gate() -> bool:
    """
    Call this at the very top of app.py.
    Returns True  → user is authenticated, main app may render.
    Returns False → login page is showing, main app must not render.
    """
    # Ensure DB tables exist on every cold start
    try:
        init_db()
    except Exception as exc:
        st.error(f"Database initialisation error: {exc}")
        st.stop()

    # Already authenticated this session
    if st.session_state.get("dcm_authenticated") and st.session_state.get("dcm_user"):
        return True

    # ── show auth page ────────────────────────────────────────────────────
    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)
    _logo()

    tab_login, tab_register = st.tabs(["Sign In", "Create Account"])

    with tab_login:
        _render_login()

    with tab_register:
        _render_register()

    return False   # block the rest of app.py


def logout() -> None:
    """Call this when the user clicks Logout in the sidebar."""
    for key in ["dcm_user", "dcm_authenticated"]:
        st.session_state.pop(key, None)
    # Also clear any cached portfolio so the next user starts fresh
    for key in ["uploaded_portfolio_path", "uploaded_portfolio_hash",
                "_util_page", "dcm_portfolio_loaded"]:
        st.session_state.pop(key, None)
    st.rerun()


def current_user() -> dict | None:
    """Return the currently logged-in user dict, or None."""
    return st.session_state.get("dcm_user")
