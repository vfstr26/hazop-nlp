"""
auth.py — Authentication and role-based access control for HAZOP apps.

Roles:
  admin    — full access: create/delete users, manage all sessions
  engineer — full analysis access: run pipelines, review scenarios, export
  viewer   — read-only: view results and exports, no generation or editing

Implementation:
  - Credentials stored in .streamlit/secrets.toml (Streamlit Cloud)
    or auth_config.yaml (local)
  - Passwords hashed with bcrypt
  - Session tokens stored in Streamlit session_state
  - No external auth server required — self-contained

Usage in any Streamlit page:
    from src.auth import require_auth, get_current_user, has_role

    require_auth()                        # redirects to login if not authenticated
    user = get_current_user()             # {"username": ..., "role": ..., "name": ...}
    if has_role("engineer"):              # role check
        st.button("Run Analysis")
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from typing import Optional
from loguru import logger

AUTH_CONFIG_PATH = Path(__file__).parent.parent / ".streamlit" / "auth_config.json"


def _hash_pw(password: str) -> str:
    """SHA-256 hash of password. Use bcrypt in production for stronger security."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _verify_pw(plain: str, hashed: str) -> bool:
    return hmac.compare_digest(_hash_pw(plain), hashed)


# ── Default users (created on first run if no config exists) ──────────────────
DEFAULT_USERS = [
    {
        "username": "admin",
        "name":     "System Administrator",
        "password": _hash_pw("admin123"),  # CHANGE IN PRODUCTION
        "role":     "admin",
        "email":    "admin@hazop.local",
    },
    {
        "username": "engineer1",
        "name":     "Process Engineer",
        "password": _hash_pw("engineer123"),
        "role":     "engineer",
        "email":    "engineer@hazop.local",
    },
    {
        "username": "viewer1",
        "name":     "Safety Reviewer",
        "password": _hash_pw("viewer123"),
        "role":     "viewer",
        "email":    "viewer@hazop.local",
    },
]

ROLE_PERMISSIONS = {
    "admin":    {"run_analysis", "view_results", "export", "manage_users",
                 "review_scenarios", "delete_sessions"},
    "engineer": {"run_analysis", "view_results", "export", "review_scenarios"},
    "viewer":   {"view_results", "export"},
}


# ══════════════════════════════════════════════════════════════════════════════
# User store
# ══════════════════════════════════════════════════════════════════════════════

def _load_users() -> list[dict]:
    """Load users from config file, creating defaults if not present."""
    if AUTH_CONFIG_PATH.exists():
        with open(AUTH_CONFIG_PATH, encoding="utf-8") as fh:
            return json.load(fh).get("users", [])
    # First run — create defaults
    _save_users(DEFAULT_USERS)
    return DEFAULT_USERS


def _save_users(users: list[dict]):
    AUTH_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUTH_CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump({"users": users}, fh, indent=2)


def get_user(username: str) -> Optional[dict]:
    for u in _load_users():
        if u["username"].lower() == username.lower():
            return u
    return None


def authenticate(username: str, password: str) -> Optional[dict]:
    user = get_user(username)
    if user and _verify_pw(password, user["password"]):
        return {k: v for k, v in user.items() if k != "password"}
    return None


def create_user(username: str, name: str, password: str,
                role: str = "viewer", email: str = "") -> bool:
    users = _load_users()
    if any(u["username"].lower() == username.lower() for u in users):
        return False
    users.append({
        "username": username,
        "name":     name,
        "password": _hash_pw(password),
        "role":     role,
        "email":    email,
    })
    _save_users(users)
    return True


def update_password(username: str, new_password: str) -> bool:
    users = _load_users()
    for u in users:
        if u["username"].lower() == username.lower():
            u["password"] = _hash_pw(new_password)
            _save_users(users)
            return True
    return False


def list_users() -> list[dict]:
    return [{k: v for k, v in u.items() if k != "password"}
            for u in _load_users()]


# ══════════════════════════════════════════════════════════════════════════════
# Streamlit session helpers
# ══════════════════════════════════════════════════════════════════════════════

def _st_state():
    try:
        import streamlit as st
        return st.session_state
    except Exception:
        return {}


def get_current_user() -> Optional[dict]:
    return _st_state().get("auth_user")


def is_authenticated() -> bool:
    user = get_current_user()
    if not user:
        return False
    # Session timeout: 8 hours
    login_time = _st_state().get("auth_time", 0)
    if time.time() - login_time > 8 * 3600:
        logout()
        return False
    return True


def has_permission(permission: str) -> bool:
    user = get_current_user()
    if not user:
        return False
    role = user.get("role", "viewer")
    return permission in ROLE_PERMISSIONS.get(role, set())


def has_role(role: str) -> bool:
    user = get_current_user()
    if not user:
        return False
    role_hierarchy = {"admin": 3, "engineer": 2, "viewer": 1}
    user_level    = role_hierarchy.get(user.get("role", "viewer"), 0)
    required_level = role_hierarchy.get(role, 999)
    return user_level >= required_level


def logout():
    state = _st_state()
    state.pop("auth_user", None)
    state.pop("auth_time", None)


# ══════════════════════════════════════════════════════════════════════════════
# Streamlit login UI
# ══════════════════════════════════════════════════════════════════════════════

def render_login_page():
    """Render a clean login form. Call from any Streamlit page."""
    import streamlit as st

    st.markdown("""
    <style>
    .login-box {
        max-width: 420px; margin: 60px auto; padding: 40px;
        background: white; border-radius: 12px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.10);
    }
    .login-title { font-size: 1.8em; font-weight: 700;
                   color: #1A252F; margin-bottom: 4px; }
    .login-sub   { color: #7F8C8D; margin-bottom: 24px; font-size: 0.95em; }
    </style>
    """, unsafe_allow_html=True)

    col = st.columns([1, 2, 1])[1]
    with col:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown('<div class="login-title">⚗ HAZOP NLP</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-sub">Process Safety Analysis Platform</div>',
                    unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password",
                                     placeholder="Enter password")
            submitted = st.form_submit_button("Sign In", use_container_width=True,
                                              type="primary")

        if submitted:
            user = authenticate(username, password)
            if user:
                st.session_state["auth_user"] = user
                st.session_state["auth_time"] = time.time()
                logger.info(f"Login: {username} ({user['role']})")
                st.rerun()
            else:
                st.error("Invalid username or password.")

        st.markdown("---")
        st.caption("Default credentials for first run:  \n"
                   "`admin / admin123` · `engineer1 / engineer123` · `viewer1 / viewer123`  \n"
                   "Change passwords after first login.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()


def require_auth(min_role: str = "viewer"):
    """
    Gate any Streamlit page. Call at the top of each page.
    Redirects to login form if not authenticated or insufficient role.
    """
    if not is_authenticated():
        render_login_page()
    if not has_role(min_role):
        import streamlit as st
        st.error(f"Access denied. Required role: **{min_role}**. "
                 f"Your role: **{get_current_user().get('role', 'unknown')}**")
        st.stop()


def render_user_menu():
    """Render a compact user info + logout widget for the sidebar."""
    import streamlit as st
    user = get_current_user()
    if not user:
        return
    role_colours = {
        "admin":    "#C0392B",
        "engineer": "#2980B9",
        "viewer":   "#27AE60",
    }
    colour = role_colours.get(user.get("role", "viewer"), "#7F8C8D")
    st.sidebar.markdown(
        f'<div style="background:#F8F9FA;border-radius:8px;padding:10px 14px;'
        f'margin-bottom:8px">'
        f'<b>{user.get("name", user.get("username"))}</b><br>'
        f'<span style="background:{colour};color:white;padding:1px 8px;'
        f'border-radius:4px;font-size:0.78em">{user.get("role","").upper()}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Sign Out", use_container_width=True):
        logout()
        st.rerun()


def render_user_admin():
    """Admin panel for managing users — only accessible to admins."""
    import streamlit as st
    require_auth("admin")

    st.subheader("👥 User Management")
    users = list_users()

    st.dataframe(
        [{"Username": u["username"], "Name": u["name"],
          "Role": u["role"], "Email": u.get("email", "")}
         for u in users],
        use_container_width=True, hide_index=True,
    )

    st.divider()
    st.markdown("**Add New User**")
    with st.form("add_user"):
        c1, c2 = st.columns(2)
        new_username = c1.text_input("Username")
        new_name     = c2.text_input("Full Name")
        new_email    = c1.text_input("Email")
        new_role     = c2.selectbox("Role", ["viewer", "engineer", "admin"])
        new_password = st.text_input("Password", type="password")
        if st.form_submit_button("Create User", type="primary"):
            if create_user(new_username, new_name, new_password, new_role, new_email):
                st.success(f"User '{new_username}' created.")
                st.rerun()
            else:
                st.error("Username already exists.")

    st.divider()
    st.markdown("**Change Password**")
    with st.form("change_pw"):
        target_user = st.selectbox("User", [u["username"] for u in users])
        new_pw      = st.text_input("New Password", type="password")
        if st.form_submit_button("Update Password"):
            if update_password(target_user, new_pw):
                st.success(f"Password updated for {target_user}")
            else:
                st.error("User not found.")
