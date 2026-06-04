"""
auth_handler.py — Streamlit UI components for login, signup, and session management.

The authentication flow is straightforward:
  1. `show_auth_page()` renders a Login / Sign Up tab pair.
  2. On successful login, user data is stored in `st.session_state`.
  3. Every protected page calls `check_authentication()` at the top.
  4. `logout()` clears state and triggers a rerun so the login screen reappears.
"""

import streamlit as st

from auth.database import create_user, verify_user


# ========================== PUBLIC API ======================================

def show_auth_page() -> None:
    """Render the login / signup tabs.  Called only when the user is NOT yet
    authenticated (see app.py)."""

    login_tab, signup_tab = st.tabs(["🔐 Login", "📝 Sign Up"])

    with login_tab:
        _render_login_form()

    with signup_tab:
        _render_signup_form()


def check_authentication() -> bool:
    """Guard for every protected page.

    Returns True if the user is logged in; otherwise shows a warning and
    halts execution with `st.stop()`.
    """
    if st.session_state.get("authenticated", False):
        return True

    st.warning("⚠️ Please log in to access this page.")
    st.info("👈 Go to the **Home** page and log in first.")
    st.stop()
    return False  # unreachable, but keeps type-checkers happy


def logout() -> None:
    """Clear every session-state key and restart the app."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ========================== PRIVATE HELPERS =================================

def _render_login_form() -> None:
    """Username + password form with validation."""
    st.markdown("#### Welcome Back 👋")
    st.markdown("Enter your credentials to access the dashboard.")

    with st.form("login_form"):
        username = st.text_input("Username", placeholder="e.g. johndoe")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        submitted = st.form_submit_button("🔓 Login", use_container_width=True)

    if submitted:
        if not username or not password:
            st.error("Please fill in both fields.")
            return

        user = verify_user(username, password)
        if user:
            # Persist login across reruns
            st.session_state.authenticated = True
            st.session_state.user = user
            st.success(f"Welcome back, **{user['full_name']}**! 🎉")
            st.rerun()
        else:
            st.error("❌ Invalid username or password.")


def _render_signup_form() -> None:
    """Full registration form with client-side-style validations."""
    st.markdown("#### Create Your Account 🚀")
    st.markdown("Join LoanGuard AI to start predicting loan defaults.")

    with st.form("signup_form"):
        full_name = st.text_input("Full Name", placeholder="John Doe")
        email = st.text_input("Email", placeholder="john@example.com")
        username = st.text_input("Choose a Username", placeholder="johndoe")
        password = st.text_input("Password", type="password", placeholder="Min 8 characters")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
        submitted = st.form_submit_button("📝 Create Account", use_container_width=True)

    if submitted:
        # ---------- validation cascade ----------
        if not all([full_name, email, username, password, confirm_password]):
            st.error("All fields are required.")
            return

        if "@" not in email or "." not in email.split("@")[-1]:
            st.error("Please enter a valid email address.")
            return

        if len(password) < 8:
            st.error("Password must be at least 8 characters long.")
            return

        if password != confirm_password:
            st.error("Passwords do not match.")
            return

        # ---------- create user ----------
        success = create_user(username, email, password, full_name)
        if success:
            st.success("✅ Account created successfully!  Switch to the **Login** tab to sign in.")
            st.balloons()
        else:
            st.error("❌ Username or email already exists. Please choose a different one.")
