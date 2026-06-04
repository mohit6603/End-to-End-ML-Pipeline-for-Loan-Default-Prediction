"""
app.py — LoanGuard AI main entry point.

Run with:  streamlit run app.py

This file handles:
  1. Page configuration (must be the very first Streamlit call).
  2. Loading the custom dark-theme CSS.
  3. Initialising the SQLite database.
  4. Session-state setup.
  5. Routing between the auth flow and the authenticated home page.
"""

import os
import streamlit as st

# ── Page config MUST be the first Streamlit command ─────────────────────────
st.set_page_config(
    page_title="LoanGuard AI — Loan Default Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Load custom CSS ────────────────────────────────────────────────────────
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# ── Database bootstrap ─────────────────────────────────────────────────────
from auth.database import init_db
init_db()


# ── Session-state defaults ─────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = None


# ══════════════════════════════════════════════════════════════════════════
#  UNAUTHENTICATED  → show login / signup
# ══════════════════════════════════════════════════════════════════════════
if not st.session_state.authenticated:
    from auth.auth_handler import show_auth_page

    st.markdown(
        """
        <div style="text-align:center; padding: 2rem 0 1rem 0;">
            <h1 class="gradient-text" style="font-size:3.2rem; margin-bottom:0.2rem;">
                🏦 LoanGuard AI
            </h1>
            <p style="color:#8b949e; font-size:1.25rem; margin-top:0;">
                Intelligent Loan Default Prediction System
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Two-column layout: left = value props, right = auth form
    col_info, col_form = st.columns([1, 1], gap="large")

    with col_info:
        st.markdown(
            """
            <div class="glass-card" style="margin-top:1rem;">
                <h3 style="color:#e6edf3;">Why LoanGuard AI?</h3>
                <ul style="color:#8b949e; line-height:2;">
                    <li>🔬 <b>ML-Powered</b> — XGBoost + Logistic Regression</li>
                    <li>📊 <b>Interactive EDA</b> — explore 10 000+ loan records</li>
                    <li>⚡ <b>Real-Time Predictions</b> — instant risk scoring</li>
                    <li>📜 <b>Audit Trail</b> — full prediction history</li>
                    <li>🔒 <b>Secure</b> — bcrypt auth + per-user data</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_form:
        show_auth_page()


# ══════════════════════════════════════════════════════════════════════════
#  AUTHENTICATED  → sidebar + home page
# ══════════════════════════════════════════════════════════════════════════
else:
    from auth.auth_handler import logout

    # ── Sidebar ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f"### 👋 Welcome, {st.session_state.user['full_name']}"
        )
        st.markdown(f"📧 {st.session_state.user['email']}")
        st.divider()
        st.markdown("#### 🗂️ Navigation")
        st.page_link("app.py", label="🏠 Home", icon="🏠")
        st.page_link("pages/1_📊_EDA_Dashboard.py", label="📊 EDA Dashboard")
        st.page_link("pages/2_🤖_Model_Training.py", label="🤖 Model Training")
        st.page_link("pages/3_🔮_Predict.py", label="🔮 Predict")
        st.page_link("pages/4_📜_History.py", label="📜 History")
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            logout()

    # ── Home content ───────────────────────────────────────────────────
    st.markdown(
        """
        <h1 class="gradient-text" style="font-size:2.6rem;">🏦 LoanGuard AI</h1>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Welcome to the Intelligent Loan Default Prediction System")

    # Feature cards
    col1, col2, col3, col4 = st.columns(4, gap="medium")

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div style="font-size:2.5rem;">📊</div>
                <div class="metric-value" style="font-size:1.2rem;">EDA Dashboard</div>
                <div class="metric-label">Explore the data</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div style="font-size:2.5rem;">🤖</div>
                <div class="metric-value" style="font-size:1.2rem;">Model Training</div>
                <div class="metric-label">Train & evaluate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div style="font-size:2.5rem;">🔮</div>
                <div class="metric-value" style="font-size:1.2rem;">Predict</div>
                <div class="metric-label">Real-time scoring</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div style="font-size:2.5rem;">📜</div>
                <div class="metric-value" style="font-size:1.2rem;">History</div>
                <div class="metric-label">Audit trail</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.info("👈 Use the **sidebar** to navigate between pages.")

    # Quick overview section
    st.markdown("---")
    st.markdown("### 🔍 How It Works")

    step1, step2, step3 = st.columns(3, gap="medium")
    with step1:
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color:#667eea;">Step 1 — Explore</h4>
                <p style="color:#8b949e;">
                    Dive into 10,000+ synthetic loan records. Understand distributions,
                    correlations, and default patterns through interactive charts.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with step2:
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color:#764ba2;">Step 2 — Train</h4>
                <p style="color:#8b949e;">
                    Train Logistic Regression and XGBoost models with SMOTE resampling.
                    Compare ROC curves, confusion matrices, and feature importances.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with step3:
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color:#00d2ff;">Step 3 — Predict</h4>
                <p style="color:#8b949e;">
                    Enter borrower details and receive instant default probability
                    with a risk level badge. Every prediction is saved for audit.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
