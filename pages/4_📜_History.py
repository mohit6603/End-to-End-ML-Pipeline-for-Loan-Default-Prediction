"""
Page 4 — Prediction History

Displays every prediction the current user has made, with summary metrics
and a CSV download button.
"""

import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

# ── Auth guard ──────────────────────────────────────────────────────────────
from auth.auth_handler import check_authentication
check_authentication()

# ── Load custom CSS ─────────────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 👋 Welcome, {st.session_state.user['full_name']}")
    st.divider()
    if st.button("🚪 Logout", use_container_width=True):
        from auth.auth_handler import logout
        logout()

# ── Data ────────────────────────────────────────────────────────────────────
from auth.database import get_user_predictions

# ════════════════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<h1 class="gradient-text">📜 Prediction History</h1>',
    unsafe_allow_html=True,
)
st.markdown("Review all of your past loan default predictions.")


# ════════════════════════════════════════════════════════════════════════════
#  FETCH & DISPLAY
# ════════════════════════════════════════════════════════════════════════════

predictions = get_user_predictions(st.session_state.user["id"])

if not predictions:
    st.markdown(
        """
        <div class="glass-card" style="text-align:center; padding:3rem;">
            <div style="font-size:3rem;">📭</div>
            <h3 style="color:#e6edf3;">No Predictions Yet</h3>
            <p style="color:#8b949e;">
                Head over to the <b>🔮 Predict</b> page to make your first
                loan default prediction.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ── Build a clean DataFrame ────────────────────────────────────────────────
history_rows = []
for p in predictions:
    row = {
        "Date": p["created_at"],
        "Prediction": "Default" if p["prediction"] == 1 else "No Default",
        "Probability": f"{p['probability'] * 100:.1f}%",
        "Risk Level": p["risk_level"],
    }
    # Flatten selected input fields for the table
    try:
        inputs = json.loads(p["input_data"])
        row["Loan Amount"] = f"${inputs.get('loan_amnt', '-'):,}"
        row["FICO"] = inputs.get("fico_score", "-")
        row["DTI"] = inputs.get("dti", "-")
        row["Grade"] = inputs.get("grade", "-")
    except (json.JSONDecodeError, TypeError):
        pass
    history_rows.append(row)

history_df = pd.DataFrame(history_rows)


# ── Summary Metrics ────────────────────────────────────────────────────────
st.markdown("### 📊 Summary")

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Total Predictions</div>
            <div class="metric-value">{len(predictions)}</div>
        </div>""",
        unsafe_allow_html=True,
    )

with col2:
    avg_prob = sum(p["probability"] for p in predictions) / len(predictions)
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Avg. Default Probability</div>
            <div class="metric-value">{avg_prob * 100:.1f}%</div>
        </div>""",
        unsafe_allow_html=True,
    )

with col3:
    high_risk_count = sum(1 for p in predictions if p["risk_level"] == "High Risk")
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">High Risk Predictions</div>
            <div class="metric-value">{high_risk_count}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ── Risk Distribution Pie Chart ────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🎯 Risk Distribution")

risk_counts = pd.Series([p["risk_level"] for p in predictions]).value_counts().reset_index()
risk_counts.columns = ["Risk Level", "Count"]

color_map = {"Low Risk": "#00d2ff", "Medium Risk": "#f7971e", "High Risk": "#ff416c"}
fig_risk = px.pie(
    risk_counts,
    names="Risk Level",
    values="Count",
    color="Risk Level",
    color_discrete_map=color_map,
    hole=0.45,
    title="Risk Level Distribution",
)
fig_risk.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
fig_risk.update_traces(textinfo="percent+label")
st.plotly_chart(fig_risk, use_container_width=True)


# ── Full History Table ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📋 Prediction Details")

st.dataframe(history_df, use_container_width=True, hide_index=True)


# ── CSV Download ───────────────────────────────────────────────────────────
csv_data = history_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download as CSV",
    data=csv_data,
    file_name="loanguard_prediction_history.csv",
    mime="text/csv",
    use_container_width=True,
)
