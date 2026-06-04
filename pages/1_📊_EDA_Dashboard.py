"""
Page 1 — EDA Dashboard

An interactive exploratory data analysis page that lets the user slice and
dice the synthetic loan dataset through interactive Plotly / Seaborn charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

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
from data.generate_data import get_data
from utils.visualization import (
    plot_distribution,
    plot_target_distribution,
    plot_correlation_heatmap,
    plot_feature_vs_target,
)

df = get_data()

# ════════════════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<h1 class="gradient-text">📊 EDA Dashboard</h1>',
    unsafe_allow_html=True,
)
st.markdown("Explore the synthetic loan dataset — understand distributions, correlations, and default patterns.")

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Dataset Overview
# ════════════════════════════════════════════════════════════════════════════
st.markdown("### 📋 Dataset Overview")

col_m1, col_m2, col_m3, col_m4 = st.columns(4, gap="medium")

with col_m1:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Total Records</div>
            <div class="metric-value">{df.shape[0]:,}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with col_m2:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Features</div>
            <div class="metric-value">{df.shape[1] - 1}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with col_m3:
    default_rate = df["loan_status"].mean() * 100
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Default Rate</div>
            <div class="metric-value">{default_rate:.1f}%</div>
        </div>""",
        unsafe_allow_html=True,
    )
with col_m4:
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Missing Data</div>
            <div class="metric-value">{missing_pct:.2f}%</div>
        </div>""",
        unsafe_allow_html=True,
    )

with st.expander("🔎 Preview raw data & column info", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    st.markdown("**Column types**")
    dtypes_df = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str).values,
        "Non-Null": df.notnull().sum().values,
        "Missing": df.isnull().sum().values,
        "Missing %": (df.isnull().mean() * 100).round(2).values,
    })
    st.dataframe(dtypes_df, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Target Distribution
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 🎯 Target Distribution")

col_pie, col_desc = st.columns([1, 1])
with col_pie:
    st.plotly_chart(plot_target_distribution(df), use_container_width=True)

with col_desc:
    st.markdown(
        """
        <div class="glass-card" style="margin-top:1rem;">
            <h4 style="color:#e6edf3;">Insight</h4>
            <p style="color:#8b949e; line-height:1.8;">
                The dataset has a ~<b>20 % default rate</b>, which is typical for
                sub-prime lending portfolios.  This class imbalance is why we
                apply <b>SMOTE</b> during training and use <b>balanced class
                weights</b> in our models.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

vc = df["loan_status"].value_counts()
st.markdown(f"- **No Default (0):** {vc.get(0, 0):,} loans ({vc.get(0, 0) / len(df) * 100:.1f}%)")
st.markdown(f"- **Default (1):** {vc.get(1, 0):,} loans ({vc.get(1, 0) / len(df) * 100:.1f}%)")

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Numeric Distributions
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📈 Feature Distributions")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "loan_status"]

selected_num = st.selectbox("Select a numeric feature:", numeric_cols, index=numeric_cols.index("fico_score") if "fico_score" in numeric_cols else 0)

st.plotly_chart(
    plot_distribution(df, selected_num, f"Distribution of {selected_num}"),
    use_container_width=True,
)

# Summary stats for the selected column
st.markdown(f"**Summary statistics for `{selected_num}`:**")
stats = df[selected_num].describe().round(2)
stat_cols = st.columns(6)
for i, (stat_name, stat_val) in enumerate(stats.items()):
    with stat_cols[i % 6]:
        st.metric(stat_name, f"{stat_val:,.2f}")

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Correlation Heatmap
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 🔥 Correlation Heatmap")

st.markdown(
    "_Select which numeric features to include in the correlation analysis._"
)
default_corr = ["int_rate", "fico_score", "dti", "annual_inc", "delinq_2yrs", "revol_util", "loan_status"]
corr_cols = st.multiselect(
    "Columns:", numeric_cols + ["loan_status"],
    default=[c for c in default_corr if c in numeric_cols + ["loan_status"]],
)

if len(corr_cols) >= 2:
    fig_corr = plot_correlation_heatmap(df[corr_cols])
    st.pyplot(fig_corr)
else:
    st.warning("Select at least 2 columns to show a correlation heatmap.")

st.markdown(
    """
    <div class="glass-card">
        <h4 style="color:#e6edf3;">Key Correlations</h4>
        <ul style="color:#8b949e; line-height:1.9;">
            <li><code>int_rate</code> ↔ <code>loan_status</code>: Higher interest rates strongly
                correlate with defaults — riskier borrowers are charged more.</li>
            <li><code>fico_score</code> ↔ <code>loan_status</code>: Lower FICO = higher default risk.</li>
            <li><code>dti</code> ↔ <code>loan_status</code>: Debt-to-income is a moderate positive predictor.</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — Feature vs Target
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📊 Feature vs Target Analysis")

feature_for_box = st.selectbox(
    "Select a feature to compare by loan status:",
    numeric_cols,
    index=numeric_cols.index("int_rate") if "int_rate" in numeric_cols else 0,
    key="feature_vs_target",
)

st.plotly_chart(
    plot_feature_vs_target(df, feature_for_box),
    use_container_width=True,
)

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — Categorical Analysis
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📌 Categorical Feature Analysis")

cat_tab_grade, cat_tab_purpose, cat_tab_home = st.tabs(["By Grade", "By Purpose", "By Home Ownership"])

import plotly.express as px

_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

with cat_tab_grade:
    grade_default = df.groupby("grade")["loan_status"].mean().reset_index()
    grade_default.columns = ["grade", "default_rate"]
    grade_default["default_rate"] = (grade_default["default_rate"] * 100).round(1)
    fig_g = px.bar(
        grade_default, x="grade", y="default_rate",
        title="Default Rate by Grade",
        color="default_rate",
        color_continuous_scale=["#00d2ff", "#f7971e", "#ff416c"],
        text="default_rate",
    )
    fig_g.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_g.update_layout(**_LAYOUT)
    fig_g.update_coloraxes(showscale=False)
    st.plotly_chart(fig_g, use_container_width=True)
    st.markdown("_Higher grades (F, G) carry significantly more default risk — as expected._")

with cat_tab_purpose:
    purp_default = df.groupby("purpose")["loan_status"].mean().reset_index()
    purp_default.columns = ["purpose", "default_rate"]
    purp_default["default_rate"] = (purp_default["default_rate"] * 100).round(1)
    purp_default = purp_default.sort_values("default_rate", ascending=True)
    fig_p = px.bar(
        purp_default, x="default_rate", y="purpose", orientation="h",
        title="Default Rate by Loan Purpose",
        color="default_rate",
        color_continuous_scale=["#00d2ff", "#f7971e", "#ff416c"],
        text="default_rate",
    )
    fig_p.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_p.update_layout(**_LAYOUT)
    fig_p.update_coloraxes(showscale=False)
    st.plotly_chart(fig_p, use_container_width=True)
    st.markdown("_Small business loans tend to have higher default rates._")

with cat_tab_home:
    home_default = df.groupby("home_ownership")["loan_status"].mean().reset_index()
    home_default.columns = ["home_ownership", "default_rate"]
    home_default["default_rate"] = (home_default["default_rate"] * 100).round(1)
    fig_h = px.bar(
        home_default, x="home_ownership", y="default_rate",
        title="Default Rate by Home Ownership",
        color="default_rate",
        color_continuous_scale=["#00d2ff", "#f7971e", "#ff416c"],
        text="default_rate",
    )
    fig_h.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_h.update_layout(**_LAYOUT)
    fig_h.update_coloraxes(showscale=False)
    st.plotly_chart(fig_h, use_container_width=True)
    st.markdown("_Home ownership type shows only moderate variation in default rates._")
