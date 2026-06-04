"""
Page 2 — Model Training

Trains Logistic Regression and XGBoost side-by-side, compares metrics,
and lets the user save the best model for use on the Predict page.
"""

import time
import os
import streamlit as st
import pandas as pd

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

# ── Imports ─────────────────────────────────────────────────────────────────
from data.generate_data import get_data
from data.preprocessor import preprocess_data
from models.trainer import (
    train_logistic_regression,
    train_xgboost,
    evaluate_model,
    get_feature_importance,
    save_model,
)
from utils.visualization import (
    plot_roc_curve,
    plot_confusion_matrix,
    plot_feature_importance,
)

# ════════════════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<h1 class="gradient-text">🤖 Model Training</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    "Train and compare **Logistic Regression** and **XGBoost** classifiers "
    "on the preprocessed loan dataset."
)

# ════════════════════════════════════════════════════════════════════════════
#  PIPELINE INFO
# ════════════════════════════════════════════════════════════════════════════
with st.expander("ℹ️ Preprocessing pipeline details", expanded=False):
    st.markdown(
        """
        | Step | Details |
        |------|---------|
        | Missing values | Median (numeric), most-frequent (categorical) |
        | Encoding | OrdinalEncoder (grade A→0 … G→6), OneHotEncoder (home_ownership, purpose) |
        | Scaling | StandardScaler on all numeric features |
        | Resampling | SMOTE on **training set only** to balance classes |
        | Split | 80/20 stratified train/test |
        """
    )

# ════════════════════════════════════════════════════════════════════════════
#  TRAIN BUTTON
# ════════════════════════════════════════════════════════════════════════════

if st.button("🚀 Train Models", use_container_width=True, type="primary"):

    # ── 1. Load & preprocess ──
    progress = st.progress(0, text="Loading and preprocessing data …")
    df = get_data()
    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(df)
    progress.progress(20, text="Data preprocessed ✅")

    # ── 2. Logistic Regression ──
    progress.progress(25, text="Training Logistic Regression …")
    t0 = time.time()
    lr_model = train_logistic_regression(X_train, y_train)
    lr_time = time.time() - t0
    progress.progress(50, text=f"Logistic Regression done in {lr_time:.2f}s ✅")

    # ── 3. XGBoost ──
    progress.progress(55, text="Training XGBoost …")
    t0 = time.time()
    xgb_model = train_xgboost(X_train, y_train)
    xgb_time = time.time() - t0
    progress.progress(80, text=f"XGBoost done in {xgb_time:.2f}s ✅")

    # ── 4. Evaluate ──
    progress.progress(85, text="Evaluating models …")
    lr_metrics = evaluate_model(lr_model, X_test, y_test)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test)
    progress.progress(100, text="All done! 🎉")

    # ── Persist to session state ──
    st.session_state.lr_model = lr_model
    st.session_state.xgb_model = xgb_model
    st.session_state.preprocessor = preprocessor
    st.session_state.feature_names = feature_names
    st.session_state.lr_metrics = lr_metrics
    st.session_state.xgb_metrics = xgb_metrics
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.models_trained = True

    st.success("✅ Both models trained and evaluated successfully!")


# ════════════════════════════════════════════════════════════════════════════
#  RESULTS  (show only if models have been trained this session)
# ════════════════════════════════════════════════════════════════════════════

if st.session_state.get("models_trained", False):
    lr_metrics = st.session_state.lr_metrics
    xgb_metrics = st.session_state.xgb_metrics

    # ── Metrics comparison table ──
    st.markdown("---")
    st.markdown("### 📊 Model Comparison")

    comparison_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
        "Logistic Regression": [
            lr_metrics["accuracy"],
            lr_metrics["precision"],
            lr_metrics["recall"],
            lr_metrics["f1"],
            lr_metrics["roc_auc"],
        ],
        "XGBoost": [
            xgb_metrics["accuracy"],
            xgb_metrics["precision"],
            xgb_metrics["recall"],
            xgb_metrics["f1"],
            xgb_metrics["roc_auc"],
        ],
    })

    # Highlight the winner for each metric
    def highlight_best(row):
        """Colour the better score green."""
        lr_val, xgb_val = row["Logistic Regression"], row["XGBoost"]
        styles = [""] * 3
        if lr_val > xgb_val:
            styles[1] = "background-color: rgba(0,210,255,0.15); font-weight:700;"
        elif xgb_val > lr_val:
            styles[2] = "background-color: rgba(0,210,255,0.15); font-weight:700;"
        return styles

    st.dataframe(
        comparison_df.style.apply(highlight_best, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # ── ROC Curves ──
    st.markdown("---")
    st.markdown("### 📈 ROC Curve")

    models_dict = {
        "Logistic Regression": st.session_state.lr_model,
        "XGBoost": st.session_state.xgb_model,
    }
    fig_roc = plot_roc_curve(models_dict, st.session_state.X_test, st.session_state.y_test)
    st.plotly_chart(fig_roc, use_container_width=True)

    # ── Confusion Matrices ──
    st.markdown("---")
    st.markdown("### 🔢 Confusion Matrices")

    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        fig_cm_lr = plot_confusion_matrix(lr_metrics["confusion_matrix"], "Logistic Regression")
        st.pyplot(fig_cm_lr)
    with cm_col2:
        fig_cm_xgb = plot_confusion_matrix(xgb_metrics["confusion_matrix"], "XGBoost")
        st.pyplot(fig_cm_xgb)

    # ── Classification Reports ──
    with st.expander("📄 Full Classification Reports"):
        rep_col1, rep_col2 = st.columns(2)
        with rep_col1:
            st.markdown("**Logistic Regression**")
            st.code(lr_metrics["classification_report"])
        with rep_col2:
            st.markdown("**XGBoost**")
            st.code(xgb_metrics["classification_report"])

    # ── Feature Importance (XGBoost) ──
    st.markdown("---")
    st.markdown("### 🏆 Feature Importance (XGBoost)")

    importance_df = get_feature_importance(
        st.session_state.xgb_model,
        st.session_state.feature_names,
    )
    fig_imp = plot_feature_importance(importance_df, top_n=15)
    st.plotly_chart(fig_imp, use_container_width=True)

    # ── Save Best Model ──
    st.markdown("---")
    st.markdown("### 💾 Save Model")

    best_name = "XGBoost" if xgb_metrics["roc_auc"] >= lr_metrics["roc_auc"] else "Logistic Regression"
    best_model = st.session_state.xgb_model if best_name == "XGBoost" else st.session_state.lr_model

    st.info(f"🏅 **{best_name}** has the highest ROC AUC ({max(xgb_metrics['roc_auc'], lr_metrics['roc_auc']):.4f}).")

    if st.button("💾 Save Best Model", use_container_width=True):
        filepath = save_model(best_model, st.session_state.preprocessor, "best_model")
        st.success(f"Model saved to `{filepath}`")

else:
    st.markdown(
        """
        <div class="glass-card" style="text-align:center; padding:3rem;">
            <div style="font-size:3rem;">🤖</div>
            <h3 style="color:#1a202c;">No Models Trained Yet</h3>
            <p style="color:#64748b;">
                Click the <b>Train Models</b> button above to train
                Logistic Regression and XGBoost classifiers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
