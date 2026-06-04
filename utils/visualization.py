"""
visualization.py — Reusable Plotly / Seaborn chart functions.

All Plotly figures use `plotly_white` template with a consistent colour palette.
Matplotlib/Seaborn charts are styled with a light background for visual consistency.

Colour palette
--------------
   primary   = #4f46e5  (indigo)
   secondary = #7c3aed  (purple)
   success   = #059669  (green)
   danger    = #dc2626  (red)
   warning   = #d97706  (amber)
"""

from __future__ import annotations

from typing import Any

import matplotlib
matplotlib.use("Agg")                 # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import roc_curve, auc


# ========================== COLOUR PALETTE ==================================

PRIMARY   = "#4f46e5"
SECONDARY = "#7c3aed"
SUCCESS   = "#059669"
DANGER    = "#dc2626"
WARNING   = "#d97706"

PALETTE = [PRIMARY, SECONDARY, SUCCESS, DANGER, WARNING, "#0ea5e9", "#ec4899"]

# Plotly layout defaults shared across all figures
_LAYOUT_DEFAULTS = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif"),
    margin=dict(l=40, r=40, t=50, b=40),
)


# ========================== DISTRIBUTION ====================================

def plot_distribution(df: pd.DataFrame, column: str, title: str | None = None) -> go.Figure:
    """Plotly histogram for a single numeric column."""
    fig = px.histogram(
        df, x=column, nbins=50,
        title=title or f"Distribution of {column}",
        color_discrete_sequence=[PRIMARY],
    )
    fig.update_layout(**_LAYOUT_DEFAULTS)
    return fig


def plot_target_distribution(df: pd.DataFrame, target_col: str = "loan_status") -> go.Figure:
    """Pie chart showing class balance (default vs. no-default)."""
    counts = df[target_col].value_counts().reset_index()
    counts.columns = [target_col, "count"]
    labels_map = {0: "No Default", 1: "Default"}
    counts[target_col] = counts[target_col].map(labels_map)

    fig = px.pie(
        counts, names=target_col, values="count",
        title="Loan Status Distribution",
        color_discrete_sequence=[SUCCESS, DANGER],
        hole=0.45,
    )
    fig.update_layout(**_LAYOUT_DEFAULTS)
    fig.update_traces(textinfo="percent+label", textfont_size=14)
    return fig


# ========================== CORRELATIONS ====================================

def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Seaborn heatmap of Pearson correlations for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8, "color": "#1a202c"},
    )
    ax.tick_params(colors="#1a202c")
    ax.set_title("Feature Correlation Matrix", color="#1a202c", fontsize=16, pad=15)
    plt.tight_layout()
    return fig


# ========================== FEATURE VS TARGET ===============================

def plot_feature_vs_target(
    df: pd.DataFrame, feature: str, target: str = "loan_status",
) -> go.Figure:
    """Box plot of a numeric feature split by the target class."""
    plot_df = df.copy()
    plot_df[target] = plot_df[target].map({0: "No Default", 1: "Default"})

    fig = px.box(
        plot_df, x=target, y=feature, color=target,
        title=f"{feature} by Loan Status",
        color_discrete_sequence=[SUCCESS, DANGER],
    )
    fig.update_layout(**_LAYOUT_DEFAULTS)
    return fig


# ========================== MODEL METRICS ===================================

def plot_roc_curve(models_dict: dict[str, Any], X_test, y_test) -> go.Figure:
    """Overlay ROC curves for multiple models on one chart."""
    fig = go.Figure()
    colours = [PRIMARY, DANGER, SUCCESS, WARNING]

    for idx, (name, model) in enumerate(models_dict.items()):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        colour = colours[idx % len(colours)]

        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC = {roc_auc:.3f})",
            line=dict(color=colour, width=2.5),
        ))

    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random Classifier",
        line=dict(color="gray", width=1, dash="dash"),
    ))

    fig.update_layout(
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        **_LAYOUT_DEFAULTS,
    )
    return fig


def plot_confusion_matrix(cm: np.ndarray, model_name: str) -> plt.Figure:
    """Annotated heatmap of the confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"],
        ax=ax, linewidths=1, linecolor="#e2e8f0",
        annot_kws={"size": 16, "weight": "bold"},
    )
    ax.set_xlabel("Predicted", color="#1a202c", fontsize=12)
    ax.set_ylabel("Actual", color="#1a202c", fontsize=12)
    ax.set_title(f"{model_name} — Confusion Matrix", color="#1a202c", fontsize=14, pad=12)
    ax.tick_params(colors="#1a202c")
    plt.tight_layout()
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of the top-N most important features."""
    top = importance_df.head(top_n).iloc[::-1]  # reverse for horizontal layout

    fig = px.bar(
        top, x="importance", y="feature", orientation="h",
        title=f"Top {top_n} Feature Importances",
        color="importance",
        color_continuous_scale=["#7c3aed", "#4f46e5", "#0ea5e9"],
    )
    fig.update_layout(**_LAYOUT_DEFAULTS)
    fig.update_coloraxes(showscale=False)
    return fig


# ========================== GAUGE / PREDICTION ==============================

def plot_probability_gauge(probability: float) -> go.Figure:
    """Speedometer-style gauge for the default probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 48, "color": "#1a202c"}},
        title={"text": "Default Probability", "font": {"size": 20, "color": "#64748b"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 2, "tickcolor": "#555"},
            "bar": {"color": SECONDARY},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 2,
            "bordercolor": "#333",
            "steps": [
                {"range": [0, 30], "color": "rgba(5,150,105,0.15)"},
                {"range": [30, 60], "color": "rgba(217,119,6,0.15)"},
                {"range": [60, 100], "color": "rgba(220,38,38,0.15)"},
            ],
            "threshold": {
                "line": {"color": DANGER, "width": 4},
                "thickness": 0.8,
                "value": probability * 100,
            },
        },
    ))
    fig.update_layout(
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    return fig
