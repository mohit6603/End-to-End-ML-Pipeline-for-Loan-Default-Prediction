"""
predictor.py — Inference helpers for the prediction page.

Wraps preprocessing → model.predict_proba → risk-level mapping in a single
function so the Streamlit page stays clean.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from data.preprocessor import preprocess_single_input


# ========================== RISK THRESHOLDS =================================

# These thresholds are intentionally simple — in production you'd calibrate
# them against business cost functions (cost of a missed default vs. cost of
# rejecting a good borrower).

_RISK_LEVELS = [
    (0.30, "Low Risk"),
    (0.60, "Medium Risk"),
    (1.01, "High Risk"),   # 1.01 so probability == 1.0 is captured
]


# ========================== PUBLIC API ======================================

def predict_loan_default(
    model: Any,
    preprocessor: Any,
    input_data: dict,
) -> tuple[int, float, str]:
    """Run end-to-end inference on raw user inputs.

    Parameters
    ----------
    model : fitted sklearn / xgboost classifier
    preprocessor : fitted ColumnTransformer
    input_data : dict
        Raw feature values from the Streamlit form.

    Returns
    -------
    prediction : int  (0 = no default, 1 = default)
    probability : float  (P(default))
    risk_level : str  ('Low Risk', 'Medium Risk', or 'High Risk')
    """
    # Transform raw inputs to the format the model expects
    X = preprocess_single_input(input_data, preprocessor)

    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])  # P(class=1)
    risk_level = _classify_risk(probability)

    return prediction, probability, risk_level


def get_prediction_color(risk_level: str) -> str:
    """Map risk level to a hex colour for UI badges / gauges."""
    return {
        "Low Risk": "#00d2ff",      # cyan / safe
        "Medium Risk": "#f7971e",   # amber / caution
        "High Risk": "#ff416c",     # red / danger
    }.get(risk_level, "#888888")


# ========================== PRIVATE =========================================

def _classify_risk(probability: float) -> str:
    """Bucket a default probability into a human-readable risk label."""
    for threshold, label in _RISK_LEVELS:
        if probability < threshold:
            return label
    return "High Risk"
