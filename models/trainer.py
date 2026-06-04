"""
trainer.py — Model training, evaluation, and persistence.

Two classifiers are offered:
  • Logistic Regression – fast, interpretable baseline.
  • XGBoost – gradient-boosted trees that usually win on tabular data.

Both are evaluated on the *original* (non-SMOTE) test set so metrics reflect
real-world class distribution.
"""

from __future__ import annotations

import os
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# Directory for persisted model files
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_DIR = os.path.join(_PROJECT_ROOT, "saved_models")


# ========================== TRAINING ========================================

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> LogisticRegression:
    """Train a balanced Logistic Regression classifier.

    `class_weight='balanced'` automatically up-weights the minority class
    in the loss function — complementary to SMOTE.
    """
    model = LogisticRegression(
        C=1.0,
        max_iter=1_000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> XGBClassifier:
    """Train an XGBoost classifier with tuned hyperparameters.

    `scale_pos_weight` is set to the class imbalance ratio so the gradient
    update gives more importance to the minority (default) class.
    """
    # Compute class ratio for scale_pos_weight
    n_negative = int(np.sum(y_train == 0))
    n_positive = int(np.sum(y_train == 1))
    scale_pos = n_negative / max(n_positive, 1)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ========================== EVALUATION ======================================

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Compute standard classification metrics.

    Returns a dict with scalar metrics, the confusion matrix, and the full
    sklearn classification report string.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, target_names=["No Default", "Default"]),
    }


# ========================== FEATURE IMPORTANCE ==============================

def get_feature_importance(
    model: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract and sort feature importances from the model.

    Works for both Logistic Regression (absolute coefficients) and
    tree-based models (built-in feature_importances_).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    # Ensure lengths match (pad names if necessary)
    n = len(importances)
    names = list(feature_names[:n]) + [f"feature_{i}" for i in range(len(feature_names), n)]

    df = pd.DataFrame({"feature": names[:n], "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


# ========================== PERSISTENCE =====================================

def save_model(model: Any, preprocessor: Any, model_name: str) -> str:
    """Persist model + preprocessor as a single joblib bundle.

    Returns the absolute path to the saved file.
    """
    os.makedirs(_MODEL_DIR, exist_ok=True)
    filepath = os.path.join(_MODEL_DIR, f"{model_name}.joblib")
    joblib.dump({"model": model, "preprocessor": preprocessor}, filepath)
    return filepath


def load_model(model_name: str) -> dict | None:
    """Load a previously saved model bundle.

    Returns
    -------
    dict with keys 'model' and 'preprocessor', or None if not found.
    """
    filepath = os.path.join(_MODEL_DIR, f"{model_name}.joblib")
    if os.path.exists(filepath):
        return joblib.load(filepath)
    return None
