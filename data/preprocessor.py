"""
preprocessor.py — Sklearn pipeline for feature engineering & transformation.

Pipeline overview
-----------------
  Numeric cols  → SimpleImputer(median) → StandardScaler
  Ordinal cols  → OrdinalEncoder (grade A-G maps to 0-6)
  Nominal cols  → SimpleImputer(most_frequent) → OneHotEncoder

Everything is wrapped in a ColumnTransformer so the fitted preprocessor can
be serialised alongside the model for one-step inference.

SMOTE is applied *after* the train/test split so the test set stays
representative of real-world class distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from imblearn.over_sampling import SMOTE


# ========================== COLUMN DEFINITIONS ==============================

# Explicitly listing columns avoids surprises if the data generator ever adds
# a new feature — we only transform what we expect.
NUMERIC_COLS = [
    "loan_amnt", "term", "int_rate", "emp_length", "annual_inc",
    "dti", "fico_score", "revol_bal", "revol_util",
    "delinq_2yrs", "pub_rec", "total_acc", "inq_last_6mths",
]

ORDINAL_COLS = ["grade"]                         # natural order A < B < … < G
NOMINAL_COLS = ["home_ownership", "purpose"]     # no order — one-hot encode

# Columns dropped before training (sub_grade is redundant given grade)
DROP_COLS = ["sub_grade"]


# ========================== PUBLIC API ======================================

@st.cache_resource(show_spinner="Building preprocessing pipeline …")
def preprocess_data(
    df: pd.DataFrame,
    target_col: str = "loan_status",
    test_size: float = 0.20,
    random_state: int = 42,
):
    """Full preprocessing + SMOTE pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Transformed arrays ready for model training / evaluation.
    preprocessor : ColumnTransformer
        Fitted transformer (save this alongside the model).
    feature_names : list[str]
        Human-readable names after transformation.
    """
    # ---- 1. Drop redundant columns & separate target ----
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    # ---- 2. Build sklearn ColumnTransformer ----
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=[list("ABCDEFG")])),
    ])

    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLS),
            ("ord", ordinal_pipeline, ORDINAL_COLS),
            ("nom", nominal_pipeline, NOMINAL_COLS),
        ],
        remainder="drop",  # any unlisted columns are silently ignored
    )

    # ---- 3. Train / test split (stratified to preserve class ratio) ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    # ---- 4. Fit on train, transform both ----
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # ---- 5. SMOTE on training data only ----
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_transformed, y_train,
    )

    # ---- 6. Derive feature names for interpretability ----
    feature_names = _get_feature_names(preprocessor)

    return (
        X_train_resampled,
        X_test_transformed,
        y_train_resampled,
        y_test,
        preprocessor,
        feature_names,
    )


def preprocess_single_input(input_dict: dict, preprocessor: ColumnTransformer) -> np.ndarray:
    """Transform a single user-submitted record for prediction.

    Parameters
    ----------
    input_dict : dict
        Raw feature values exactly as entered in the Streamlit form.
    preprocessor : ColumnTransformer
        The *fitted* transformer saved during training.

    Returns
    -------
    np.ndarray of shape (1, n_features)
    """
    # Build a one-row DataFrame so column names align with the pipeline
    single_df = pd.DataFrame([input_dict])

    # Ensure columns exist (fill missing optional cols with NaN)
    for col in NUMERIC_COLS + ORDINAL_COLS + NOMINAL_COLS:
        if col not in single_df.columns:
            single_df[col] = np.nan

    return preprocessor.transform(single_df)


# ========================== PRIVATE HELPERS =================================

def _get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract human-readable feature names after transformation.

    Sklearn's get_feature_names_out() can be fragile across versions, so we
    build the list manually as a fallback.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        names: list[str] = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == "remainder":
                continue
            if hasattr(transformer, "get_feature_names_out"):
                names.extend(transformer.get_feature_names_out())
            else:
                names.extend(columns if isinstance(columns, list) else [columns])
        return names
