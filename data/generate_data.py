"""
generate_data.py — Synthetic Lending-Club-style dataset generator.

Why synthetic data?
  • No licensing / privacy issues — safe for a portfolio project.
  • We control the signal-to-noise ratio and default rate (~20 %).
  • The feature distributions mimic real Lending Club files so interviewers
    recognise the domain instantly.

The target `loan_status` is generated via a logistic function of the most
influential features (int_rate, fico_score, dti, annual_inc, delinq_2yrs).
"""

import numpy as np
import pandas as pd
import streamlit as st


# ========================== PUBLIC API ======================================

@st.cache_data(show_spinner="Generating synthetic loan data …")
def generate_loan_data(n_samples: int = 10_000, random_state: int = 42) -> pd.DataFrame:
    """Create a realistic synthetic loan dataset.

    Parameters
    ----------
    n_samples : int
        Number of loan records to generate.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with 18 feature columns + ``loan_status`` target.
    """
    rng = np.random.RandomState(random_state)

    # ------------------------------------------------------------------ #
    #  1. Core numeric features                                           #
    # ------------------------------------------------------------------ #
    loan_amnt = np.clip(rng.lognormal(mean=9.5, sigma=0.7, size=n_samples), 1_000, 40_000).astype(int)
    term = rng.choice([36, 60], size=n_samples, p=[0.70, 0.30])

    # Grade drives interest rate in real life
    grades = list("ABCDEFG")
    grade_weights = [0.20, 0.22, 0.20, 0.15, 0.10, 0.08, 0.05]
    grade = rng.choice(grades, size=n_samples, p=grade_weights)

    # Sub-grade: append a number 1-5
    sub_grade = np.array([f"{g}{rng.randint(1, 6)}" for g in grade])

    # Interest rate ~ grade + noise
    _grade_base_rate = {"A": 6.5, "B": 9.5, "C": 12.5, "D": 16.0, "E": 20.0, "F": 24.0, "G": 27.0}
    int_rate = np.array([_grade_base_rate[g] + rng.normal(0, 1.2) for g in grade])
    int_rate = np.clip(int_rate, 5.0, 30.0).round(2)

    # Employment length (0 = < 1 year, 10 = 10+ years)
    emp_length = rng.choice(range(0, 11), size=n_samples).astype(float)
    # Inject ~4 % NaN to simulate missing data
    emp_length[rng.rand(n_samples) < 0.04] = np.nan

    annual_inc = np.clip(rng.lognormal(mean=11.0, sigma=0.5, size=n_samples), 20_000, 200_000).astype(int)
    dti = np.clip(rng.uniform(0, 40, size=n_samples) + rng.normal(0, 2, size=n_samples), 0, 50).round(2)
    fico_score = np.clip(rng.normal(700, 50, size=n_samples), 300, 850).astype(int)

    # ------------------------------------------------------------------ #
    #  2. Categorical features                                            #
    # ------------------------------------------------------------------ #
    home_ownership = rng.choice(
        ["RENT", "OWN", "MORTGAGE"],
        size=n_samples,
        p=[0.40, 0.15, 0.45],
    )
    purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement",
         "major_purchase", "small_business", "other"],
        size=n_samples,
        p=[0.40, 0.20, 0.12, 0.10, 0.08, 0.10],
    )

    # ------------------------------------------------------------------ #
    #  3. Credit-history features                                         #
    # ------------------------------------------------------------------ #
    revol_bal = (annual_inc * rng.uniform(0.05, 0.5, size=n_samples)).astype(int)
    revol_util = np.clip(rng.beta(2, 5, size=n_samples) * 100, 0, 100).round(1)
    # ~3 % NaN in revol_util
    revol_util_arr = revol_util.copy().astype(float)
    revol_util_arr[rng.rand(n_samples) < 0.03] = np.nan

    delinq_2yrs = rng.choice(range(0, 6), size=n_samples, p=[0.70, 0.15, 0.08, 0.04, 0.02, 0.01])
    pub_rec = rng.choice(range(0, 4), size=n_samples, p=[0.85, 0.10, 0.03, 0.02])
    total_acc = rng.randint(5, 51, size=n_samples)
    inq_last_6mths = rng.choice(range(0, 9), size=n_samples, p=[0.35, 0.25, 0.15, 0.10, 0.06, 0.04, 0.02, 0.02, 0.01])

    # ------------------------------------------------------------------ #
    #  4. Target — logistic function → ~20 % default rate                 #
    # ------------------------------------------------------------------ #
    # Higher int_rate, lower FICO, higher DTI, lower income, more
    # delinquencies → higher probability of default.
    log_odds = (
        -3.5
        + 0.12 * (int_rate - 15)
        - 0.03 * (fico_score - 700)
        + 0.06 * (dti - 18)
        - 0.000015 * (annual_inc - 60_000)
        + 0.35 * delinq_2yrs
        + 0.15 * pub_rec
        + rng.normal(0, 0.5, size=n_samples)   # irreducible noise
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    loan_status = (rng.rand(n_samples) < prob_default).astype(int)

    # ------------------------------------------------------------------ #
    #  5. Assemble DataFrame                                              #
    # ------------------------------------------------------------------ #
    df = pd.DataFrame({
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": int_rate,
        "grade": grade,
        "sub_grade": sub_grade,
        "emp_length": emp_length,
        "annual_inc": annual_inc,
        "dti": dti,
        "fico_score": fico_score,
        "home_ownership": home_ownership,
        "purpose": purpose,
        "revol_bal": revol_bal,
        "revol_util": revol_util_arr,
        "delinq_2yrs": delinq_2yrs,
        "pub_rec": pub_rec,
        "total_acc": total_acc,
        "inq_last_6mths": inq_last_6mths,
        "loan_status": loan_status,
    })

    return df


def get_data() -> pd.DataFrame:
    """Convenience wrapper — just calls the cached generator."""
    return generate_loan_data()
