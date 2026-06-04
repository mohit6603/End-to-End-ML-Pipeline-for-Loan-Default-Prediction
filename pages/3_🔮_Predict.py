"""
3_🔮_Predict.py — Real-time loan default prediction page.

Users fill in borrower details and receive an instant risk assessment
complete with a probability gauge and colour-coded risk badge.
Every prediction is automatically saved to the user's history.
"""

import streamlit as st

from auth.auth_handler import check_authentication
from auth.database import save_prediction
from data.preprocessor import preprocess_single_input
from models.predictor import predict_loan_default, get_prediction_color
from models.trainer import load_model
from utils.visualization import plot_probability_gauge

# ── Auth guard ──────────────────────────────────────────────────────────
check_authentication()

# ── Page header ─────────────────────────────────────────────────────────
st.markdown(
    '<h1 class="gradient-text">🔮 Loan Default Prediction</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    "Enter the borrower's details below and click **Predict** to get an "
    "instant risk assessment."
)
st.markdown("---")

# ── Resolve model + preprocessor ────────────────────────────────────────
model = None
preprocessor = None

# Priority 1: model trained in this session
if "xgb_model" in st.session_state and "preprocessor" in st.session_state:
    model = st.session_state["xgb_model"]
    preprocessor = st.session_state["preprocessor"]
elif "lr_model" in st.session_state and "preprocessor" in st.session_state:
    model = st.session_state["lr_model"]
    preprocessor = st.session_state["preprocessor"]
else:
    # Priority 2: load persisted model from disk
    bundle = load_model("xgboost_best")
    if bundle is None:
        bundle = load_model("logistic_regression_best")
    if bundle is not None:
        model = bundle["model"]
        preprocessor = bundle["preprocessor"]

if model is None:
    st.warning(
        "⚠️ No trained model found. Please go to the **🤖 Model Training** "
        "page and train a model first."
    )
    st.stop()

# ── Input form ──────────────────────────────────────────────────────────
st.markdown("### 📋 Borrower Details")

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown(
        '<div class="glass-card" style="padding:1.2rem;">',
        unsafe_allow_html=True,
    )
    st.markdown("##### 💰 Loan Information")

    loan_amnt = st.slider(
        "Loan Amount ($)", min_value=1_000, max_value=40_000,
        value=15_000, step=500, help="Total amount the borrower is requesting.",
    )
    term = st.selectbox(
        "Term (months)", options=[36, 60], index=0,
        help="Repayment period — 36 or 60 months.",
    )
    int_rate = st.slider(
        "Interest Rate (%)", min_value=5.0, max_value=30.0,
        value=12.0, step=0.25, help="Annual interest rate assigned to the loan.",
    )
    grade = st.selectbox(
        "Grade", options=list("ABCDEFG"), index=2,
        help="Risk grade assigned by the lender (A = safest, G = riskiest).",
    )
    purpose = st.selectbox(
        "Loan Purpose",
        options=[
            "debt_consolidation", "credit_card", "home_improvement",
            "major_purchase", "small_business", "other",
        ],
        help="Primary reason for the loan.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown(
        '<div class="glass-card" style="padding:1.2rem;">',
        unsafe_allow_html=True,
    )
    st.markdown("##### 👤 Borrower Profile")

    annual_inc = st.number_input(
        "Annual Income ($)", min_value=20_000, max_value=200_000,
        value=65_000, step=5_000, help="Borrower's self-reported yearly income.",
    )
    dti = st.slider(
        "DTI Ratio", min_value=0.0, max_value=50.0, value=18.0, step=0.5,
        help="Debt-to-Income: monthly debt payments ÷ gross monthly income × 100.",
    )
    fico_score = st.slider(
        "FICO Score", min_value=300, max_value=850, value=700, step=5,
        help="Credit score (300–850). Higher = more creditworthy.",
    )
    emp_length = st.slider(
        "Employment Length (years)", min_value=0, max_value=10,
        value=5, step=1, help="Number of years at current employer.",
    )
    home_ownership = st.selectbox(
        "Home Ownership", options=["RENT", "OWN", "MORTGAGE"], index=0,
        help="Borrower's housing situation.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ── Additional credit history ──────────────────────────────────────────
st.markdown("### 📈 Credit History")

cred1, cred2, cred3, cred4 = st.columns(4, gap="medium")

with cred1:
    revol_util = st.slider(
        "Revolving Utilisation (%)", min_value=0.0, max_value=100.0,
        value=45.0, step=1.0,
        help="% of available revolving credit currently used.",
    )
with cred2:
    delinq_2yrs = st.number_input(
        "Delinquencies (2 yr)", min_value=0, max_value=10, value=0,
        help="Past-due payments in the last 2 years.",
    )
with cred3:
    pub_rec = st.number_input(
        "Public Records", min_value=0, max_value=5, value=0,
        help="Derogatory public records (e.g. bankruptcies).",
    )
with cred4:
    total_acc = st.slider(
        "Total Accounts", min_value=5, max_value=50, value=20,
        help="Total number of credit lines ever opened.",
    )

extra1, extra2, extra3 = st.columns([1, 1, 2], gap="medium")

with extra1:
    revol_bal = st.number_input(
        "Revolving Balance ($)", min_value=0, max_value=100_000,
        value=12_000, step=1_000,
        help="Total revolving balance (e.g. credit card debt).",
    )
with extra2:
    inq_last_6mths = st.number_input(
        "Inquiries (6 mo)", min_value=0, max_value=8, value=1,
        help="Credit inquiries in the last 6 months.",
    )

# ── Predict button ──────────────────────────────────────────────────────
st.markdown("---")

if st.button("🔮 Predict Default Risk", use_container_width=True, type="primary"):

    # Assemble raw input dict matching the column names used in training
    input_data = {
        "loan_amnt": loan_amnt,
        "term": term,
        "int_rate": int_rate,
        "grade": grade,
        "emp_length": float(emp_length),
        "annual_inc": annual_inc,
        "dti": dti,
        "fico_score": fico_score,
        "home_ownership": home_ownership,
        "purpose": purpose,
        "revol_bal": revol_bal,
        "revol_util": revol_util,
        "delinq_2yrs": delinq_2yrs,
        "pub_rec": pub_rec,
        "total_acc": total_acc,
        "inq_last_6mths": inq_last_6mths,
    }

    with st.spinner("Running prediction model …"):
        try:
            prediction, probability, risk_level = predict_loan_default(
                model, preprocessor, input_data,
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    # ── Results ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Prediction Results")

    res_left, res_right = st.columns([1, 1], gap="large")

    with res_left:
        # Probability gauge
        gauge_fig = plot_probability_gauge(probability)
        st.plotly_chart(gauge_fig, use_container_width=True)

    with res_right:
        color = get_prediction_color(risk_level)

        # Risk badge
        st.markdown(
            f"""
            <div style="
                text-align:center; padding:2rem 1rem;
                border-radius:16px;
                background: #f8f9fc;
                border: 2px solid {color};
            ">
                <p style="font-size:1.1rem; color:#64748b; margin-bottom:0.3rem;">
                    Risk Level
                </p>
                <h2 style="color:{color}; font-size:2.2rem; margin:0;">
                    {risk_level}
                </h2>
                <p style="font-size:2.5rem; margin:0.5rem 0;">
                    {"✅" if prediction == 0 else "⚠️"}
                </p>
                <p style="color:#1a202c; font-size:1.05rem;">
                    {"Loan is likely to be <b>Fully Paid</b>" if prediction == 0
                     else "Loan is likely to <b>Default</b>"}
                </p>
                <p style="color:#64748b; font-size:0.9rem; margin-top:0.6rem;">
                    Default Probability: <b>{probability:.1%}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Interpretation ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💡 Interpretation")

    factors = []
    if int_rate > 18:
        factors.append("🔴 **High interest rate** (>18%) significantly increases default risk.")
    elif int_rate < 10:
        factors.append("🟢 **Low interest rate** (<10%) indicates lower risk.")

    if fico_score < 650:
        factors.append("🔴 **Low FICO score** (<650) is a strong default predictor.")
    elif fico_score > 740:
        factors.append("🟢 **High FICO score** (>740) indicates excellent creditworthiness.")

    if dti > 30:
        factors.append("🔴 **High DTI** (>30) means heavy existing debt burden.")
    elif dti < 15:
        factors.append("🟢 **Low DTI** (<15) suggests manageable debt levels.")

    if delinq_2yrs > 0:
        factors.append(f"🟡 **{delinq_2yrs} delinquency** record(s) in the past 2 years.")

    if annual_inc < 40_000:
        factors.append("🔴 **Low annual income** may reduce repayment ability.")
    elif annual_inc > 100_000:
        factors.append("🟢 **Strong annual income** supports repayment capacity.")

    if revol_util > 70:
        factors.append("🟡 **High revolving utilisation** (>70%) suggests credit strain.")

    if factors:
        for f in factors:
            st.markdown(f)
    else:
        st.info("The borrower profile shows balanced risk indicators across all features.")

    # ── Save to history ─────────────────────────────────────────────────
    try:
        save_prediction(
            user_id=st.session_state.user["id"],
            input_data=input_data,
            prediction=prediction,
            probability=probability,
            risk_level=risk_level,
        )
        st.success("✅ Prediction saved to your history.")
    except Exception as e:
        st.warning(f"Could not save prediction: {e}")
