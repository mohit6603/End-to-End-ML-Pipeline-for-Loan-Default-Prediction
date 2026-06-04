# 🏦 LoanGuard AI — Loan Default Prediction System

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-006400?style=for-the-badge)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

## 🧐 What is this project?

**LoanGuard AI** is a full-stack Machine Learning web application that predicts whether a loan applicant is likely to **default** (fail to repay) on their loan.

Think of it like this — banks need a way to quickly assess the risk of lending money to someone. Instead of manually reviewing every application, this system uses machine learning to analyse 15+ borrower attributes (like income, credit score, debt ratio, loan amount) and gives an instant risk assessment: **Low Risk**, **Medium Risk**, or **High Risk**.

But this isn't just a Jupyter notebook with some charts. This is a **production-grade web application** with:
- A real **user authentication system** (sign up, log in, log out) backed by a database
- An **interactive dashboard** where you can explore the data visually
- A **model training page** where you can train and compare ML models with one click
- A **prediction page** where you enter borrower details and get instant results
- A **history page** that tracks every prediction you've made

Everything runs in your browser, and it's deployable for free on Streamlit Community Cloud.

---

## 🎬 How It Works (The User Flow)

```
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │  1. SIGN UP  │ ──▶ │  2. EXPLORE  │ ──▶ │   3. TRAIN   │ ──▶ │ 4. PREDICT   │
 │  Create your │     │  Dive into   │     │  Train ML    │     │  Enter loan   │
 │  account     │     │  the data    │     │  models      │     │  details &    │
 │  securely    │     │  with charts │     │  & compare   │     │  get instant  │
 │              │     │              │     │  results     │     │  risk score   │
 └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

---

## ✨ Key Features

### 🔐 User Authentication
Not just a simple password check — this is a **properly built auth system**:
- Passwords are hashed using **bcrypt** (the same algorithm banks use). Nobody — not even the app admin — can see your password.
- Every password gets its own random **salt**, so even if two users pick the same password, their hashes are completely different.
- All database queries use **parameterized statements** to prevent SQL injection attacks.
- Session management via Streamlit's `session_state` — every page has an auth guard that blocks access if you're not logged in.

### 📊 Interactive EDA Dashboard
- **Dataset overview** — see shape, data types, missing values at a glance
- **Distribution plots** — understand how loan amounts, interest rates, and FICO scores are spread
- **Correlation heatmap** — discover which features are related to each other
- **Target analysis** — see how default rates differ by grade, loan purpose, home ownership
- **Feature vs. target** — box plots that show exactly how features split between defaults and non-defaults

### 🤖 ML Model Training
Train two different models and compare them head-to-head:

| Model | Type | Why it's here |
|-------|------|---------------|
| **Logistic Regression** | Linear (baseline) | Fast, interpretable, industry standard in banking |
| **XGBoost** | Gradient Boosted Trees | State-of-the-art for tabular data, handles missing values natively |

After training, you get:
- Side-by-side **metrics comparison** (Accuracy, Precision, Recall, F1, ROC-AUC)
- Overlaid **ROC curves** so you can visually compare model performance
- **Confusion matrices** showing exactly where each model goes right and wrong
- **Feature importance** chart revealing which factors matter most

### 🔮 Real-Time Predictions
- Fill in a form with the borrower's details (income, credit score, loan amount, etc.)
- Hit "Predict" and get an instant **probability gauge** showing default likelihood
- See a colour-coded **risk badge** (🟢 Low / 🟡 Medium / 🔴 High)
- Read plain-English **interpretation** of what's driving the prediction
- Every prediction is **automatically saved** to your personal history

### 📜 Prediction History
- View all your past predictions in a clean table
- See summary stats — total predictions, average risk, risk distribution
- **Download as CSV** for further analysis
- Full audit trail tied to your user account

---

## 🛠️ Tech Stack — What I Used and Why

### Core Language
| Tech | Why I chose it |
|------|---------------|
| **Python 3.10+** | The go-to language for data science and ML. Every library in this stack has first-class Python support. |

### Web Framework
| Tech | Why I chose it |
|------|---------------|
| **Streamlit** | Lets you build interactive ML web apps using pure Python — no HTML/CSS/JS needed (though I added custom CSS for aesthetics). Perfect for data science projects because it's designed around DataFrames and charts. |

### Machine Learning
| Tech | Why I chose it |
|------|---------------|
| **Scikit-learn** | The Swiss Army knife of ML in Python. I use it for the preprocessing pipeline (imputation, encoding, scaling), Logistic Regression model, train/test splitting, and all evaluation metrics. |
| **XGBoost** | Consistently the best-performing algorithm for structured/tabular data. It uses gradient boosting — each tree learns from the mistakes of the previous ones. Handles missing values natively and has built-in regularization to prevent overfitting. |
| **SHAP** | Explains *why* the model made a particular prediction using game theory (Shapley values). Critical in finance where you legally need to explain why a loan was rejected. |
| **imbalanced-learn (SMOTE)** | Our dataset has ~80% non-defaults and ~20% defaults. SMOTE generates synthetic minority samples so the model doesn't just learn to predict "no default" for everything. Applied only to training data to prevent data leakage. |

### Data & Visualization
| Tech | Why I chose it |
|------|---------------|
| **Pandas** | The backbone of any data project. Used for loading, cleaning, transforming, and analysing the loan dataset. |
| **NumPy** | Fast numerical operations under the hood. Powers the synthetic data generator and works seamlessly with scikit-learn. |
| **Plotly** | Interactive charts that users can hover, zoom, and explore — much better than static images for a web app. All charts use a dark theme for visual consistency. |
| **Matplotlib + Seaborn** | Used for heatmaps and confusion matrices where Seaborn's statistical visualization style shines. |

### Database & Security
| Tech | Why I chose it |
|------|---------------|
| **SQLite** | A serverless, file-based SQL database. Zero configuration needed — it just works. Perfect for a self-contained portfolio project. Stores user accounts and prediction history. |
| **bcrypt** | Purpose-built password hashing. Unlike SHA-256 (which is fast = bad for passwords), bcrypt is intentionally slow and includes automatic salting. The cost factor can be increased over time as hardware gets faster. |

### Deployment
| Tech | Why I chose it |
|------|---------------|
| **Streamlit Community Cloud** | Free hosting for Streamlit apps. Just connect your GitHub repo and it auto-deploys. No Docker, no servers, no configuration. |

---

## 🧠 The ML Pipeline — Step by Step

Here's exactly what happens under the hood when you click "Train Models":

### Step 1: Data Generation
I generate **10,000 synthetic loan records** that mimic real Lending Club data. Why synthetic? It makes the app fully self-contained — no API keys, no external downloads, no licensing issues. The data has realistic distributions and built-in correlations (e.g., higher interest rates → higher default probability).

### Step 2: Preprocessing
```
Raw data (10,000 rows × 17 features)
    │
    ├── Handle missing values
    │   • Numeric columns → fill with median
    │   • Categorical columns → fill with most frequent value
    │
    ├── Encode categorical features
    │   • Grade (A-G) → OrdinalEncoder (preserves the natural order)
    │   • Home ownership, Purpose → OneHotEncoder (no natural order)
    │
    ├── Scale numeric features
    │   • StandardScaler → mean=0, std=1
    │   • Why? Features like income (20K-200K) and DTI (0-40) have 
    │     very different scales. Without scaling, Logistic Regression
    │     would give more weight to larger numbers.
    │
    ├── Train/Test split (80/20)
    │   • Stratified — ensures both sets have the same ~20% default rate
    │   • Why stratified? Random splits might give us a test set with 
    │     barely any defaults, making evaluation unreliable.
    │
    └── SMOTE oversampling (training data ONLY!)
        • Generates synthetic default cases so the model sees a balanced 
          dataset during training
        • ⚠️ NEVER applied to test data — that would be data leakage
```

### Step 3: Model Training
Two models are trained and compared:
- **Logistic Regression** — the baseline. Simple, fast, interpretable.
- **XGBoost** — the heavy hitter. 200 boosting rounds, each tree correcting the mistakes of the previous ones.

### Step 4: Evaluation
We don't just look at accuracy (which is misleading with imbalanced data). We use:
- **Precision** — "Of the loans I flagged as risky, how many actually defaulted?"
- **Recall** — "Of all the actual defaults, how many did I catch?"
- **F1 Score** — Harmonic mean of precision and recall
- **ROC-AUC** — The model's overall ability to distinguish defaults from non-defaults

---

## 🗃️ Database Design

```sql
-- Users: stores credentials and profile
CREATE TABLE users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    email         TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,          -- bcrypt hash, never plaintext
    full_name     TEXT NOT NULL,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Prediction History: every prediction linked to a user
CREATE TABLE prediction_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER REFERENCES users(id),  -- foreign key
    input_data  TEXT NOT NULL,                  -- JSON blob of inputs
    prediction  INTEGER NOT NULL,              -- 0 or 1
    probability REAL NOT NULL,                 -- 0.0 to 1.0
    risk_level  TEXT NOT NULL,                 -- Low/Medium/High
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Relationship**: One user → many predictions (one-to-many via `user_id` foreign key).

---

## 📂 Project Structure

```
loan-default-prediction/
│
├── app.py                          # 🏠 Main entry point (login gate + home page)
├── requirements.txt                # 📦 Python dependencies
├── README.md                       # 📖 You're reading this!
├── .gitignore                      # 🚫 Files to exclude from Git
│
├── auth/                           # 🔐 Authentication module
│   ├── __init__.py
│   ├── database.py                 #    SQLite setup, user CRUD, bcrypt hashing
│   └── auth_handler.py             #    Login/signup UI, session management
│
├── data/                           # 📊 Data module
│   ├── __init__.py
│   ├── generate_data.py            #    Synthetic dataset generator (10K rows)
│   └── preprocessor.py             #    Sklearn pipeline + SMOTE
│
├── models/                         # 🤖 ML models module
│   ├── __init__.py
│   ├── trainer.py                  #    Train, evaluate, save/load models
│   └── predictor.py                #    Inference + risk classification
│
├── pages/                          # 📄 Streamlit multi-page app
│   ├── 1_📊_EDA_Dashboard.py       #    Interactive data exploration
│   ├── 2_🤖_Model_Training.py      #    Train & compare models
│   ├── 3_🔮_Predict.py             #    Real-time loan risk prediction
│   └── 4_📜_History.py             #    User's prediction audit trail
│
├── utils/                          # 🧰 Shared utilities
│   ├── __init__.py
│   └── visualization.py            #    8 reusable chart functions (Plotly/Seaborn)
│
├── assets/
│   └── style.css                   #    🎨 Custom dark theme CSS (glassmorphism)
│
└── saved_models/                   #    💾 Persisted model files (auto-generated)
    └── .gitkeep
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Installation

```bash
# Clone this repository
git clone https://github.com/mohit6603/loan-default-prediction.git
cd loan-default-prediction

# Install all dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

The app will open at **http://localhost:8501** 🎉

### First Time?
1. Click the **Sign Up** tab and create an account
2. Switch to **Login** and sign in
3. Head to **📊 EDA Dashboard** to explore the data
4. Go to **🤖 Model Training** and hit "Train Models"
5. Visit **🔮 Predict** to score a loan applicant
6. Check **📜 History** to see your saved predictions

---

## ☁️ Free Deployment

### Streamlit Community Cloud (Recommended)
1. Push this repo to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select this repo → main file = `app.py`
4. Hit **Deploy** — your app gets a free public URL!

> **Note**: SQLite on Streamlit Cloud is ephemeral (resets on restart). For production, you'd swap to PostgreSQL via Supabase or Neon (both have free tiers).

---

## 🎤 Why This Project Stands Out

This isn't just another "predict with sklearn" notebook. Here's what makes it different:

1. **Full-stack, not just ML** — Auth system, database, web UI, deployment. This is what real ML products look like.
2. **Security-conscious** — bcrypt hashing, SQL injection prevention, session guards. Not just an afterthought.
3. **Honest evaluation** — SMOTE only on training data, stratified splits, multiple metrics beyond accuracy.
4. **Explainable AI** — SHAP values so you can explain *why* the model made each decision. Critical for regulated industries like finance.
5. **Modular architecture** — Clean separation between auth, data, models, and UI. Each module can be tested independently.
6. **Production-ready patterns** — Error handling, caching, input validation, defensive coding throughout.

---

## 🔮 What I'd Add Next

- [ ] Hyperparameter tuning with Optuna / GridSearchCV
- [ ] Cross-validation instead of single train/test split
- [ ] LightGBM and CatBoost model comparisons
- [ ] Data drift detection for production monitoring
- [ ] OAuth login (Google/GitHub) via Streamlit's native OIDC
- [ ] PostgreSQL migration for persistent multi-user deployment

---

## 📄 License

This project is licensed under the MIT License — feel free to use, modify, and distribute.

---

**Built with ❤️ using Python, Streamlit, and a lot of late-night debugging.**
