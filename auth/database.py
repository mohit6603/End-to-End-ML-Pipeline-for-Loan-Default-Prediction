"""
database.py — SQLite database layer for LoanGuard AI.

Handles two tables:
  • users          – credentials & profile info (bcrypt-hashed passwords)
  • prediction_history – every prediction a user has made

Design decisions
----------------
* We use parameterised queries everywhere to prevent SQL injection.
* bcrypt is chosen over hashlib because it's purpose-built for passwords
  (adaptive cost factor, built-in salting).
* The DB file lives in  <project_root>/data/app.db  so the data/ package
  doubles as a runtime artefact store.
"""

import os
import json
import sqlite3
from datetime import datetime

import bcrypt

# ---------------------------------------------------------------------------
# Path setup — DB sits alongside generated datasets in  data/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
DB_PATH = os.path.join(_DATA_DIR, "app.db")


def _get_connection() -> sqlite3.Connection:
    """Return a new connection with row-factory enabled so rows behave like dicts."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row          # access columns by name
    conn.execute("PRAGMA foreign_keys = ON")  # enforce FK constraints
    return conn


# ========================== SCHEMA SETUP ====================================

def init_db() -> None:
    """Create the users and prediction_history tables if they don't exist.

    Safe to call on every app startup — IF NOT EXISTS makes it idempotent.
    """
    conn = _get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER   PRIMARY KEY AUTOINCREMENT,
                username      TEXT      UNIQUE NOT NULL,
                email         TEXT      UNIQUE NOT NULL,
                password_hash TEXT      NOT NULL,
                full_name     TEXT      NOT NULL,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS prediction_history (
                id          INTEGER   PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER   NOT NULL
                                      REFERENCES users(id) ON DELETE CASCADE,
                input_data  TEXT      NOT NULL,   -- JSON blob of raw features
                prediction  INTEGER  NOT NULL,    -- 0 = no default, 1 = default
                probability REAL     NOT NULL,    -- model's P(default)
                risk_level  TEXT     NOT NULL,     -- Low / Medium / High Risk
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
    finally:
        conn.close()


# ========================== USER OPERATIONS =================================

def create_user(username: str, email: str, password: str, full_name: str) -> bool:
    """Register a new user with a bcrypt-hashed password.

    Returns True on success, False if the username or email already exists.
    """
    # bcrypt wants bytes — encode, hash, then store the UTF-8 string
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, email, password_hash, full_name) "
            "VALUES (?, ?, ?, ?)",
            (username, email, password_hash.decode("utf-8"), full_name),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Duplicate username or email — UNIQUE constraint violated
        return False
    finally:
        conn.close()


def verify_user(username: str, password: str) -> dict | None:
    """Authenticate a user by username + plaintext password.

    Returns a user dict on success, or None on failure (wrong password / user
    not found).  We never reveal *which* of the two failed — that's a security
    best practice.
    """
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()

        if row is None:
            return None

        stored_hash = row["password_hash"].encode("utf-8")
        if bcrypt.checkpw(password.encode("utf-8"), stored_hash):
            return dict(row)  # convert sqlite3.Row → plain dict
        return None
    finally:
        conn.close()


def get_user_by_username(username: str) -> dict | None:
    """Look up a user record by username (no password check)."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ========================== PREDICTION HISTORY ==============================

def save_prediction(
    user_id: int,
    input_data: dict,
    prediction: int,
    probability: float,
    risk_level: str,
) -> None:
    """Persist a single prediction to the history table.

    `input_data` is serialised as JSON so we can reconstruct the exact inputs
    later (useful for audit trails and the History page).
    """
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO prediction_history "
            "(user_id, input_data, prediction, probability, risk_level) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, json.dumps(input_data), int(prediction), float(probability), risk_level),
        )
        conn.commit()
    finally:
        conn.close()


def get_user_predictions(user_id: int) -> list[dict]:
    """Return all predictions for a user, newest first."""
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM prediction_history "
            "WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
