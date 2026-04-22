"""
config.py — Central configuration for Bayesian Stock Forecasting System
All constants live here. Never hardcode values in other files.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).parent
DATA_RAW       = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
PLOTS_DIR      = ROOT_DIR / "plots"
MODELS_DIR     = ROOT_DIR / "models"

for d in [DATA_RAW, DATA_PROCESSED, PLOTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Tickers ──────────────────────────────────────────────────────────────────
TICKERS = {
    "NIFTY50":  "^NSEI",
    "SP500":    "^GSPC",
    "RELIANCE": "RELIANCE.NS",   # bonus: individual stock
}

# ── Data Window ──────────────────────────────────────────────────────────────
START_DATE = "2010-01-01"
END_DATE   = "2024-12-31"

# ── Train / Val / Test Split (time-based — NEVER random) ─────────────────────
TRAIN_END = "2020-12-31"
VAL_END   = "2022-12-31"
# Test = 2023-01-01 onwards

# ── Feature Engineering ──────────────────────────────────────────────────────
LAG_PERIODS     = [1, 5, 10, 21]          # trading days
ROLLING_WINDOWS = [5, 10, 21, 63]         # 1w, 2w, 1m, 1q
VOLATILITY_WIN  = 21                       # 1-month rolling σ

# ── Forecasting ──────────────────────────────────────────────────────────────
FORECAST_HORIZON = 5                       # predict next 5 trading days
SEQUENCE_LENGTH  = 60                      # LSTM lookback window (days)

# ── Random Seed ──────────────────────────────────────────────────────────────
SEED = 42