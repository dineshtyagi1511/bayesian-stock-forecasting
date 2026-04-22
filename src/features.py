"""
src/features.py — Feature Engineering (Statistics-Heavy)
Week 1, Step 2: Build all features that showcase M.Sc. Statistics knowledge

Key features built here:
  - Cyclical time encoding (sin/cos) — avoids ordinal assumption
  - Rolling statistics (mean, std, skew, kurtosis)
  - Lag returns
  - Technical indicators (RSI, Bollinger Bands, ATR)
  - Stationarity-aware log returns
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LAG_PERIODS, ROLLING_WINDOWS, VOLATILITY_WIN, DATA_PROCESSED


# ── 1. Returns & Log Returns ──────────────────────────────────────────────────

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple and log returns.
    Log returns are additive over time and more statistically well-behaved
    (approximately normally distributed) — key M.Sc. talking point.
    """
    df = df.copy()
    df["return_1d"]     = df["Close"].pct_change()
    df["log_return_1d"] = np.log(df["Close"] / df["Close"].shift(1))

    # Cumulative log return (price relative to start)
    df["log_return_cum"] = df["log_return_1d"].cumsum()
    return df


# ── 2. Cyclical Time Features ─────────────────────────────────────────────────

def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode day-of-week and month using sin/cos transformation.

    WHY: Raw encoding (Mon=0, Tue=1...) implies Mon is "close to" Tue but "far
    from" Fri — which is wrong. Sin/cos preserves circular distance.
    This is a pure M.Sc. Statistics insight that impresses interviewers.
    """
    df = df.copy()
    dow   = df.index.dayofweek          # 0=Mon … 4=Fri
    month = df.index.month              # 1 … 12

    df["dow_sin"]   = np.sin(2 * np.pi * dow / 5)
    df["dow_cos"]   = np.cos(2 * np.pi * dow / 5)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Additional calendar flags
    df["is_month_end"]   = df.index.is_month_end.astype(int)
    df["is_quarter_end"] = df.index.is_quarter_end.astype(int)
    df["week_of_year"]   = df.index.isocalendar().week.astype(int)

    return df


# ── 3. Lag Returns ────────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lagged return features.
    These give the model 'memory' of past performance.
    """
    df = df.copy()
    for lag in LAG_PERIODS:
        df[f"return_lag{lag}d"] = df["Close"].pct_change(lag)
    return df


# ── 4. Rolling Statistics ─────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling window statistics — where Statistics degree really shines.
    Includes: mean, std (volatility), skewness, kurtosis.

    Interview point: rolling skewness captures asymmetric risk;
    rolling kurtosis captures tail risk (fat tails in financial returns).
    """
    df = df.copy()
    r = df["log_return_1d"]

    for w in ROLLING_WINDOWS:
        prefix = f"roll{w}d"
        df[f"{prefix}_mean"]     = r.rolling(w).mean()
        df[f"{prefix}_std"]      = r.rolling(w).std()    # realized volatility
        df[f"{prefix}_skew"]     = r.rolling(w).skew()   # asymmetry
        df[f"{prefix}_kurt"]     = r.rolling(w).kurt()   # fat tails
        df[f"{prefix}_close_ma"] = df["Close"].rolling(w).mean()

    # Annualized volatility (σ × √252 — standard in finance)
    df["volatility_ann"] = df[f"roll{VOLATILITY_WIN}d_std"] * np.sqrt(252)

    return df


# ── 5. Technical Indicators ───────────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RSI, Bollinger Bands, ATR — standard in quant finance.
    These are derived features that capture momentum and mean-reversion signals.
    """
    df = df.copy()

    # RSI (Relative Strength Index) — momentum oscillator
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands — mean-reversion signal
    ma20 = df["Close"].rolling(20).mean()
    sd20 = df["Close"].rolling(20).std()
    df["bb_upper"]  = ma20 + 2 * sd20
    df["bb_lower"]  = ma20 - 2 * sd20
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / ma20   # normalized width
    df["bb_pct"]    = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # ATR (Average True Range) — volatility measure
    high_low   = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close  = (df["Low"]  - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean()

    # MA crossover signal (golden/death cross proxy)
    df["ma_cross_50_200"] = (
        df["Close"].rolling(50).mean() - df["Close"].rolling(200).mean()
    ) / df["Close"]

    return df


# ── 6. Target Variable ────────────────────────────────────────────────────────

def add_target(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Create the forecast target: forward return over `horizon` trading days.
    target_5d = (Close[t+5] - Close[t]) / Close[t]

    NOTE: This is the future — never use it as an input feature (data leakage).
    """
    df = df.copy()
    df[f"target_{horizon}d"] = df["Close"].pct_change(horizon).shift(-horizon)
    return df


# ── 7. Master Pipeline ────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline on a raw OHLCV DataFrame.
    Returns a feature-rich DataFrame ready for modeling.
    """
    logger.info("Building features...")
    df = add_returns(df)
    df = add_cyclical_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_technical_indicators(df)
    df = add_target(df, horizon=horizon)

    # Drop rows with NaN (from rolling windows at start of series)
    n_before = len(df)
    df = df.dropna()
    n_after  = len(df)
    logger.info(f"Dropped {n_before - n_after} NaN rows (rolling window warmup)")
    logger.success(f"Feature matrix: {df.shape[0]} rows × {df.shape[1]} columns")

    return df


def save_features(df: pd.DataFrame, ticker_name: str) -> None:
    path = DATA_PROCESSED / f"{ticker_name.lower()}_features.csv"
    df.to_csv(path)
    logger.success(f"Saved features → {path}")


def load_features(ticker_name: str) -> pd.DataFrame:
    path = DATA_PROCESSED / f"{ticker_name.lower()}_features.csv"
    return pd.read_csv(path, index_col="Date", parse_dates=True)


if __name__ == "__main__":
    from src.data_pipeline import load_raw
    logger.info("=== Week 1 · Step 2: Feature Engineering ===")

    for ticker in ["NIFTY50", "SP500"]:
        raw = load_raw(ticker)
        features = build_features(raw)
        save_features(features, ticker)

        print(f"\n{ticker} — Feature columns ({len(features.columns)}):")
        for col in features.columns:
            print(f"  {col}")