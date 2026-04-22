"""
src/splitter.py — Time-Series Aware Data Splitting
Week 1, Step 3: Proper train/val/test split for time-series data

CRITICAL interview point:
  ❌ random_state split → data leakage (future data in training)
  ✅ time-based split   → realistic, no lookahead bias
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from dataclasses import dataclass
from pathlib import Path
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TRAIN_END, VAL_END, FORECAST_HORIZON


@dataclass
class DataSplit:
    """Container for train/val/test splits with metadata."""
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val:   pd.DataFrame
    y_val:   pd.Series
    X_test:  pd.DataFrame
    y_test:  pd.Series

    # Raw price series (needed for Bayesian models)
    train_prices: pd.Series
    val_prices:   pd.Series
    test_prices:  pd.Series

    # Feature column names (exclude target & raw OHLCV)
    feature_cols: list[str]

    def summary(self) -> str:
        return (
            f"Train: {len(self.X_train):,} rows  "
            f"({self.X_train.index[0].date()} → {self.X_train.index[-1].date()})\n"
            f"Val:   {len(self.X_val):,} rows  "
            f"({self.X_val.index[0].date()} → {self.X_val.index[-1].date()})\n"
            f"Test:  {len(self.X_test):,} rows  "
            f"({self.X_test.index[0].date()} → {self.X_test.index[-1].date()})\n"
            f"Features: {len(self.feature_cols)}"
        )


# Columns that are NOT input features
_EXCLUDE = {
    "Open", "High", "Low", "Close", "Volume",
    "return_1d", "log_return_1d", "log_return_cum",
}


def split_data(df: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> DataSplit:
    """
    Time-based train / val / test split.

    Split dates (from config.py):
      Train: start → 2020-12-31
      Val:   2021-01-01 → 2022-12-31
      Test:  2023-01-01 → end
    """
    target_col   = f"target_{horizon}d"
    feature_cols = [
        c for c in df.columns
        if c not in _EXCLUDE and c != target_col
    ]

    train = df[df.index <= TRAIN_END]
    val   = df[(df.index > TRAIN_END) & (df.index <= VAL_END)]
    test  = df[df.index > VAL_END]

    split = DataSplit(
        X_train = train[feature_cols],
        y_train = train[target_col],
        X_val   = val[feature_cols],
        y_val   = val[target_col],
        X_test  = test[feature_cols],
        y_test  = test[target_col],
        train_prices = train["Close"],
        val_prices   = val["Close"],
        test_prices  = test["Close"],
        feature_cols = feature_cols,
    )

    logger.info(f"\nData split complete:\n{split.summary()}")
    _check_leakage(split)
    return split


def _check_leakage(split: DataSplit) -> None:
    """Assert no temporal overlap between splits (catches bugs early)."""
    assert split.X_train.index.max() < split.X_val.index.min(), \
        "DATA LEAKAGE: train and val overlap!"
    assert split.X_val.index.max() < split.X_test.index.min(), \
        "DATA LEAKAGE: val and test overlap!"
    logger.success("Leakage check passed — no temporal overlap between splits.")


def get_walk_forward_splits(
    X: pd.DataFrame,
    n_splits: int = 5
) -> TimeSeriesSplit:
    """
    Walk-forward cross-validation for time-series.
    Use this inside model training, NOT for final evaluation.

    Each fold: train on past, validate on immediate future.
    This simulates real trading conditions.
    """
    return TimeSeriesSplit(
        n_splits=n_splits,
        gap=0,                          # no gap between train & val
        test_size=len(X) // (n_splits + 1),
    )


if __name__ == "__main__":
    from src.features import load_features
    logger.info("=== Week 1 · Step 3: Data Splitting ===")

    df = load_features("NIFTY50")
    split = split_data(df)
    print(split.summary())

    print("\nWalk-forward CV splits on training data:")
    tscv = get_walk_forward_splits(split.X_train, n_splits=5)
    for fold, (train_idx, val_idx) in enumerate(tscv.split(split.X_train)):
        tr = split.X_train.iloc[train_idx]
        vl = split.X_train.iloc[val_idx]
        print(
            f"  Fold {fold+1}: "
            f"train={tr.index[0].date()}→{tr.index[-1].date()} | "
            f"val={vl.index[0].date()}→{vl.index[-1].date()}"
        )