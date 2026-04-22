"""
src/data_pipeline.py — Data Download & Validation
Week 1, Step 1: Download raw OHLCV data from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW, TICKERS, START_DATE, END_DATE


def download_ticker(ticker_name: str, ticker_symbol: str) -> pd.DataFrame:
    """
    Download OHLCV data for one ticker and save to data/raw/.
    Returns a clean DataFrame with DatetimeIndex.
    """
    logger.info(f"Downloading {ticker_name} ({ticker_symbol})...")

    df = yf.download(
        ticker_symbol,
        start=START_DATE,
        end=END_DATE,
        progress=False,
        auto_adjust=True,          # adjusts for splits & dividends
    )

    if df.empty:
        raise ValueError(f"No data returned for {ticker_symbol}")

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only OHLCV
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # ── Data Quality Checks ──────────────────────────────────
    _validate(df, ticker_name)

    # Save raw CSV
    path = DATA_RAW / f"{ticker_name.lower()}_raw.csv"
    df.to_csv(path)
    logger.success(f"Saved {len(df)} rows → {path}")

    return df


def _validate(df: pd.DataFrame, name: str) -> None:
    """Basic data quality assertions — catch issues early."""
    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        logger.warning(f"{name}: {n_missing} missing values found — will forward-fill later")

    n_duplicates = df.index.duplicated().sum()
    if n_duplicates > 0:
        logger.warning(f"{name}: {n_duplicates} duplicate dates found")

    # Check for price anomalies (close = 0 or negative)
    zero_prices = (df["Close"] <= 0).sum()
    if zero_prices > 0:
        raise ValueError(f"{name}: {zero_prices} rows with Close ≤ 0")

    # Check for large single-day gaps (> 50% move = likely data error)
    returns = df["Close"].pct_change().dropna()
    extreme = (returns.abs() > 0.5).sum()
    if extreme > 0:
        logger.warning(f"{name}: {extreme} day(s) with >50% price move — check data")

    logger.info(f"{name} validation passed | rows={len(df)} | "
                f"from={df.index[0].date()} to={df.index[-1].date()}")


def load_raw(ticker_name: str) -> pd.DataFrame:
    """Load previously downloaded raw CSV."""
    path = DATA_RAW / f"{ticker_name.lower()}_raw.csv"
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


def download_all() -> dict[str, pd.DataFrame]:
    """Download all tickers defined in config.py."""
    results = {}
    for name, symbol in TICKERS.items():
        try:
            results[name] = download_ticker(name, symbol)
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")
    return results


if __name__ == "__main__":
    logger.info("=== Week 1 · Step 1: Data Download ===")
    data = download_all()
    for name, df in data.items():
        print(f"\n{name}:")
        print(df.tail(3))
        print(f"Shape: {df.shape}")