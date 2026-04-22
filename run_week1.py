"""
run_week1.py — Master script for Week 1
Run this file to execute the complete data pipeline end-to-end.

  python run_week1.py

What it does:
  1. Downloads NIFTY50 & S&P500 data (2010–2024)
  2. Engineers 40+ statistical features
  3. Creates time-based train/val/test split
  4. Runs full EDA + saves all plots
  5. Prints a summary report
"""

import sys
from pathlib import Path
from loguru import logger
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import TICKERS, PLOTS_DIR
from src.data_pipeline  import download_all, load_raw

from src.features       import build_features, save_features, load_features
from src.splitter       import split_data, get_walk_forward_splits
from src.eda            import run_full_eda

# ── Logger setup ─────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
logger.add("logs/week1.log", rotation="1 MB")

Path("logs").mkdir(exist_ok=True)


def main():
    logger.info("╔══════════════════════════════════════╗")
    logger.info("║  Bayesian Stock Forecasting System   ║")
    logger.info("║  Week 1: Data Pipeline               ║")
    logger.info("╚══════════════════════════════════════╝")

    # ── Step 1: Download (fallback to synthetic if network blocked) ──
    logger.info("\n── STEP 1: Download Raw Data ──")
    raw_data = download_all()
    if not raw_data:
        logger.warning("yfinance unavailable ")
        

    # ── Step 2: Feature Engineering ──────────────────────────
    logger.info("\n── STEP 2: Feature Engineering ──")
    feature_data = {}
    for ticker in raw_data:
        df_feat = build_features(raw_data[ticker])
        save_features(df_feat, ticker)
        feature_data[ticker] = df_feat

    # ── Step 3: Train/Val/Test Split ─────────────────────────
    logger.info("\n── STEP 3: Data Splitting ──")
    splits = {}
    for ticker, df_feat in feature_data.items():
        splits[ticker] = split_data(df_feat)

    # Walk-forward CV preview
    nifty_split = splits["NIFTY50"]
    tscv = get_walk_forward_splits(nifty_split.X_train, n_splits=5)
    logger.info("Walk-forward CV folds on NIFTY50 training data:")
    for fold, (tr_idx, vl_idx) in enumerate(tscv.split(nifty_split.X_train)):
        tr = nifty_split.X_train.iloc[tr_idx]
        vl = nifty_split.X_train.iloc[vl_idx]
        logger.info(
            f"  Fold {fold+1}: train {tr.index[0].date()}→{tr.index[-1].date()} "
            f"| val {vl.index[0].date()}→{vl.index[-1].date()}"
        )

    # ── Step 4: EDA ──────────────────────────────────────────
    logger.info("\n── STEP 4: Exploratory Data Analysis ──")
    for ticker, df_feat in feature_data.items():
        run_full_eda(df_feat, ticker)

    # ── Final Summary ─────────────────────────────────────────
    logger.info("\n╔══════════════════════════════════════╗")
    logger.info("║           WEEK 1 COMPLETE ✅          ║")
    logger.info("╚══════════════════════════════════════╝")

    print("\n" + "="*55)
    print("WEEK 1 SUMMARY REPORT")
    print("="*55)

    for ticker, split in splits.items():
        df_feat = feature_data[ticker]
        print(f"\n📊 {ticker}")
        print(f"   Total rows      : {len(df_feat):,}")
        print(f"   Features built  : {len(split.feature_cols)}")
        print(f"   Train rows      : {len(split.X_train):,}")
        print(f"   Val rows        : {len(split.X_val):,}")
        print(f"   Test rows       : {len(split.X_test):,}")
        print(f"   Date range      : {df_feat.index[0].date()} → {df_feat.index[-1].date()}")

        returns = df_feat["log_return_1d"]
        print(f"   Ann. volatility : {returns.std() * (252**0.5) * 100:.2f}%")
        print(f"   Skewness        : {returns.skew():.4f}")
        print(f"   Excess Kurtosis : {returns.kurt():.4f}  (fat tails)")

    print(f"\n📁 Plots saved to   : {PLOTS_DIR}/")
    print(f"📁 Features saved to: data/processed/")
    print("\nNext → Week 2: SARIMA + XGBoost + LSTM models")
    print("="*55)


if __name__ == "__main__":
    main()
