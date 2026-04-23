"""
run_week2_3.py — Master runner for Weeks 2 & 3
Runs all models end-to-end on NIFTY50 data and prints final comparison.

  python run_week2_3.py

What it does:
  Week 2:
    1. SARIMA walk-forward forecast (statistical baseline)
    2. XGBoost with walk-forward CV (ML model)

  Week 3:
    3. Bayesian Linear Regression with 95% credible intervals
    4. Bayesian A/B test: SARIMA strategy vs XGBoost strategy

  Final:
    5. Model comparison table + hero chart
    6. Resume-ready bullet points printed to console
"""

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import PLOTS_DIR
from src.features         import load_features
from src.splitter         import split_data
from src.model_sarima     import run_sarima,   walk_forward_sarima
from src.model_xgboost    import run_xgboost
from src.model_bayesian   import run_bayesian
from src.model_comparison import plot_model_comparison, print_final_report

# Logger
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")
logger.add("logs/week2_3.log", rotation="1 MB")
Path("logs").mkdir(exist_ok=True)


TICKER = "NIFTY50"   # change to SP500 or RELIANCE to run on other tickers


def main():
    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  Bayesian Stock Forecasting System       ║")
    logger.info("║  Week 2: SARIMA + XGBoost                ║")
    logger.info("║  Week 3: Bayesian + A/B Test             ║")
    logger.info("╚══════════════════════════════════════════╝")

    # ── Load Week 1 outputs ──────────────────────────────────────
    logger.info(f"\n── Loading {TICKER} features from Week 1 ──")
    df    = load_features(TICKER)
    split = split_data(df)
    logger.info(f"Train: {len(split.X_train):,}  Val: {len(split.X_val):,}  Test: {len(split.X_test):,}")

    # ── WEEK 2 · Step 1: SARIMA ──────────────────────────────────
    logger.info("\n" + "█"*50)
    logger.info("█  WEEK 2 · STEP 1 — SARIMA (Statistical Baseline)")
    logger.info("█"*50)
    sarima_metrics = run_sarima(split, TICKER)

    # Re-run walk-forward to get predictions array for A/B test
    logger.info("Re-running SARIMA walk-forward to capture predictions for A/B test...")
    sarima_y_true, sarima_y_pred = walk_forward_sarima(
        split.train_prices, split.test_prices
    )

    # ── WEEK 2 · Step 2: XGBoost ─────────────────────────────────
    logger.info("\n" + "█"*50)
    logger.info("█  WEEK 2 · STEP 2 — XGBoost (ML Model)")
    logger.info("█"*50)
    xgb_metrics = run_xgboost(split, TICKER)

    # Get XGBoost predictions for A/B test
    import pickle
    from config import MODELS_DIR
    with open(MODELS_DIR / "xgboost_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    xgb_preds_test = xgb_model.predict(split.X_test)

    # Align lengths for A/B test (SARIMA may have fewer predictions)
    n = min(len(sarima_y_pred), len(xgb_preds_test))
    sarima_preds_aligned = sarima_y_pred[:n]
    xgb_preds_aligned    = xgb_preds_test[:n]

    # ── WEEK 3 · Bayesian ────────────────────────────────────────
    logger.info("\n" + "█"*50)
    logger.info("█  WEEK 3 — Bayesian Regression + A/B Test")
    logger.info("█"*50)
    bayes_metrics, ab_result = run_bayesian(
        split, TICKER,
        sarima_preds=sarima_preds_aligned,
        xgb_preds=xgb_preds_aligned,
    )

    # ── Final Comparison ──────────────────────────────────────────
    logger.info("\n" + "█"*50)
    logger.info("█  FINAL — Model Comparison")
    logger.info("█"*50)

    all_metrics = [sarima_metrics, xgb_metrics, bayes_metrics]
    plot_model_comparison(all_metrics, TICKER)
    print_final_report(all_metrics, ab_result, TICKER)

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║        WEEKS 2 & 3 COMPLETE ✅           ║")
    logger.info("╚══════════════════════════════════════════╝")

    plots = list(PLOTS_DIR.glob(f"{TICKER.lower()}*.png"))
    logger.info(f"\nAll plots saved to {PLOTS_DIR}/")
    for p in sorted(plots):
        logger.info(f"  {p.name}")

    print(f"\n→ Next: Week 4 — FastAPI + Docker + MLflow deployment")


if __name__ == "__main__":
    main()