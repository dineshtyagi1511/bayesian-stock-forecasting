"""
src/model_sarima.py — SARIMA Classical Statistical Baseline
Week 2, Step 1

Why SARIMA first?
  - Proves you understand the statistical foundations before jumping to ML
  - Serves as the baseline every other model must beat
  - Uses your M.Sc. knowledge: stationarity, ACF/PACF, information criteria

Interview talking points:
  - "I always start with a statistical baseline — if XGBoost can't beat ARIMA,
     something is wrong with my feature engineering"
  - "I use AIC (not BIC) for order selection because it penalises complexity
     less aggressively on financial data with weak signals"
  - "I test residuals for autocorrelation (Ljung-Box) to validate model fit"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
from pathlib import Path
from loguru import logger
import sys, pickle

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PLOTS_DIR, MODELS_DIR, FORECAST_HORIZON, SEED
from src.metrics import evaluate, ModelMetrics

warnings.filterwarnings("ignore")
plt.style.use("dark_background")
COLORS = {"primary": "#00d4ff", "accent": "#a78bfa", "pos": "#34d399", "neg": "#f87171", "neutral": "#94a3b8"}


# ── SARIMA Fitting ────────────────────────────────────────────────────────────

def fit_sarima(
    train_prices: pd.Series,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 0, 1, 5),   # weekly seasonality (5 trading days)
) -> SARIMAX:
    """
    Fit SARIMA on log-price series.
    We model log(price) so that predictions are always positive.
    order         = (p, d, q)   — AR, differencing, MA
    seasonal_order= (P, D, Q, s)— seasonal components, s=5 for weekly
    """
    log_prices = np.log(train_prices)

    logger.info(f"Fitting SARIMA{order}x{seasonal_order}...")
    model = SARIMAX(
        log_prices,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False, maxiter=200)

    logger.info(f"  AIC={result.aic:.2f}  BIC={result.bic:.2f}  "
                f"LogLik={result.llf:.2f}")

    # Ljung-Box residual test (H0: no autocorrelation in residuals)
    lb = acorr_ljungbox(result.resid, lags=[10, 20], return_df=True)
    logger.info(f"  Ljung-Box p-values: {lb['lb_pvalue'].values.round(4).tolist()}")
    if (lb['lb_pvalue'] < 0.05).any():
        logger.warning("  Residuals show autocorrelation — model may be mis-specified")
    else:
        logger.success("  Residuals pass Ljung-Box test ✅")

    # Save model
    path = MODELS_DIR / "sarima_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(result, f)
    logger.success(f"  Saved → {path}")

    return result


# ── Walk-Forward Forecasting ──────────────────────────────────────────────────

def walk_forward_sarima(
    train_prices: pd.Series,
    test_prices:  pd.Series,
    horizon: int = FORECAST_HORIZON,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 0, 1, 5),
    step: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk-forward prediction: refit model every `step` days using expanding window.
    This is the honest evaluation method for time-series.

    Returns (y_true, y_pred) as forward return arrays.
    """
    logger.info(f"Walk-forward SARIMA — {len(test_prices)} test points...")

    all_prices = pd.concat([train_prices, test_prices])
    log_all    = np.log(all_prices)

    train_end_idx = len(train_prices)
    preds, trues  = [], []

    for i in range(0, len(test_prices) - horizon, step):
        current_idx = train_end_idx + i

        # Expanding window: use all data up to current point
        history = log_all.iloc[:current_idx]

        try:
            m = SARIMAX(history, order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
            res = m.fit(disp=False, maxiter=100)

            # Forecast horizon steps ahead (in log space)
            fc = res.forecast(steps=horizon)
            pred_log_price = fc.iloc[-1]
            curr_log_price = history.iloc[-1]

            pred_return = np.exp(pred_log_price - curr_log_price) - 1
            true_return = (test_prices.iloc[i + horizon] / test_prices.iloc[i]) - 1

            preds.append(pred_return)
            trues.append(true_return)

        except Exception as e:
            logger.warning(f"  SARIMA failed at step {i}: {e}")
            continue

        if i % 50 == 0:
            logger.info(f"  Progress: {i}/{len(test_prices) - horizon}")

    logger.success(f"  Walk-forward complete: {len(preds)} predictions")
    return np.array(trues), np.array(preds)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_sarima_results(
    train_prices: pd.Series,
    test_prices:  pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ticker: str,
    metrics: ModelMetrics,
) -> None:
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor("#080c14")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Full price history + test region highlight
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#0a0f1a")
    ax1.plot(train_prices.index, train_prices, color=COLORS["neutral"], linewidth=0.8, label="Train")
    ax1.plot(test_prices.index,  test_prices,  color=COLORS["primary"], linewidth=1.2, label="Test")
    ax1.axvline(test_prices.index[0], color=COLORS["accent"], linestyle="--", alpha=0.7, label="Test start")
    ax1.set_title(f"{ticker} — Price History with Train/Test Split", color="#f1f5f9", fontsize=12)
    ax1.legend(fontsize=9); ax1.tick_params(colors=COLORS["neutral"])
    ax1.spines[:].set_color("#1e293b")

    # 2. Predicted vs actual returns
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#0a0f1a")
    x = np.arange(len(y_true))
    ax2.plot(x, y_true * 100, color=COLORS["primary"],  linewidth=1,   label="Actual", alpha=0.9)
    ax2.plot(x, y_pred * 100, color=COLORS["accent"], linewidth=1, label="SARIMA", alpha=0.9, linestyle="--")
    ax2.set_title(f"SARIMA: Predicted vs Actual Returns\nMAPE={metrics.mape:.2f}%  DA={metrics.da:.1f}%",
                  color="#f1f5f9", fontsize=10)
    ax2.set_ylabel("5-day Return %", color=COLORS["neutral"])
    ax2.legend(fontsize=9); ax2.tick_params(colors=COLORS["neutral"])
    ax2.spines[:].set_color("#1e293b")

    # 3. Scatter: pred vs actual
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#0a0f1a")
    ax3.scatter(y_pred * 100, y_true * 100, color=COLORS["primary"], alpha=0.4, s=12)
    lim = max(abs(y_true).max(), abs(y_pred).max()) * 100 * 1.1
    ax3.plot([-lim, lim], [-lim, lim], color=COLORS["neg"], linewidth=1, linestyle="--", label="Perfect")
    ax3.axhline(0, color=COLORS["neutral"], linewidth=0.5, alpha=0.5)
    ax3.axvline(0, color=COLORS["neutral"], linewidth=0.5, alpha=0.5)
    ax3.set_xlabel("Predicted Return %", color=COLORS["neutral"])
    ax3.set_ylabel("Actual Return %",    color=COLORS["neutral"])
    ax3.set_title(f"Prediction Scatter\nRMSE={metrics.rmse:.4f}  Sharpe={metrics.sharpe:.2f}",
                  color="#f1f5f9", fontsize=10)
    ax3.legend(fontsize=9); ax3.tick_params(colors=COLORS["neutral"])
    ax3.spines[:].set_color("#1e293b")

    plt.suptitle(f"SARIMA Model — {ticker}", color="#f1f5f9", fontsize=14, y=1.01)
    path = PLOTS_DIR / f"{ticker.lower()}_sarima_results.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#080c14")
    plt.close()
    logger.success(f"Saved → {path}")


def run_sarima(split, ticker: str) -> ModelMetrics:
    logger.info(f"\n{'='*50}\nSARIMA — {ticker}\n{'='*50}")
    fit_sarima(split.train_prices)
    y_true, y_pred = walk_forward_sarima(split.train_prices, split.test_prices)
    metrics = evaluate(y_true, y_pred, "SARIMA")
    plot_sarima_results(split.train_prices, split.test_prices, y_true, y_pred, ticker, metrics)
    logger.success(f"SARIMA done: {metrics}")
    return metrics