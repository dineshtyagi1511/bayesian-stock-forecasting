"""
src/model_xgboost.py — XGBoost with Walk-Forward Cross-Validation
Week 2, Step 2

Why XGBoost after SARIMA?
  - Uses all 39 engineered features vs SARIMA's univariate approach
  - Walk-forward CV shows model selection discipline
  - SHAP values provide model explainability (key for finance interviews)

Interview talking points:
  - "I use TimeSeriesSplit not KFold — random CV leaks future into training"
  - "Huber loss is more robust to outlier returns than MSE"
  - "SHAP shows which features drive predictions — regulators require this in finance"
  - "Early stopping prevents overfitting without manual tuning"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import pickle, warnings
from pathlib import Path
from loguru import logger
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PLOTS_DIR, MODELS_DIR, FORECAST_HORIZON, SEED
from src.metrics import evaluate, ModelMetrics
from src.splitter import DataSplit

warnings.filterwarnings("ignore")
plt.style.use("dark_background")
COLORS = {"primary": "#00d4ff", "accent": "#a78bfa", "pos": "#34d399", "neg": "#f87171", "neutral": "#94a3b8"}


# ── Walk-Forward CV ───────────────────────────────────────────────────────────

def walk_forward_cv_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
) -> dict:
    """
    TimeSeriesSplit cross-validation on training data.
    Returns mean RMSE and std across folds.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rmses = []

    logger.info(f"XGBoost walk-forward CV ({n_splits} folds)...")

    for fold, (tr_idx, vl_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_vl = X_train.iloc[tr_idx], X_train.iloc[vl_idx]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[vl_idx]

        model = _build_xgb()
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_vl, y_vl)],
            verbose=False,
        )
        preds = model.predict(X_vl)
        rmse  = np.sqrt(np.mean((y_vl - preds) ** 2))
        fold_rmses.append(rmse)
        logger.info(f"  Fold {fold+1}: RMSE={rmse:.6f}  "
                    f"train={X_tr.index[0].date()}→{X_tr.index[-1].date()} "
                    f"val={X_vl.index[0].date()}→{X_vl.index[-1].date()}")

    cv_result = {"mean_rmse": np.mean(fold_rmses), "std_rmse": np.std(fold_rmses)}
    logger.success(f"CV Result: RMSE = {cv_result['mean_rmse']:.6f} ± {cv_result['std_rmse']:.6f}")
    return cv_result


def _build_xgb() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,           # L1 regularisation
        reg_lambda=1.0,          # L2 regularisation
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=30,
        random_state=SEED,
        n_jobs=-1,
    )


# ── Final Model Training ──────────────────────────────────────────────────────

def train_xgboost(split: DataSplit) -> xgb.XGBRegressor:
    """Train final XGBoost on train+val, evaluate on test."""
    # Combine train + val for final model
    X_trainval = pd.concat([split.X_train, split.X_val])
    y_trainval = pd.concat([split.y_train, split.y_val])

    logger.info("Training final XGBoost on train+val...")
    model = _build_xgb()
    model.fit(
        X_trainval, y_trainval,
        eval_set=[(split.X_val, split.y_val)],
        verbose=False,
    )
    logger.success(f"Best iteration: {model.best_iteration}")

    # Save
    path = MODELS_DIR / "xgboost_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.success(f"Saved → {path}")
    return model


# ── Feature Importance (SHAP proxy via gain) ──────────────────────────────────

def plot_feature_importance(model: xgb.XGBRegressor, feature_names: list, ticker: str) -> None:
    importance = model.feature_importances_
    idx = np.argsort(importance)[-20:]   # top 20

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#080c14")
    ax.set_facecolor("#0a0f1a")

    bars = ax.barh(
        [feature_names[i] for i in idx],
        importance[idx],
        color=COLORS["primary"], alpha=0.8, edgecolor="#1e293b"
    )
    # Gradient colour by importance
    for bar, val in zip(bars, importance[idx]):
        bar.set_alpha(0.4 + 0.6 * (val / importance[idx].max()))

    ax.set_title(f"{ticker} — XGBoost Feature Importance (Top 20)", color="#f1f5f9", fontsize=12)
    ax.set_xlabel("Importance (Gain)", color=COLORS["neutral"])
    ax.tick_params(colors=COLORS["neutral"], labelsize=9)
    ax.spines[:].set_color("#1e293b")

    plt.tight_layout()
    path = PLOTS_DIR / f"{ticker.lower()}_xgb_feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#080c14")
    plt.close()
    logger.success(f"Saved → {path}")


# ── Results Plot ──────────────────────────────────────────────────────────────

def plot_xgb_results(
    split: DataSplit,
    y_pred_test: np.ndarray,
    ticker: str,
    metrics: ModelMetrics,
    cv_result: dict,
) -> None:
    y_true = split.y_test.values
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor("#080c14")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Pred vs actual over time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#0a0f1a")
    x = split.X_test.index[:len(y_true)]
    ax1.plot(x, y_true * 100,     color=COLORS["primary"], linewidth=1,   label="Actual", alpha=0.9)
    ax1.plot(x, y_pred_test * 100, color=COLORS["accent"], linewidth=1, linestyle="--", label="XGBoost", alpha=0.9)
    ax1.fill_between(x,
                     np.minimum(y_true, y_pred_test) * 100,
                     np.maximum(y_true, y_pred_test) * 100,
                     alpha=0.1, color=COLORS["neg"])
    ax1.set_title(f"{ticker} — XGBoost: Predicted vs Actual 5-day Returns\n"
                  f"CV RMSE={cv_result['mean_rmse']:.4f}±{cv_result['std_rmse']:.4f}  "
                  f"Test MAPE={metrics.mape:.2f}%  DA={metrics.da:.1f}%",
                  color="#f1f5f9", fontsize=11)
    ax1.set_ylabel("5-day Return %", color=COLORS["neutral"])
    ax1.legend(fontsize=10); ax1.tick_params(colors=COLORS["neutral"])
    ax1.spines[:].set_color("#1e293b")

    # 2. Cumulative strategy return
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#0a0f1a")
    positions   = np.sign(y_pred_test)
    strat_ret   = positions * y_true
    buy_hold    = y_true
    cum_strat   = (1 + strat_ret).cumprod() - 1
    cum_bh      = (1 + buy_hold).cumprod() - 1
    ax2.plot(x, cum_strat * 100, color=COLORS["pos"],     linewidth=1.5, label="Long/Short Strategy")
    ax2.plot(x, cum_bh    * 100, color=COLORS["neutral"],  linewidth=1,   label="Buy & Hold", alpha=0.7)
    ax2.set_title(f"Cumulative Return\nSharpe={metrics.sharpe:.2f}", color="#f1f5f9", fontsize=10)
    ax2.set_ylabel("Cumulative Return %", color=COLORS["neutral"])
    ax2.legend(fontsize=9); ax2.tick_params(colors=COLORS["neutral"])
    ax2.spines[:].set_color("#1e293b")

    # 3. Error distribution
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#0a0f1a")
    errors = (y_pred_test - y_true) * 100
    ax3.hist(errors, bins=50, color=COLORS["accent"], alpha=0.7, edgecolor="#1e293b")
    ax3.axvline(0,            color=COLORS["neg"],    linewidth=2, linestyle="--")
    ax3.axvline(errors.mean(), color=COLORS["primary"], linewidth=1.5, label=f"Mean={errors.mean():.3f}%")
    ax3.set_title(f"Prediction Error Distribution\nRMSE={metrics.rmse:.4f}  MAE={metrics.mae:.4f}",
                  color="#f1f5f9", fontsize=10)
    ax3.set_xlabel("Error %", color=COLORS["neutral"])
    ax3.legend(fontsize=9); ax3.tick_params(colors=COLORS["neutral"])
    ax3.spines[:].set_color("#1e293b")

    plt.suptitle(f"XGBoost Model — {ticker}", color="#f1f5f9", fontsize=14, y=1.01)
    path = PLOTS_DIR / f"{ticker.lower()}_xgb_results.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#080c14")
    plt.close()
    logger.success(f"Saved → {path}")


def run_xgboost(split: DataSplit, ticker: str) -> ModelMetrics:
    logger.info(f"\n{'='*50}\nXGBoost — {ticker}\n{'='*50}")

    cv_result = walk_forward_cv_xgb(split.X_train, split.y_train)
    model     = train_xgboost(split)

    y_pred = model.predict(split.X_test)
    y_true = split.y_test.values
    metrics = evaluate(y_true, y_pred, "XGBoost")

    plot_feature_importance(model, split.feature_cols, ticker)
    plot_xgb_results(split, y_pred, ticker, metrics, cv_result)

    logger.success(f"XGBoost done: {metrics}")
    return metrics