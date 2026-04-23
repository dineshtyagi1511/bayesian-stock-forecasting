"""
src/metrics.py — Shared Evaluation Metrics
Used by SARIMA, XGBoost, LSTM, and Bayesian models for apples-to-apples comparison.

Metrics:
  RMSE  — Root Mean Squared Error (penalises large errors)
  MAE   — Mean Absolute Error (robust to outliers)
  MAPE  — Mean Absolute Percentage Error (interpretable %)
  DA    — Directional Accuracy (did we get up/down right?)
  Sharpe— Sharpe ratio of a simple long/short strategy
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class ModelMetrics:
    model_name: str
    rmse:  float
    mae:   float
    mape:  float
    da:    float   # directional accuracy %
    sharpe: float  # annualised Sharpe of long/short

    def __str__(self) -> str:
        return (
            f"{self.model_name:<20} "
            f"RMSE={self.rmse:.6f}  "
            f"MAE={self.mae:.6f}  "
            f"MAPE={self.mape:.2f}%  "
            f"DA={self.da:.1f}%  "
            f"Sharpe={self.sharpe:.2f}"
        )


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> ModelMetrics:
    """
    Compute all regression + trading metrics.
    y_true / y_pred: forward return series (e.g. 5-day pct change)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Remove NaNs
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae  = np.mean(np.abs(y_true - y_pred))

    # MAPE — avoid division by zero
    nonzero = y_true != 0
    mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100

    # Directional accuracy
    da = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100

    # Sharpe: go long if pred>0, short if pred<0; trade at true return
    positions = np.sign(y_pred)
    strat_returns = positions * y_true
    ann_factor = np.sqrt(252 / 5)  # 5-day horizon
    sharpe = (strat_returns.mean() / (strat_returns.std() + 1e-9)) * ann_factor

    return ModelMetrics(
        model_name=model_name,
        rmse=rmse, mae=mae, mape=mape, da=da, sharpe=sharpe,
    )


def comparison_table(metrics_list: list[ModelMetrics]) -> pd.DataFrame:
    """Return a tidy DataFrame for README / MLflow logging."""
    rows = []
    for m in metrics_list:
        rows.append({
            "Model":  m.model_name,
            "RMSE":   round(m.rmse,  6),
            "MAE":    round(m.mae,   6),
            "MAPE %": round(m.mape,  2),
            "Dir.Acc %": round(m.da, 1),
            "Sharpe": round(m.sharpe, 2),
        })
    df = pd.DataFrame(rows).set_index("Model")
    return df