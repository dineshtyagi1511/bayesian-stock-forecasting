"""
app/schemas.py — Pydantic request / response schemas
All API data contracts live here. Validated automatically by FastAPI.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal
from datetime import date


# ── Request Models ────────────────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    ticker: Literal["NIFTY50", "SP500", "RELIANCE"] = Field(
        default="NIFTY50",
        description="Ticker to forecast"
    )
    model: Literal["xgboost", "sarima", "bayesian", "ensemble"] = Field(
        default="xgboost",
        description="Model to use for forecasting"
    )
    horizon_days: int = Field(
        default=5,
        ge=1, le=21,
        description="Forecast horizon in trading days (1–21)"
    )
    include_uncertainty: bool = Field(
        default=True,
        description="Return Bayesian credible intervals (only for bayesian/ensemble)"
    )

    model_config = {"json_schema_extra": {
        "example": {
            "ticker": "NIFTY50",
            "model": "xgboost",
            "horizon_days": 5,
            "include_uncertainty": True
        }
    }}


class MetricsRequest(BaseModel):
    ticker: Literal["NIFTY50", "SP500", "RELIANCE"] = "NIFTY50"


# ── Response Models ───────────────────────────────────────────────────────────

class ForecastResponse(BaseModel):
    ticker: str
    model_used: str
    horizon_days: int
    predicted_return_pct: float = Field(description="Predicted 5-day return %")
    direction: Literal["UP", "DOWN", "FLAT"]
    confidence: float = Field(description="Model confidence 0–1 (directional accuracy on val set)")

    # Bayesian uncertainty (None for non-Bayesian models)
    lower_95_pct: float | None = Field(default=None, description="95% credible interval lower bound %")
    upper_95_pct: float | None = Field(default=None, description="95% credible interval upper bound %")
    interval_width_pct: float | None = Field(default=None, description="CI width — proxy for uncertainty")

    model_version: str
    mlflow_run_id: str | None = None

    model_config = {"json_schema_extra": {
        "example": {
            "ticker": "NIFTY50",
            "model_used": "xgboost",
            "horizon_days": 5,
            "predicted_return_pct": 1.23,
            "direction": "UP",
            "confidence": 0.534,
            "lower_95_pct": -2.1,
            "upper_95_pct": 4.5,
            "interval_width_pct": 6.6,
            "model_version": "1.0.0",
            "mlflow_run_id": None
        }
    }}


class ModelMetricsResponse(BaseModel):
    ticker: str
    models: dict[str, dict]    # model_name → {rmse, mae, mape, da, sharpe}
    best_model_by_sharpe: str
    best_model_by_rmse: str


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    models_loaded: list[str]
    version: str
    uptime_seconds: float


class ABTestResponse(BaseModel):
    ticker: str
    strategy_a: str
    strategy_b: str
    prob_b_beats_a: float
    verdict: str
    sharpe_a: float
    sharpe_b: float
    ci_a: list[float]
    ci_b: list[float]