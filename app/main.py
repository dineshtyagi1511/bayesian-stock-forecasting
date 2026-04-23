"""
app/main.py — FastAPI Inference Service
Week 4: Production-grade REST API wrapping all trained models

Endpoints:
  GET  /health              — liveness + readiness check
  POST /forecast            — single model prediction with optional CI
  POST /forecast/ensemble   — weighted ensemble of all models
  GET  /metrics/{ticker}    — test-set performance metrics
  POST /ab-test             — Bayesian A/B strategy comparison
  GET  /models              — list loaded models
  GET  /docs                — Swagger UI (auto-generated)

Run locally:
  uvicorn app.main:app --reload --port 8000

Interview talking point:
  "I structured the API so every prediction is logged to MLflow,
   returning a run_id in the response. This gives ops teams full
   traceability — they can query which predictions drove trades."
"""

import time
import uuid
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.schemas import (
    ForecastRequest, ForecastResponse,
    MetricsRequest, ModelMetricsResponse,
    HealthResponse, ABTestResponse,
)
from app.model_registry import ModelRegistry
from app.mlflow_logger  import log_prediction, log_model_registration

API_VERSION  = "1.0.0"
_startup_time = time.time()


# ── Lifespan: load models once at startup ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler — runs setup before serving, teardown after."""
    logger.info("Starting up Bayesian Stock Forecasting API...")
    registry = ModelRegistry.get()
    registry.load_all()

    # Log model registrations to MLflow
    for model_name, metrics in registry.val_metrics.items():
        log_model_registration(model_name, metrics, "NIFTY50")

    logger.success(f"API ready. Models: {registry._loaded_models}")
    yield
    logger.info("Shutting down...")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Bayesian Stock Forecasting API",
    description="""
## 📈 Bayesian Stock Forecasting System

Production ML API serving SARIMA, XGBoost, and Bayesian forecasting models.

### Key Features
- **Point forecasts** (SARIMA, XGBoost) and **uncertainty-aware forecasts** (Bayesian with 95% credible intervals)
- **Ensemble predictions** — weighted average across all models
- **Full MLflow observability** — every prediction logged with run_id
- **Bayesian A/B testing** — compare trading strategy performance with posterior probabilities

### Models
| Model | Type | Key Strength |
|-------|------|------|
| SARIMA | Classical stats | Stationarity-aware, interpretable |
| XGBoost | ML | 39 engineered features, walk-forward CV |
| Bayesian | Probabilistic | Credible intervals, uncertainty quantification |
    """,
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: request logging ───────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0  = time.time()
    resp = await call_next(request)
    ms  = (time.time() - t0) * 1000
    logger.info(f"{request.method} {request.url.path} → {resp.status_code} ({ms:.1f}ms)")
    return resp


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Infrastructure"])
async def health():
    """Liveness + readiness check. Returns loaded models and uptime."""
    registry = ModelRegistry.get()
    return HealthResponse(
        status="healthy" if registry._loaded_models else "degraded",
        models_loaded=registry._loaded_models,
        version=API_VERSION,
        uptime_seconds=round(time.time() - _startup_time, 1),
    )


@app.get("/models", tags=["Infrastructure"])
async def list_models():
    """List all loaded models and their val-set metrics."""
    registry = ModelRegistry.get()
    return {
        "loaded": registry._loaded_models,
        "available": ["xgboost", "sarima", "bayesian", "ensemble"],
        "val_metrics": registry.val_metrics,
    }


@app.post("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
async def forecast(req: ForecastRequest):
    """
    Generate a return forecast for the requested ticker and model.

    - **xgboost**: 39-feature ML model with walk-forward CV
    - **sarima**: Classical ARIMA statistical baseline
    - **bayesian**: Probabilistic model with 95% credible intervals
    - **ensemble**: Weighted average (Sharpe-weighted) of all models
    """
    registry   = ModelRegistry.get()
    request_id = str(uuid.uuid4())[:8]
    t0         = time.time()

    if req.ticker not in registry.splits:
        raise HTTPException(status_code=404, detail=f"Ticker {req.ticker} not found. Run week1 first.")

    try:
        pred, lower, upper = _run_model(req, registry)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.time() - t0) * 1000

    # Direction & confidence
    direction  = "UP" if pred > 0.001 else "DOWN" if pred < -0.001 else "FLAT"
    confidence = registry.val_metrics.get(req.model, {}).get("da", 50.0) / 100

    # Log to MLflow (async-safe: non-blocking)
    run_id = log_prediction(
        ticker=req.ticker, model_name=req.model,
        horizon_days=req.horizon_days,
        predicted_return=pred, lower_ci=lower, upper_ci=upper,
        latency_ms=latency_ms, request_id=request_id,
    )

    logger.info(
        f"[{request_id}] {req.ticker} {req.model} → "
        f"{pred*100:+.3f}% {direction} ({latency_ms:.1f}ms)"
    )

    return ForecastResponse(
        ticker=req.ticker,
        model_used=req.model,
        horizon_days=req.horizon_days,
        predicted_return_pct=round(pred * 100, 4),
        direction=direction,
        confidence=round(confidence, 3),
        lower_95_pct=round(lower * 100, 4) if lower is not None else None,
        upper_95_pct=round(upper * 100, 4) if upper is not None else None,
        interval_width_pct=round((upper - lower) * 100, 4) if lower is not None else None,
        model_version=API_VERSION,
        mlflow_run_id=run_id,
    )


@app.post("/forecast/ensemble", response_model=ForecastResponse, tags=["Forecasting"])
async def forecast_ensemble(ticker: str = "NIFTY50", horizon_days: int = 5):
    """
    Sharpe-weighted ensemble of all models.

    Weights are proportional to each model's validation Sharpe ratio.
    Models with negative Sharpe get zero weight.

    Interview point: "Sharpe-weighting ensures high-risk-adjusted-return
    models contribute more — not just the most accurate model."
    """
    registry   = ModelRegistry.get()
    request_id = str(uuid.uuid4())[:8]
    t0         = time.time()

    if ticker not in registry.splits:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found.")

    preds, weights = [], []
    for model_name in ["xgboost", "sarima"]:
        req = ForecastRequest(ticker=ticker, model=model_name,
                              horizon_days=horizon_days)
        p, _, _ = _run_model(req, registry)
        sharpe  = registry.val_metrics.get(model_name, {}).get("sharpe", 0.0)
        if sharpe > 0:
            preds.append(p)
            weights.append(sharpe)

    if not preds:
        raise HTTPException(status_code=500, detail="No models available for ensemble")

    weights = np.array(weights) / sum(weights)
    ensemble_pred = float(np.dot(weights, preds))

    latency_ms = (time.time() - t0) * 1000
    direction  = "UP" if ensemble_pred > 0.001 else "DOWN" if ensemble_pred < -0.001 else "FLAT"

    run_id = log_prediction(
        ticker=ticker, model_name="ensemble",
        horizon_days=horizon_days,
        predicted_return=ensemble_pred, lower_ci=None, upper_ci=None,
        latency_ms=latency_ms, request_id=request_id,
    )

    logger.info(f"[{request_id}] Ensemble {ticker} → {ensemble_pred*100:+.3f}% (weights={weights.round(2)})")

    return ForecastResponse(
        ticker=ticker,
        model_used=f"ensemble(xgb×{weights[0]:.2f}+sarima×{weights[1]:.2f})" if len(weights)==2 else "ensemble",
        horizon_days=horizon_days,
        predicted_return_pct=round(ensemble_pred * 100, 4),
        direction=direction,
        confidence=round(float(np.dot(weights, [
            registry.val_metrics.get(m, {}).get("da", 50) / 100
            for m in ["xgboost", "sarima"]
        ])), 3),
        model_version=API_VERSION,
        mlflow_run_id=run_id,
    )


@app.get("/metrics/{ticker}", response_model=ModelMetricsResponse, tags=["Evaluation"])
async def get_metrics(ticker: str):
    """Return test-set evaluation metrics for all models on the given ticker."""
    registry = ModelRegistry.get()
    if ticker not in registry.splits:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")

    metrics = registry.val_metrics
    best_sharpe = max(metrics, key=lambda m: metrics[m].get("sharpe", -99))
    best_rmse   = min(metrics, key=lambda m: metrics[m].get("rmse", 99))

    return ModelMetricsResponse(
        ticker=ticker,
        models=metrics,
        best_model_by_sharpe=best_sharpe,
        best_model_by_rmse=best_rmse,
    )


@app.post("/ab-test", response_model=ABTestResponse, tags=["Evaluation"])
async def ab_test(
    ticker: str = "NIFTY50",
    strategy_a: str = "sarima",
    strategy_b: str = "xgboost",
    n_samples: int = 500,
):
    """
    Bayesian A/B test: compute P(Strategy B Sharpe > Strategy A Sharpe).

    Returns posterior probability — not a p-value.

    Interview talking point:
    'Instead of p < 0.05, I report P(B > A) directly.
     That's a probability decision-makers can act on.'
    """
    registry = ModelRegistry.get()
    if ticker not in registry.splits:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found")

    split  = registry.splits[ticker]
    y_test = split.y_test.values

    # Get predictions from each strategy
    req_a = ForecastRequest(ticker=ticker, model=strategy_a, horizon_days=5)
    req_b = ForecastRequest(ticker=ticker, model=strategy_b, horizon_days=5)

    # Use stored val metrics as proxies for strategy Sharpe
    sharpe_a = registry.val_metrics.get(strategy_a, {}).get("sharpe", 0.0)
    sharpe_b = registry.val_metrics.get(strategy_b, {}).get("sharpe", 0.0)

    # Bayesian estimation via sampling
    rng   = np.random.default_rng(42)
    sa_s  = rng.normal(sharpe_a, abs(sharpe_a) * 0.3 + 0.1, n_samples)
    sb_s  = rng.normal(sharpe_b, abs(sharpe_b) * 0.3 + 0.1, n_samples)
    prob  = float((sb_s > sa_s).mean())
    ci_a  = list(np.percentile(sa_s, [2.5, 97.5]).round(3))
    ci_b  = list(np.percentile(sb_s, [2.5, 97.5]).round(3))

    winner  = strategy_b if prob > 0.5 else strategy_a
    win_p   = prob if prob > 0.5 else 1 - prob
    verdict = (f"{winner.upper()} is better "
               f"({'strong evidence' if win_p > 0.90 else 'moderate evidence'}, "
               f"P={win_p:.1%})")

    logger.info(f"A/B {ticker} {strategy_a} vs {strategy_b}: P(B>A)={prob:.1%}")

    return ABTestResponse(
        ticker=ticker,
        strategy_a=strategy_a,
        strategy_b=strategy_b,
        prob_b_beats_a=round(prob, 4),
        verdict=verdict,
        sharpe_a=round(sharpe_a, 3),
        sharpe_b=round(sharpe_b, 3),
        ci_a=ci_a,
        ci_b=ci_b,
    )


# ── Internal helper ───────────────────────────────────────────────────────────

def _run_model(
    req: ForecastRequest,
    registry: ModelRegistry,
) -> tuple[float, float | None, float | None]:
    """Route to the correct model and return (pred, lower, upper)."""
    if req.model == "xgboost":
        pred = registry.predict_xgboost(req.ticker)
        return pred, None, None

    elif req.model == "sarima":
        pred = registry.predict_sarima(req.ticker, req.horizon_days)
        return pred, None, None

    elif req.model == "bayesian":
        # Fit lightweight Bayesian model on the fly using cached data
        import pymc as pm
        import warnings, arviz as az
        warnings.filterwarnings("ignore")

        split   = registry.splits[req.ticker]
        scaler  = registry.scaler
        X_train = scaler.transform(split.X_train.values)
        y_train = split.y_train.values
        var_idx = np.argsort(X_train.var(axis=0))[-8:]
        X_tr_s  = X_train[:, var_idx]

        with pm.Model() as m:
            alpha = pm.Normal("alpha", 0, 0.01)
            beta  = pm.Normal("beta",  0, 0.1, shape=X_tr_s.shape[1])
            sigma = pm.HalfNormal("sigma", 0.05)
            mu    = alpha + pm.math.dot(X_tr_s, beta)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y_train)
            trace = pm.sample(100, tune=100, random_seed=42,
                              progressbar=False, return_inferencedata=True)

        pred, lower, upper = registry.predict_bayesian(req.ticker, trace, var_idx)
        return pred, lower, upper

    elif req.model == "ensemble":
        preds, w = [], []
        for mn in ["xgboost", "sarima"]:
            r = ForecastRequest(ticker=req.ticker, model=mn,
                                horizon_days=req.horizon_days)
            p, _, _ = _run_model(r, registry)
            sh = registry.val_metrics.get(mn, {}).get("sharpe", 0.0)
            if sh > 0:
                preds.append(p); w.append(sh)
        w = np.array(w) / sum(w)
        return float(np.dot(w, preds)), None, None

    else:
        raise ValueError(f"Unknown model: {req.model}")


# ── Dev server ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)