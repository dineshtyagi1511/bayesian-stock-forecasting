"""
app/mlflow_logger.py — MLflow Experiment Tracking
Logs every API prediction call as an MLflow run for full observability.

Interview talking point:
  "Every prediction is logged to MLflow — ticker, model, predicted return,
   confidence, latency. This gives me a complete audit trail and lets me
   detect model drift over time by comparing live predictions to test metrics."
"""

import time
import mlflow
import mlflow.sklearn
from pathlib import Path
from loguru import logger
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Point MLflow at local tracking store
MLFLOW_TRACKING_URI = f"file:///{(ROOT / 'mlruns').as_posix()}"
EXPERIMENT_NAME     = "bayesian-stock-forecasting-api"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def get_or_create_experiment() -> str:
    """Get existing experiment or create new one."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        exp_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            tags={"project": "bayesian-stock-forecasting", "version": "1.0.0"}
        )
        logger.info(f"Created MLflow experiment: {EXPERIMENT_NAME} (id={exp_id})")
        return exp_id
    return exp.experiment_id


def log_prediction(
    ticker: str,
    model_name: str,
    horizon_days: int,
    predicted_return: float,
    lower_ci: float | None,
    upper_ci: float | None,
    latency_ms: float,
    request_id: str,
) -> str | None:
    try:
        exp_id = get_or_create_experiment()

        with mlflow.start_run(
            experiment_id=exp_id,
            run_name=f"{model_name}_{ticker}"
        ) as run:

            # ✅ ALWAYS set tags inside run
            mlflow.set_tag("model_version", "v1")

            # Parameters
            mlflow.log_params({
                "ticker": ticker,
                "model": model_name,
                "horizon_days": horizon_days,
                "request_id": request_id,
            })

            # Metrics
            mlflow.log_metrics({
                "predicted_return_pct": predicted_return * 100,
                "latency_ms": latency_ms,
            })

            if lower_ci is not None:
                mlflow.log_metrics({
                    "lower_95_pct": lower_ci * 100,
                    "upper_95_pct": upper_ci * 100,
                    "interval_width_pct": (upper_ci - lower_ci) * 100,
                })

            # Additional tags
            mlflow.set_tags({
                "direction": "UP" if predicted_return > 0.001 else
                             "DOWN" if predicted_return < -0.001 else "FLAT",
                "source": "api",
            })

            return run.info.run_id

    except Exception as e:
        logger.warning(f"MLflow logging failed (non-fatal): {e}")
        return None


def log_model_registration(
    model_name: str,
    metrics: dict,
    ticker: str,
) -> None:
    """Log a model's test-set metrics to MLflow at registration time."""
    try:
        exp_id = get_or_create_experiment()
        with mlflow.start_run(
            experiment_id=exp_id,
            run_name=f"registration_{model_name}_{ticker}"
        ):
            mlflow.log_params({"model": model_name, "ticker": ticker, "stage": "registration"})
            mlflow.log_metrics(metrics)
            mlflow.set_tag("type", "model_registration")
        logger.info(f"Registered {model_name} metrics in MLflow")
    except Exception as e:
        logger.warning(f"MLflow registration log failed: {e}")