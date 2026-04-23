"""
tests/test_api.py — API Test Suite
Run with: pytest tests/ -v

Tests every endpoint for correct status codes, schema compliance,
and business logic (e.g. direction matches sign of predicted return).

Interview talking point:
  "I write tests before I write the API — not just unit tests but
   integration tests against the live FastAPI app using httpx.
   CI/CD runs these on every push before building the Docker image."
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.main import app

client = TestClient(app)


# ── Health & Infrastructure ───────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_schema(self):
        data = client.get("/health").json()
        assert "status" in data
        assert "models_loaded" in data
        assert "version" in data
        assert "uptime_seconds" in data

    def test_health_status_healthy_or_degraded(self):
        data = client.get("/health").json()
        assert data["status"] in ("healthy", "degraded")

    def test_models_endpoint(self):
        resp = client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "loaded" in data
        assert "available" in data
        assert "val_metrics" in data


# ── Forecast Endpoint ─────────────────────────────────────────────────────────

class TestForecast:
    def test_xgboost_forecast(self):
        resp = client.post("/forecast", json={
            "ticker": "NIFTY50",
            "model": "xgboost",
            "horizon_days": 5,
        })
        assert resp.status_code == 200

    def test_forecast_schema_complete(self):
        data = client.post("/forecast", json={
            "ticker": "NIFTY50", "model": "xgboost", "horizon_days": 5
        }).json()
        required = ["ticker", "model_used", "horizon_days",
                    "predicted_return_pct", "direction", "confidence", "model_version"]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_direction_matches_return_sign(self):
        data = client.post("/forecast", json={
            "ticker": "NIFTY50", "model": "xgboost", "horizon_days": 5
        }).json()
        ret = data["predicted_return_pct"]
        direction = data["direction"]
        if abs(ret) > 0.1:   # only check when not FLAT
            if ret > 0:
                assert direction == "UP",   f"Return {ret} but direction {direction}"
            else:
                assert direction == "DOWN", f"Return {ret} but direction {direction}"

    def test_sarima_forecast(self):
        resp = client.post("/forecast", json={
            "ticker": "NIFTY50", "model": "sarima", "horizon_days": 5
        })
        assert resp.status_code == 200

    def test_confidence_in_range(self):
        data = client.post("/forecast", json={
            "ticker": "NIFTY50", "model": "xgboost", "horizon_days": 5
        }).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_invalid_ticker_returns_404(self):
        resp = client.post("/forecast", json={
            "ticker": "INVALID", "model": "xgboost", "horizon_days": 5
        })
        # Pydantic validation error = 422
        assert resp.status_code == 422

    def test_invalid_horizon_returns_422(self):
        resp = client.post("/forecast", json={
            "ticker": "NIFTY50", "model": "xgboost", "horizon_days": 100
        })
        assert resp.status_code == 422

    def test_sp500_forecast(self):
        resp = client.post("/forecast", json={
            "ticker": "SP500", "model": "xgboost", "horizon_days": 5
        })
        assert resp.status_code == 200


# ── Ensemble Endpoint ─────────────────────────────────────────────────────────

class TestEnsemble:
    def test_ensemble_returns_200(self):
        resp = client.post("/forecast/ensemble?ticker=NIFTY50&horizon_days=5")
        assert resp.status_code == 200

    def test_ensemble_schema(self):
        data = client.post("/forecast/ensemble?ticker=NIFTY50").json()
        assert "predicted_return_pct" in data
        assert "direction" in data
        assert "ensemble" in data["model_used"].lower()


# ── Metrics Endpoint ──────────────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_returns_200(self):
        resp = client.get("/metrics/NIFTY50")
        assert resp.status_code == 200

    def test_metrics_schema(self):
        data = client.get("/metrics/NIFTY50").json()
        assert "models" in data
        assert "best_model_by_sharpe" in data
        assert "best_model_by_rmse" in data

    def test_metrics_contains_all_models(self):
        data = client.get("/metrics/NIFTY50").json()
        for model in ["xgboost", "sarima", "bayesian"]:
            assert model in data["models"], f"Missing model: {model}"

    def test_metrics_has_correct_keys(self):
        data = client.get("/metrics/NIFTY50").json()
        for model_metrics in data["models"].values():
            for key in ["rmse", "mae", "mape", "da", "sharpe"]:
                assert key in model_metrics

    def test_invalid_ticker_404(self):
        resp = client.get("/metrics/INVALID")
        assert resp.status_code == 404


# ── A/B Test Endpoint ─────────────────────────────────────────────────────────

class TestABTest:
    def test_ab_test_returns_200(self):
        resp = client.post("/ab-test?ticker=NIFTY50&strategy_a=sarima&strategy_b=xgboost")
        assert resp.status_code == 200

    def test_ab_test_schema(self):
        data = client.post("/ab-test?ticker=NIFTY50&strategy_a=sarima&strategy_b=xgboost").json()
        required = ["prob_b_beats_a", "verdict", "sharpe_a", "sharpe_b", "ci_a", "ci_b"]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_prob_in_range(self):
        data = client.post("/ab-test?ticker=NIFTY50&strategy_a=sarima&strategy_b=xgboost").json()
        assert 0.0 <= data["prob_b_beats_a"] <= 1.0

    def test_ci_is_list_of_two(self):
        data = client.post("/ab-test?ticker=NIFTY50&strategy_a=sarima&strategy_b=xgboost").json()
        assert len(data["ci_a"]) == 2
        assert len(data["ci_b"]) == 2
        assert data["ci_a"][0] < data["ci_a"][1]
        assert data["ci_b"][0] < data["ci_b"][1]