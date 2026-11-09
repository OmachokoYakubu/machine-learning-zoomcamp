"""Integration tests for the BikePulse FastAPI service (`predict.py`)."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from predict import app  # noqa: E402

client = TestClient(app)


def test_health_and_dashboard() -> None:
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert len(body["model_sha256"]) == 64

    ui = client.get("/")
    assert ui.status_code == 200
    assert "BikePulse Operations Console" in ui.text


def test_single_prediction_commuter_vs_night() -> None:
    rush_res = client.post(
        "/predict",
        json={
            "season": 3,
            "yr": 1,
            "mnth": 9,
            "hr": 17,
            "holiday": 0,
            "weekday": 3,
            "workingday": 1,
            "weathersit": 1,
            "temp_celsius": 24.0,
            "humidity_pct": 50.0,
            "windspeed_kmh": 11.0,
        },
    )
    assert rush_res.status_code == 200
    rush = rush_res.json()

    night_res = client.post(
        "/predict",
        json={
            "season": 1,
            "yr": 1,
            "mnth": 1,
            "hr": 3,
            "holiday": 0,
            "weekday": 2,
            "workingday": 1,
            "weathersit": 3,
            "temp": 0.12,
            "atemp": 0.12,
            "hum": 0.90,
            "windspeed": 0.40,
        },
    )
    assert night_res.status_code == 200
    night = night_res.json()

    assert rush["predicted_hourly_rentals"] > 300.0
    assert night["predicted_hourly_rentals"] < 50.0
    assert rush["predicted_hourly_rentals"] > night["predicted_hourly_rentals"]


def test_batch_forecast_and_validation() -> None:
    batch_res = client.post(
        "/predict/batch",
        json={
            "hours": [
                {"hr": 3, "mnth": 9, "season": 3, "weekday": 3, "weathersit": 1, "temp": 0.50},
                {"hr": 8, "mnth": 9, "season": 3, "weekday": 3, "weathersit": 1, "temp": 0.56},
                {"hr": 17, "mnth": 9, "season": 3, "weekday": 3, "weathersit": 1, "temp": 0.64},
            ]
        },
    )
    assert batch_res.status_code == 200
    payload = batch_res.json()
    assert payload["horizon_hours"] == 3
    assert payload["peak_hour"] in (8, 17)

    invalid_res = client.post("/predict", json={"hr": 29})
    assert invalid_res.status_code == 422

