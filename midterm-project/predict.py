"""FastAPI inference service and interactive dispatch dashboard for BikePulse."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from model import classify_dispatch_tier, engineer_record, postprocess_predictions

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", PROJECT_DIR / "artifacts"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", ARTIFACTS_DIR / "model.bin"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", ARTIFACTS_DIR / "model_metadata.json"))
INDEX_HTML_PATH = PROJECT_DIR / "static" / "index.html"


def _load_bundle() -> tuple[dict[str, Any], str, dict[str, Any]]:
    with MODEL_PATH.open("rb") as f:
        loaded_bundle = pickle.load(f)
    sha256_hex = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    metadata: dict[str, Any] = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return loaded_bundle, sha256_hex, metadata


bundle, MODEL_SHA256, MODEL_METADATA = _load_bundle()

app = FastAPI(
    title="BikePulse Demand Forecasting API",
    description=(
        "Real-time hourly micromobility bike rental demand forecasting and "
        "fleet rebalancing tier service (ML Zoomcamp 2026 Midterm Project)."
    ),
    version="1.0.0",
)


class HourlyContext(BaseModel):
    season: int = Field(
        default=3, ge=1, le=4, description="1: Spring, 2: Summer, 3: Fall, 4: Winter"
    )
    yr: int = Field(
        default=1, ge=0, le=1, description="Network baseline year index (0: 2011, 1: mature scale)"
    )
    mnth: int = Field(default=9, ge=1, le=12, description="Calendar month (1..12)")
    hr: int = Field(default=8, ge=0, le=23, description="Hour of day (0..23)")
    holiday: int = Field(default=0, ge=0, le=1, description="1 if public holiday else 0")
    weekday: int = Field(default=2, ge=0, le=6, description="0: Sun, 1: Mon, ..., 6: Sat")
    workingday: int | None = Field(
        default=None,
        ge=0,
        le=1,
        description="1 if working day (neither weekend nor holiday), inferred if omitted",
    )
    weathersit: int = Field(
        default=1,
        ge=1,
        le=4,
        description="1: Clear, 2: Mist/Cloudy, 3: Light Snow/Rain, 4: Heavy Rain/Storm",
    )
    temp: float | None = Field(
        default=None, ge=0.0, le=1.2, description="Normalized temperature (Celsius / 41)"
    )
    atemp: float | None = Field(
        default=None,
        ge=0.0,
        le=1.2,
        description="Normalized feeling temperature (Celsius / 50)",
    )
    hum: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Normalized relative humidity (0..1)"
    )
    windspeed: float | None = Field(
        default=None, ge=0.0, le=1.2, description="Normalized wind speed (km/h / 67)"
    )
    temp_celsius: float | None = Field(
        default=None, ge=-15.0, le=45.0, description="Optional temperature in Celsius"
    )
    feels_like_celsius: float | None = Field(
        default=None,
        ge=-20.0,
        le=50.0,
        description="Optional feels-like temperature in Celsius",
    )
    humidity_pct: float | None = Field(
        default=None, ge=1.0, le=100.0, description="Optional relative humidity percentage"
    )
    windspeed_kmh: float | None = Field(
        default=None, ge=0.0, le=80.0, description="Optional wind speed in km/h"
    )

    def to_raw_dict(self) -> dict[str, Any]:
        payload = self.model_dump()
        if payload["temp"] is None and payload["temp_celsius"] is not None:
            payload["temp"] = payload["temp_celsius"] / 41.0
        if payload["atemp"] is None and payload["feels_like_celsius"] is not None:
            payload["atemp"] = payload["feels_like_celsius"] / 50.0
        elif payload["atemp"] is None and payload["temp"] is not None:
            payload["atemp"] = min(payload["temp"] * (41.0 / 50.0) + 0.05, 1.0)
        if payload["hum"] is None and payload["humidity_pct"] is not None:
            payload["hum"] = payload["humidity_pct"] / 100.0
        if payload["windspeed"] is None and payload["windspeed_kmh"] is not None:
            payload["windspeed"] = payload["windspeed_kmh"] / 67.0
        if payload["workingday"] is None:
            payload["workingday"] = (
                1 if (payload["holiday"] == 0 and payload["weekday"] not in (0, 6)) else 0
            )
        return payload


class BatchForecastRequest(BaseModel):
    hours: list[HourlyContext] = Field(..., min_length=1, max_length=168)


def _score_contexts(contexts: list[HourlyContext]) -> list[dict[str, Any]]:
    dv = bundle["dv"]
    reg_model = bundle["model"]
    defaults = bundle["defaults"]

    engineered = [engineer_record(ctx.to_raw_dict(), defaults=defaults) for ctx in contexts]
    X = dv.transform(engineered)
    raw_preds = reg_model.predict(X)
    bike_preds = postprocess_predictions(raw_preds)

    results: list[dict[str, Any]] = []
    for feat, pred_val in zip(engineered, bike_preds):
        bikes = float(pred_val)
        dispatch = classify_dispatch_tier(bikes)
        results.append(
            {
                "predicted_hourly_rentals": round(bikes, 2),
                "predicted_hourly_rentals_rounded": int(round(bikes)),
                **dispatch,
                "context_summary": {
                    "hour": feat["hr"],
                    "day_regime": feat["day_regime"],
                    "season": feat["season_name"],
                    "weather": feat["weather_condition"],
                    "temp_norm": feat["temp"],
                    "humidity_norm": feat["hum"],
                },
            }
        )
    return results


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    if INDEX_HTML_PATH.exists():
        return HTMLResponse(INDEX_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>BikePulse Demand Forecasting API</h1><p>Visit /docs</p>")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "bikepulse-demand-forecasting",
        "model_family": MODEL_METADATA.get("champion_family", "XGBoost"),
        "model_sha256": MODEL_SHA256,
        "test_metrics": MODEL_METADATA.get("test_metrics", {}),
    }


@app.post("/predict")
def predict(context: HourlyContext) -> dict[str, Any]:
    scored = _score_contexts([context])[0]
    return {
        **scored,
        "model_sha256": MODEL_SHA256,
    }


@app.post("/predict/batch")
def predict_batch(request: BatchForecastRequest) -> dict[str, Any]:
    forecasts = _score_contexts(request.hours)
    total = sum(item["predicted_hourly_rentals"] for item in forecasts)
    peak = max(forecasts, key=lambda item: item["predicted_hourly_rentals"])
    return {
        "horizon_hours": len(forecasts),
        "total_predicted_rentals": round(total, 2),
        "peak_hour": peak["context_summary"]["hour"],
        "peak_predicted_rentals": peak["predicted_hourly_rentals"],
        "forecasts": forecasts,
        "model_sha256": MODEL_SHA256,
    }

