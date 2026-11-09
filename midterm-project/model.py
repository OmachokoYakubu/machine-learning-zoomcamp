"""Shared domain schema, feature engineering, and operational dispatch logic for BikePulse."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

TARGET_COLUMN = "cnt"
LEAKAGE_COLUMNS = ("casual", "registered")
IDENTIFIER_COLUMNS = ("instant", "dteday")

RAW_FEATURE_COLUMNS = (
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
)

SEASON_MAP = {
    1: "spring",
    2: "summer",
    3: "fall",
    4: "winter",
}

WEATHER_MAP = {
    1: "clear",
    2: "mist_cloudy",
    3: "light_precip",
    4: "severe_storm",
}

WEEKDAY_MAP = {
    0: "sun",
    1: "mon",
    2: "tue",
    3: "wed",
    4: "thu",
    5: "fri",
    6: "sat",
}

DEFAULT_HUMIDITY = 0.62


def compute_training_defaults(df_train: pd.DataFrame) -> dict[str, float]:
    """Compute training-set imputation defaults (excluding zero-humidity sensor faults)."""
    valid_hum = df_train.loc[df_train["hum"] > 0, "hum"]
    return {
        "temp": float(df_train["temp"].median()),
        "atemp": float(df_train["atemp"].median()),
        "hum": float(valid_hum.median()) if len(valid_hum) else DEFAULT_HUMIDITY,
        "windspeed": float(df_train["windspeed"].median()),
    }


def engineer_record(
    raw: dict[str, Any], defaults: dict[str, float] | None = None
) -> dict[str, str | int | float]:
    """Transform a raw hourly observation into a model-ready feature dictionary.

    Strictly excludes target (`cnt`), leakage (`casual`, `registered`), and row IDs.
    """
    defaults = defaults or {
        "temp": 0.50,
        "atemp": 0.4848,
        "hum": DEFAULT_HUMIDITY,
        "windspeed": 0.1642,
    }

    season = int(raw.get("season", 3))
    yr = int(raw.get("yr", 1))
    mnth = int(raw.get("mnth", 6))
    hr = int(raw.get("hr", 12))
    holiday = int(raw.get("holiday", 0))
    weekday = int(raw.get("weekday", 3))
    workingday = int(raw.get("workingday", 1 if (holiday == 0 and weekday not in (0, 6)) else 0))
    weathersit = int(raw.get("weathersit", 1))

    temp_val = raw.get("temp")
    temp = float(defaults["temp"] if temp_val is None or pd.isna(temp_val) else temp_val)

    atemp_val = raw.get("atemp")
    atemp = float(temp if atemp_val is None or pd.isna(atemp_val) else atemp_val)

    hum_val = raw.get("hum")
    if hum_val is None or pd.isna(hum_val) or float(hum_val) <= 0.0:
        hum = float(defaults["hum"])
    else:
        hum = float(hum_val)

    wind_val = raw.get("windspeed")
    windspeed = float(
        defaults["windspeed"] if wind_val is None or pd.isna(wind_val) else wind_val
    )

    temp = min(max(temp, 0.0), 1.2)
    atemp = min(max(atemp, 0.0), 1.2)
    hum = min(max(hum, 0.01), 1.0)
    windspeed = min(max(windspeed, 0.0), 1.2)

    season_name = SEASON_MAP.get(season, "fall")
    weather_name = WEATHER_MAP.get(weathersit, "clear")
    weekday_name = WEEKDAY_MAP.get(weekday % 7, "wed")

    if holiday == 1:
        day_regime = "holiday"
    elif workingday == 1:
        day_regime = "workingday"
    else:
        day_regime = "weekend"

    hr_angle = 2.0 * math.pi * (hr % 24) / 24.0
    mnth_angle = 2.0 * math.pi * ((mnth - 1) % 12) / 12.0

    is_morning_rush = int(workingday == 1 and hr in (7, 8, 9))
    is_evening_rush = int(workingday == 1 and hr in (16, 17, 18, 19))
    is_night = int(hr in (0, 1, 2, 3, 4, 5))

    return {
        "season_name": season_name,
        "weather_condition": weather_name,
        "weekday_name": weekday_name,
        "day_regime": day_regime,
        "hour_bucket": f"hr_{hr:02d}",
        "regime_hour": f"{day_regime}_hr_{hr:02d}",
        "yr": yr,
        "mnth": mnth,
        "hr": hr,
        "holiday": holiday,
        "weekday": weekday,
        "workingday": workingday,
        "weathersit": weathersit,
        "temp": round(temp, 6),
        "atemp": round(atemp, 6),
        "hum": round(hum, 6),
        "windspeed": round(windspeed, 6),
        "hr_sin": round(math.sin(hr_angle), 6),
        "hr_cos": round(math.cos(hr_angle), 6),
        "mnth_sin": round(math.sin(mnth_angle), 6),
        "mnth_cos": round(math.cos(mnth_angle), 6),
        "is_morning_rush": is_morning_rush,
        "is_evening_rush": is_evening_rush,
        "is_night": is_night,
        "thermal_comfort": round(temp * (1.0 - 0.5 * hum), 6),
        "weather_severity_wind": round(weathersit * (1.0 + windspeed), 6),
    }


def dataframe_to_feature_dicts(
    df: pd.DataFrame, defaults: dict[str, float] | None = None
) -> list[dict[str, str | int | float]]:
    """Convert a DataFrame of raw observations into feature dictionaries."""
    safe_cols = [c for c in RAW_FEATURE_COLUMNS if c in df.columns]
    records = df[safe_cols].to_dict(orient="records")
    return [engineer_record(row, defaults=defaults) for row in records]


def postprocess_predictions(log_preds: np.ndarray | float) -> np.ndarray | float:
    """Invert log1p predictions and enforce non-negative bike counts."""
    preds = np.expm1(log_preds)
    return np.maximum(preds, 0.0)


def classify_dispatch_tier(predicted_bikes: float) -> dict[str, str]:
    """Map hourly network demand forecast to an actionable fleet rebalancing tier."""
    if predicted_bikes < 75.0:
        return {
            "dispatch_tier": "low_demand",
            "recommended_action": (
                "Standard passive monitoring; ideal window for depot charging and maintenance."
            ),
        }
    if predicted_bikes < 225.0:
        return {
            "dispatch_tier": "normal_operations",
            "recommended_action": (
                "Balanced network flow; maintain scheduled van rebalancing loops."
            ),
        }
    if predicted_bikes < 400.0:
        return {
            "dispatch_tier": "rebalance_watch",
            "recommended_action": (
                "Elevated commuter turnover; pre-stage rebalancing vans near metro transit hubs."
            ),
        }
    return {
        "dispatch_tier": "surge_dispatch",
        "recommended_action": (
            "Peak surge alert; actively clear downtown destination docks and replenish residential origins."
        ),
    }

