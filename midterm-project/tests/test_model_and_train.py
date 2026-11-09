"""Unit tests for BikePulse feature engineering, leakage guards, and dispatch tiers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from model import (  # noqa: E402
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
    classify_dispatch_tier,
    compute_training_defaults,
    dataframe_to_feature_dicts,
    engineer_record,
    postprocess_predictions,
)


def test_leakage_and_target_excluded_from_features() -> None:
    sample_df = pd.DataFrame(
        [
            {
                "instant": 10,
                "dteday": "2012-09-15",
                "season": 3,
                "yr": 1,
                "mnth": 9,
                "hr": 17,
                "holiday": 0,
                "weekday": 3,
                "workingday": 1,
                "weathersit": 1,
                "temp": 0.62,
                "atemp": 0.60,
                "hum": 0.50,
                "windspeed": 0.16,
                "casual": 95,
                "registered": 410,
                "cnt": 505,
            }
        ]
    )
    records = dataframe_to_feature_dicts(sample_df)
    assert len(records) == 1
    feat = records[0]
    assert TARGET_COLUMN not in feat
    for col in LEAKAGE_COLUMNS:
        assert col not in feat
    assert "instant" not in feat
    assert "dteday" not in feat
    assert feat["is_evening_rush"] == 1
    assert feat["regime_hour"] == "workingday_hr_17"


def test_zero_humidity_sensor_fault_imputation() -> None:
    df = pd.DataFrame(
        [
            {"temp": 0.4, "atemp": 0.4, "hum": 0.0, "windspeed": 0.1},
            {"temp": 0.5, "atemp": 0.5, "hum": 0.64, "windspeed": 0.2},
            {"temp": 0.6, "atemp": 0.6, "hum": 0.70, "windspeed": 0.2},
        ]
    )
    defaults = compute_training_defaults(df)
    assert abs(defaults["hum"] - 0.67) < 1e-6

    rec = engineer_record({"hum": 0.0, "temp": 0.4}, defaults=defaults)
    assert abs(float(rec["hum"]) - 0.67) < 1e-6


def test_postprocess_and_dispatch_tiers() -> None:
    preds = postprocess_predictions(np.array([0.0, np.log1p(250.0)]))
    assert preds[0] == 0.0
    assert abs(preds[1] - 250.0) < 1e-6

    assert classify_dispatch_tier(30.0)["dispatch_tier"] == "low_demand"
    assert classify_dispatch_tier(150.0)["dispatch_tier"] == "normal_operations"
    assert classify_dispatch_tier(310.0)["dispatch_tier"] == "rebalance_watch"
    assert classify_dispatch_tier(520.0)["dispatch_tier"] == "surge_dispatch"

