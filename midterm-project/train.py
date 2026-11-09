#!/usr/bin/env python3
"""Train, benchmark, and export the BikePulse hourly demand forecasting model.

Usage:
    uv run python train.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from model import (
    TARGET_COLUMN,
    compute_training_defaults,
    dataframe_to_feature_dicts,
    postprocess_predictions,
)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "bike_sharing_hour.csv"
DEFAULT_ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
RANDOM_STATE = 42


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
    }


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform deterministic 60% train / 20% validation / 20% test split."""
    df_full_train, df_test = train_test_split(
        df, test_size=0.20, random_state=RANDOM_STATE
    )
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=RANDOM_STATE
    )
    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


def run_model_selection(
    X_train,
    y_train: np.ndarray,
    y_train_log: np.ndarray,
    X_val,
    y_val: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Train and tune Linear, Decision Tree, Random Forest, and XGBoost models."""
    leaderboard: list[dict[str, Any]] = []

    # 1. Ridge Regression tuning (raw vs log1p target)
    print("1/4 Tuning Ridge Regression (alpha x target transform)...")
    best_ridge_metrics: dict[str, Any] | None = None
    for use_log in (False, True):
        for alpha in (0.01, 0.1, 1.0, 10.0, 50.0):
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train, y_train_log if use_log else y_train)
            raw_pred = ridge.predict(X_val)
            preds = (
                postprocess_predictions(raw_pred)
                if use_log
                else np.maximum(raw_pred, 0.0)
            )
            metrics = evaluate_predictions(y_val, preds)
            candidate = {
                "family": "Linear (Ridge)",
                "params": {
                    "alpha": alpha,
                    "target_transform": "log1p" if use_log else "raw",
                },
                **metrics,
            }
            if best_ridge_metrics is None or candidate["rmse"] < best_ridge_metrics["rmse"]:
                best_ridge_metrics = candidate
    assert best_ridge_metrics is not None
    leaderboard.append(best_ridge_metrics)
    print(
        f"    Best Ridge -> RMSE={best_ridge_metrics['rmse']:.2f}, "
        f"R2={best_ridge_metrics['r2']:.4f} ({best_ridge_metrics['params']})"
    )

    # 2. Decision Tree Regressor tuning (max_depth x min_samples_leaf on log1p target)
    print("2/4 Tuning DecisionTreeRegressor (max_depth x min_samples_leaf)...")
    best_dt_metrics: dict[str, Any] | None = None
    for max_depth in (8, 10, 12, 15, 20, None):
        for min_leaf in (1, 5, 10, 20):
            dt = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=min_leaf,
                random_state=RANDOM_STATE,
            )
            dt.fit(X_train, y_train_log)
            preds = postprocess_predictions(dt.predict(X_val))
            metrics = evaluate_predictions(y_val, preds)
            candidate = {
                "family": "Decision Tree",
                "params": {
                    "max_depth": max_depth,
                    "min_samples_leaf": min_leaf,
                    "target_transform": "log1p",
                },
                **metrics,
            }
            if best_dt_metrics is None or candidate["rmse"] < best_dt_metrics["rmse"]:
                best_dt_metrics = candidate
    assert best_dt_metrics is not None
    leaderboard.append(best_dt_metrics)
    print(
        f"    Best DecisionTree -> RMSE={best_dt_metrics['rmse']:.2f}, "
        f"R2={best_dt_metrics['r2']:.4f} ({best_dt_metrics['params']})"
    )

    # 3. Random Forest Regressor tuning (max_depth x n_estimators on log1p target)
    print("3/4 Tuning RandomForestRegressor (max_depth x n_estimators)...")
    best_rf_metrics: dict[str, Any] | None = None
    for max_depth in (12, 18, 24):
        for n_estimators in (50, 100, 160):
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train_log)
            preds = postprocess_predictions(rf.predict(X_val))
            metrics = evaluate_predictions(y_val, preds)
            candidate = {
                "family": "Random Forest",
                "params": {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_leaf": 2,
                    "target_transform": "log1p",
                },
                **metrics,
            }
            if best_rf_metrics is None or candidate["rmse"] < best_rf_metrics["rmse"]:
                best_rf_metrics = candidate
    assert best_rf_metrics is not None
    leaderboard.append(best_rf_metrics)
    print(
        f"    Best RandomForest -> RMSE={best_rf_metrics['rmse']:.2f}, "
        f"R2={best_rf_metrics['r2']:.4f} ({best_rf_metrics['params']})"
    )

    # 4. XGBoost Regressor tuning (learning_rate x max_depth x min_child_weight)
    print("4/4 Tuning XGBRegressor (learning_rate x max_depth x min_child_weight)...")
    best_xgb_metrics: dict[str, Any] | None = None
    xgb_grid = [
        {"n_estimators": 300, "learning_rate": 0.10, "max_depth": 6, "min_child_weight": 1, "subsample": 0.85, "colsample_bytree": 0.85},
        {"n_estimators": 400, "learning_rate": 0.08, "max_depth": 6, "min_child_weight": 3, "subsample": 0.85, "colsample_bytree": 0.85},
        {"n_estimators": 450, "learning_rate": 0.06, "max_depth": 7, "min_child_weight": 3, "subsample": 0.85, "colsample_bytree": 0.85},
        {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 8, "min_child_weight": 3, "subsample": 0.85, "colsample_bytree": 0.85},
        {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 8, "min_child_weight": 5, "subsample": 0.90, "colsample_bytree": 0.85},
        {"n_estimators": 350, "learning_rate": 0.10, "max_depth": 8, "min_child_weight": 3, "subsample": 0.85, "colsample_bytree": 0.85},
    ]
    for cfg in xgb_grid:
        xgb_model = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **cfg,
        )
        xgb_model.fit(X_train, y_train_log)
        preds = postprocess_predictions(xgb_model.predict(X_val))
        metrics = evaluate_predictions(y_val, preds)
        candidate = {
            "family": "XGBoost",
            "params": {**cfg, "target_transform": "log1p"},
            **metrics,
        }
        if best_xgb_metrics is None or candidate["rmse"] < best_xgb_metrics["rmse"]:
            best_xgb_metrics = candidate

    assert best_xgb_metrics is not None
    leaderboard.append(best_xgb_metrics)
    print(
        f"    Best XGBoost -> RMSE={best_xgb_metrics['rmse']:.2f}, "
        f"R2={best_xgb_metrics['r2']:.4f} ({best_xgb_metrics['params']})"
    )

    leaderboard.sort(key=lambda row: row["rmse"])
    return leaderboard, best_xgb_metrics


def cross_validate_champion(
    df_full_train: pd.DataFrame, xgb_params: dict[str, Any], n_splits: int = 5
) -> dict[str, float]:
    """Run K-fold cross-validation on full_train (80%) to measure generalization stability."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rmses: list[float] = []
    r2s: list[float] = []

    clean_params = {k: v for k, v in xgb_params.items() if k != "target_transform"}

    for train_idx, val_idx in kf.split(df_full_train):
        fold_train = df_full_train.iloc[train_idx]
        fold_val = df_full_train.iloc[val_idx]

        fold_defaults = compute_training_defaults(fold_train)
        dv = DictVectorizer(sparse=True)
        X_tr = dv.fit_transform(dataframe_to_feature_dicts(fold_train, fold_defaults))
        X_va = dv.transform(dataframe_to_feature_dicts(fold_val, fold_defaults))

        y_tr_log = np.log1p(fold_train[TARGET_COLUMN].to_numpy(dtype=float))
        y_va = fold_val[TARGET_COLUMN].to_numpy(dtype=float)

        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **clean_params,
        )
        model.fit(X_tr, y_tr_log)
        preds = postprocess_predictions(model.predict(X_va))
        metrics = evaluate_predictions(y_va, preds)
        rmses.append(metrics["rmse"])
        r2s.append(metrics["r2"])

    return {
        "cv_folds": n_splits,
        "cv_rmse_mean": round(float(np.mean(rmses)), 4),
        "cv_rmse_std": round(float(np.std(rmses)), 4),
        "cv_r2_mean": round(float(np.mean(r2s)), 4),
    }


def train_and_export(data_path: Path, artifacts_dir: Path) -> dict[str, Any]:
    df = pd.read_csv(data_path)
    df_train, df_val, df_test = split_dataset(df)
    df_full_train = pd.concat([df_train, df_val], ignore_index=True)

    train_defaults = compute_training_defaults(df_train)
    dv_tune = DictVectorizer(sparse=True)
    X_train = dv_tune.fit_transform(dataframe_to_feature_dicts(df_train, train_defaults))
    X_val = dv_tune.transform(dataframe_to_feature_dicts(df_val, train_defaults))

    y_train = df_train[TARGET_COLUMN].to_numpy(dtype=float)
    y_train_log = np.log1p(y_train)
    y_val = df_val[TARGET_COLUMN].to_numpy(dtype=float)
    y_test = df_test[TARGET_COLUMN].to_numpy(dtype=float)

    leaderboard, best_xgb = run_model_selection(
        X_train, y_train, y_train_log, X_val, y_val
    )

    print("\nRunning 5-fold cross-validation on full_train (80%)...")
    cv_summary = cross_validate_champion(df_full_train, best_xgb["params"], n_splits=5)
    print(
        f"    5-Fold CV RMSE: {cv_summary['cv_rmse_mean']:.2f} +/- {cv_summary['cv_rmse_std']:.2f} "
        f"(mean R2={cv_summary['cv_r2_mean']:.4f})"
    )

    print("\nTraining final champion pipeline on full_train and evaluating on held-out test set...")
    full_defaults = compute_training_defaults(df_full_train)
    dv_final = DictVectorizer(sparse=True)
    X_full_train = dv_final.fit_transform(
        dataframe_to_feature_dicts(df_full_train, full_defaults)
    )
    X_test = dv_final.transform(dataframe_to_feature_dicts(df_test, full_defaults))
    y_full_train_log = np.log1p(df_full_train[TARGET_COLUMN].to_numpy(dtype=float))

    clean_xgb_params = {
        k: v for k, v in best_xgb["params"].items() if k != "target_transform"
    }
    final_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **clean_xgb_params,
    )
    final_model.fit(X_full_train, y_full_train_log)

    test_preds = postprocess_predictions(final_model.predict(X_test))
    test_metrics = evaluate_predictions(y_test, test_preds)
    print(
        f"    Held-Out Test -> RMSE={test_metrics['rmse']:.2f}, "
        f"MAE={test_metrics['mae']:.2f}, R2={test_metrics['r2']:.4f}"
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "model.bin"
    bundle = {
        "dv": dv_final,
        "model": final_model,
        "defaults": full_defaults,
        "target_transform": "log1p",
        "feature_count": len(dv_final.feature_names_),
    }
    with model_path.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()

    importances = sorted(
        zip(dv_final.feature_names_, final_model.feature_importances_.tolist()),
        key=lambda pair: pair[1],
        reverse=True,
    )[:12]

    metadata = {
        "project": "BikePulse Hourly Demand Forecasting",
        "dataset": data_path.name,
        "dataset_sha256": data_sha256,
        "rows_total": int(len(df)),
        "rows_train": int(len(df_train)),
        "rows_val": int(len(df_val)),
        "rows_test": int(len(df_test)),
        "target": TARGET_COLUMN,
        "target_transform": "log1p",
        "champion_family": best_xgb["family"],
        "champion_params": clean_xgb_params,
        "validation_metrics": {
            "rmse": best_xgb["rmse"],
            "mae": best_xgb["mae"],
            "r2": best_xgb["r2"],
        },
        "cross_validation": cv_summary,
        "test_metrics": test_metrics,
        "numeric_defaults": full_defaults,
        "encoded_feature_count": len(dv_final.feature_names_),
        "top_feature_importances": [
            {"feature": name, "importance": round(float(score), 5)}
            for name, score in importances
        ],
        "versions": {
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "model_artifact": model_path.name,
        "model_sha256": model_sha256,
    }

    (artifacts_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (artifacts_dir / "benchmark_results.json").write_text(
        json.dumps(
            {
                "validation_leaderboard": leaderboard,
                "cross_validation": cv_summary,
                "final_test_metrics": test_metrics,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\n[OK] Exported champion model to {model_path} (sha256={model_sha256[:16]}...)")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BikePulse hourly demand model")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    args = parser.parse_args()
    train_and_export(args.data, args.artifacts)

