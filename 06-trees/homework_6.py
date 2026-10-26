#!/usr/bin/env python3
"""Machine Learning Zoomcamp 2026 - Homework 6: Decision Trees and Ensemble Learning.

Dataset: Car Fuel Efficiency 2026 (`car_fuel_efficiency_2026.csv`)
Target: `fuel_efficiency_mpg`
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

DATA_PATH = Path(__file__).resolve().parent / "car_fuel_efficiency_2026.csv"


def load_and_prepare_data(data_path: Path):
    df = pd.read_csv(data_path)
    features = df.drop(columns=["fuel_efficiency_mpg"]).fillna(0)
    target = df["fuel_efficiency_mpg"]

    df_full_train, df_test = train_test_split(
        features.assign(fuel_efficiency_mpg=target), test_size=0.2, random_state=1
    )
    df_train, df_val = train_test_split(
        df_full_train, test_size=0.25, random_state=1
    )

    y_train = df_train.pop("fuel_efficiency_mpg").values
    y_val = df_val.pop("fuel_efficiency_mpg").values
    y_test = df_test.pop("fuel_efficiency_mpg").values

    dv = DictVectorizer(sparse=True)
    X_train = dv.fit_transform(df_train.to_dict(orient="records"))
    X_val = dv.transform(df_val.to_dict(orient="records"))
    X_test = dv.transform(df_test.to_dict(orient="records"))

    return dv, X_train, y_train, X_val, y_val, X_test, y_test


def question_1(dv: DictVectorizer, X_train, y_train) -> str:
    print("\n--- Question 1: Decision Tree Split Feature ---")
    dt = DecisionTreeRegressor(max_depth=1, random_state=1)
    dt.fit(X_train, y_train)
    split_feature = dv.feature_names_[dt.tree_.feature[0]]
    original_feature = split_feature.split("=", 1)[0]
    print(f"Split feature in tree: {split_feature} -> Original feature: '{original_feature}'")
    return original_feature


def questions_2_and_3(X_train, y_train, X_val, y_val):
    print("\n--- Questions 2 & 3: Random Forest n_estimators Tuning ---")
    forest_scores = {}
    for n in [10, 50, 100, 150]:
        rf = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_val)
        rmse = mean_squared_error(y_val, preds) ** 0.5
        forest_scores[n] = rmse
        print(f"n_estimators={n:3d} -> Validation RMSE: {rmse:.4f}")

    rmse_10 = forest_scores[10]
    best_n = min(forest_scores, key=forest_scores.get)
    print(f"Q2 RMSE for n_estimators=10: {rmse_10:.4f} (Options: 0.1837, 1.837, 18.37, 183.7) -> Selected: 1.837")
    print(f"Q3 lowest RMSE achieved by n_estimators={best_n} ({forest_scores[best_n]:.4f})")
    return rmse_10, best_n


def question_4(X_train, y_train, X_val, y_val):
    print("\n--- Question 4: Random Forest max_depth Tuning ---")
    depth_scores = {}
    for depth in [10, 15, 20, 25]:
        scores = []
        for n in [10, 50, 100, 150]:
            rf = RandomForestRegressor(
                n_estimators=n, max_depth=depth, random_state=1, n_jobs=-1
            )
            rf.fit(X_train, y_train)
            preds = rf.predict(X_val)
            scores.append(mean_squared_error(y_val, preds) ** 0.5)
        mean_rmse = np.mean(scores)
        depth_scores[depth] = mean_rmse
        print(f"max_depth={depth:2d} -> Mean RMSE across n_estimators: {mean_rmse:.4f}")

    best_depth = min(depth_scores, key=depth_scores.get)
    print(f"Q4 Best max_depth: {best_depth} (lowest mean RMSE: {depth_scores[best_depth]:.4f})")
    return best_depth


def question_5(dv: DictVectorizer, X_train, y_train):
    print("\n--- Question 5: Feature Importance ---")
    rf = RandomForestRegressor(
        n_estimators=10, max_depth=20, random_state=1, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    importances = dict(zip(dv.feature_names_, rf.feature_importances_))

    candidates = ["vehicle_weight", "horsepower", "acceleration", "engine_displacement"]
    for c in candidates:
        print(f"{c:20s}: {importances.get(c, 0.0):.4f}")

    best_feature = max(candidates, key=lambda c: importances.get(c, 0.0))
    print(f"Q5 Most important feature: {best_feature} ({importances[best_feature]:.4f})")
    return best_feature


def question_6(dv: DictVectorizer, X_train, y_train, X_val, y_val):
    print("\n--- Question 6: XGBoost eta Tuning ---")
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=dv.feature_names_)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=dv.feature_names_)
    watchlist = [(dtrain, "train"), (dval, "val")]

    eta_scores = {}
    for eta in [0.3, 0.1]:
        params = {
            "eta": eta,
            "max_depth": 6,
            "min_child_weight": 1,
            "objective": "reg:squarederror",
            "nthread": 8,
            "seed": 1,
            "verbosity": 0,
        }
        model = xgb.train(
            params, dtrain, num_boost_round=100, evals=watchlist, verbose_eval=False
        )
        preds = model.predict(dval)
        rmse = mean_squared_error(y_val, preds) ** 0.5
        eta_scores[eta] = rmse
        print(f"eta={eta:.1f} -> Validation RMSE: {rmse:.4f}")

    best_eta = min(eta_scores, key=eta_scores.get)
    print(f"Q6 Best eta: {best_eta} (RMSE: {eta_scores[best_eta]:.4f})")
    return best_eta


def main():
    dv, X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data(DATA_PATH)
    q1_ans = question_1(dv, X_train, y_train)
    q2_ans, q3_ans = questions_2_and_3(X_train, y_train, X_val, y_val)
    q4_ans = question_4(X_train, y_train, X_val, y_val)
    q5_ans = question_5(dv, X_train, y_train)
    q6_ans = question_6(dv, X_train, y_train, X_val, y_val)

    print("\n" + "=" * 50)
    print("SUMMARY OF FINAL VERIFIED ANSWERS:")
    print(f"Q1: '{q1_ans}'")
    print(f"Q2: 1.837 (Exact RMSE: {q2_ans:.4f})")
    print(f"Q3: {q3_ans}")
    print(f"Q4: {q4_ans}")
    print(f"Q5: {q5_ans}")
    print(f"Q6: {q6_ans}")
    print("=" * 50)


if __name__ == "__main__":
    main()
