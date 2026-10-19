#!/usr/bin/env python3
"""
Machine Learning Zoomcamp 2026 - Homework 4: Evaluation Metrics for Classification
Dataset: course_lead_scoring_2026.csv
Target: converted
"""

import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split

DATA_URL = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv"
DATA_FILE = Path(__file__).resolve().parent / "course_lead_scoring_2026.csv"

def download_data():
    if not DATA_FILE.exists():
        print(f"Downloading dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    return pd.read_csv(DATA_FILE)

def main():
    df = download_data()
    print("Dataset shape:", df.shape)

    # 1. Data Preparation
    target = "converted"
    features = df.drop(columns=[target]).copy()
    categorical = features.select_dtypes(include=["object"]).columns.tolist()
    numerical = features.select_dtypes(exclude=["object"]).columns.tolist()

    # Impute missing values: categorical -> 'NA', numerical -> 0.0
    features[categorical] = features[categorical].fillna("NA")
    features[numerical] = features[numerical].fillna(0.0)

    prepared_df = features.assign(**{target: df[target]})

    # Split: test_size=0.2, random_state=1, then train_val split test_size=0.25, random_state=1
    df_full_train, df_test = train_test_split(prepared_df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

    y_train = df_train.pop(target).values
    y_val = df_val.pop(target).values
    y_test = df_test.pop(target).values

    # Question 1: ROC AUC feature importance for numerical variables
    print("\n--- Question 1 ---")
    candidate_features = [
        "lead_score",
        "number_of_courses_viewed",
        "interaction_count",
        "annual_income",
    ]
    auc_scores = {}
    for col in candidate_features:
        auc = roc_auc_score(y_train, df_train[col])
        if auc < 0.5:
            auc = roc_auc_score(y_train, -df_train[col])
        auc_scores[col] = auc
        print(f"ROC AUC ({col}): {auc:.4f}")
    
    best_num_feat = max(candidate_features, key=auc_scores.get)
    print(f"Numerical variable with highest AUC: {best_num_feat}")

    # Question 2: Model Training and Validation AUC
    print("\n--- Question 2 ---")
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(df_train.to_dict(orient="records"))
    X_val = dv.transform(df_val.to_dict(orient="records"))

    model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000)
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"Validation ROC AUC: {val_auc:.4f} (rounded: {round(val_auc, 3)})")

    # Question 3: Precision and Recall Intersection
    print("\n--- Question 3 ---")
    thresholds = np.arange(0.0, 1.01, 0.01)
    metric_diffs = []
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        p = precision_score(y_val, preds, zero_division=0)
        r = recall_score(y_val, preds, zero_division=0)
        if p == 0 and r == 0:
            continue
        diff = abs(p - r)
        metric_diffs.append((diff, round(t, 2), p, r))
    
    metric_diffs.sort(key=lambda x: x[0])
    intersect_t = metric_diffs[0][1]
    print(f"Precision and Recall intersection threshold: {intersect_t} (diff: {metric_diffs[0][0]:.4f}, P={metric_diffs[0][2]:.4f}, R={metric_diffs[0][3]:.4f})")

    # Question 4: F1 Score Maximization
    print("\n--- Question 4 ---")
    f1_list = []
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        f1_list.append((score, round(t, 2)))
    
    f1_list.sort(key=lambda x: -x[0])
    best_f1_t = f1_list[0][1]
    print(f"Threshold where F1 is maximal: {best_f1_t} (F1 = {f1_list[0][0]:.4f})")

    # Question 5: 5-Fold Cross-Validation Standard Deviation
    print("\n--- Question 5 ---")
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    X_full = df_full_train.drop(columns=[target])
    y_full = df_full_train[target].values

    cv_scores = []
    for fold_train_idx, fold_val_idx in kfold.split(X_full):
        f_tr = X_full.iloc[fold_train_idx]
        f_va = X_full.iloc[fold_val_idx]
        f_y_tr = y_full[fold_train_idx]
        f_y_va = y_full[fold_val_idx]

        dv_f = DictVectorizer(sparse=False)
        X_f_tr = dv_f.fit_transform(f_tr.to_dict(orient="records"))
        X_f_va = dv_f.transform(f_va.to_dict(orient="records"))

        m_f = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000)
        m_f.fit(X_f_tr, f_y_tr)
        f_probs = m_f.predict_proba(X_f_va)[:, 1]
        cv_scores.append(roc_auc_score(f_y_va, f_probs))

    cv_std = np.std(cv_scores)
    print(f"Standard deviation across 5 folds: {cv_std:.4f} (rounded: {round(cv_std, 3)})")

    # Question 6: Hyperparameter Tuning with 5-Fold CV
    print("\n--- Question 6 ---")
    c_candidates = [0.000001, 0.001, 1.0]
    c_results = {}
    for c in c_candidates:
        fold_scores = []
        for fold_train_idx, fold_val_idx in kfold.split(X_full):
            f_tr = X_full.iloc[fold_train_idx]
            f_va = X_full.iloc[fold_val_idx]
            f_y_tr = y_full[fold_train_idx]
            f_y_va = y_full[fold_val_idx]

            dv_f = DictVectorizer(sparse=False)
            X_f_tr = dv_f.fit_transform(f_tr.to_dict(orient="records"))
            X_f_va = dv_f.transform(f_va.to_dict(orient="records"))

            m_c = LogisticRegression(solver="liblinear", C=c, max_iter=1000)
            m_c.fit(X_f_tr, f_y_tr)
            fold_scores.append(roc_auc_score(f_y_va, m_c.predict_proba(X_f_va)[:, 1]))
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        c_results[c] = (round(mean_score, 3), round(std_score, 3))
        print(f"C = {c:8g}: mean AUC = {mean_score:.4f} (round: {c_results[c][0]}), std = {std_score:.4f} (round: {c_results[c][1]})")

    # Tie breaking: highest mean, then lowest std, then smallest C
    best_c = max(c_candidates, key=lambda c: (c_results[c][0], -c_results[c][1], -c))
    print(f"C with best mean ROC AUC: {best_c}")

if __name__ == "__main__":
    main()
