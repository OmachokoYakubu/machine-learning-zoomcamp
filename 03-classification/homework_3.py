#!/usr/bin/env python3
"""
Machine Learning Zoomcamp 2026 - Homework 3: Machine Learning for Classification
Dataset: course_lead_scoring_2026.csv
Target: converted
"""

import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.model_selection import train_test_split

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

    # Impute missing values
    features[categorical] = features[categorical].fillna("NA")
    features[numerical] = features[numerical].fillna(0.0)

    prepared_df = features.assign(**{target: df[target]})

    # Question 1: Most frequent observation (mode) for industry
    mode_industry = df["industry"].mode()[0]
    print(f"\n--- Question 1 ---")
    print(f"Mode for 'industry': {mode_industry}")

    # Question 2: Correlation matrix for numerical features
    corr_matrix = features[numerical].corr().abs()
    print(f"\n--- Question 2 ---")
    candidate_pairs = [
        ("interaction_count", "lead_score"),
        ("number_of_courses_viewed", "lead_score"),
        ("number_of_courses_viewed", "interaction_count"),
        ("annual_income", "interaction_count"),
    ]
    for p1, p2 in candidate_pairs:
        print(f"Correlation({p1}, {p2}) = {corr_matrix.loc[p1, p2]:.4f}")
    best_pair = max(candidate_pairs, key=lambda p: corr_matrix.loc[p[0], p[1]])
    print(f"Features with biggest correlation: {best_pair[0]} and {best_pair[1]}")

    # Split the data
    df_full_train, df_test = train_test_split(prepared_df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

    y_train = df_train.pop(target).values
    y_val = df_val.pop(target).values
    y_test = df_test.pop(target).values

    # Question 3: Mutual information score between converted and categorical variables
    print(f"\n--- Question 3 ---")
    mi_scores = {}
    for col in categorical:
        score = mutual_info_score(y_train, df_train[col])
        mi_scores[col] = score
        print(f"Mutual Information ({col}): {score:.4f} (rounded: {round(score, 2)})")
    best_mi = max(mi_scores, key=mi_scores.get)
    print(f"Categorical variable with biggest MI: {best_mi}")

    # Question 4: Logistic Regression with DictVectorizer
    print(f"\n--- Question 4 ---")
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(df_train.to_dict(orient="records"))
    X_val = dv.transform(df_val.to_dict(orient="records"))

    model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    original_acc = accuracy_score(y_val, y_val_pred)
    print(f"Validation accuracy: {original_acc:.4f} (rounded: {round(original_acc, 2)})")

    # Question 5: Feature elimination
    print(f"\n--- Question 5 ---")
    elim_features = ["lead_source", "number_of_courses_viewed", "interaction_count"]
    diffs = {}
    for col in elim_features:
        tr_sub = df_train.drop(columns=[col])
        va_sub = df_val.drop(columns=[col])
        dv_sub = DictVectorizer(sparse=False)
        X_tr_sub = dv_sub.fit_transform(tr_sub.to_dict(orient="records"))
        X_va_sub = dv_sub.transform(va_sub.to_dict(orient="records"))
        
        m_sub = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
        m_sub.fit(X_tr_sub, y_train)
        acc_sub = accuracy_score(y_val, m_sub.predict(X_va_sub))
        diff = original_acc - acc_sub
        diffs[col] = diff
        print(f"Without '{col}': accuracy = {acc_sub:.4f}, diff = {diff:+.4f}, abs_diff = {abs(diff):.4f}")
    
    smallest_diff_feature = min(diffs, key=lambda c: abs(diffs[c]))
    print(f"Feature with smallest difference: '{smallest_diff_feature}'")

    # Question 6: Regularized Logistic Regression parameter C
    print(f"\n--- Question 6 ---")
    c_values = [0.000001, 0.00001, 0.0001, 0.001]
    c_scores = {}
    for c in c_values:
        m_c = LogisticRegression(solver="liblinear", C=c, max_iter=1000, random_state=42)
        m_c.fit(X_train, y_train)
        acc_c = accuracy_score(y_val, m_c.predict(X_val))
        c_scores[c] = round(acc_c, 3)
        print(f"C = {c:8f}: accuracy = {acc_c:.4f} (rounded: {c_scores[c]})")
    
    best_c = max(c_values, key=lambda c: (c_scores[c], -c))
    print(f"Best parameter C: {best_c}")

if __name__ == "__main__":
    main()
