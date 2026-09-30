"""
Homework 2: Machine Learning for Regression
Machine Learning Zoomcamp 2026
Student: Omachoko Yakubu
"""

import numpy as np
import pandas as pd

# Load 2026 dataset
DATA_URL = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
print(f"Loading data from {DATA_URL}...")
df_raw = pd.read_csv(DATA_URL)

# Preparing the dataset
CAR_COLUMNS = [
    "engine_displacement",
    "horsepower",
    "vehicle_weight",
    "model_year",
]
TARGET = "fuel_efficiency_mpg"

df = df_raw[CAR_COLUMNS + [TARGET]].copy()

print("=" * 60)
print(f"Dataset shape: {df.shape}")

# Question 1: Missing values
missing = df.isnull().sum()
print("\nMissing values per column:")
for col, cnt in missing.items():
    print(f"  - {col}: {cnt}")
missing_col = df.columns[df.isnull().any()][0]
print(f"Q1. Column with missing values: {missing_col}")

# Question 2: Median for horsepower
med_hp = df["horsepower"].median()
print(f"\nQ2. Median for variable 'horsepower': {med_hp:.0f}")

# Helper functions for linear regression and lecture shuffle
def split_with_lecture_shuffle(data_df: pd.DataFrame, seed: int):
    n = len(data_df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    indices = np.arange(n)
    np.random.RandomState(seed).shuffle(indices)

    train_set = data_df.iloc[indices[:n_train]].copy()
    val_set = data_df.iloc[indices[n_train : n_train + n_val]].copy()
    test_set = data_df.iloc[indices[n_train + n_val :]].copy()

    return train_set, val_set, test_set

def train_linear_regression(X_train: np.ndarray, y_train: np.ndarray, r: float = 0.0):
    ones = np.ones(X_train.shape[0])
    X = np.column_stack([ones, X_train])
    XTX = X.T @ X
    if r > 0:
        reg = r * np.eye(XTX.shape[0])
        reg[0, 0] = 0.0
        XTX = XTX + reg
    w = np.linalg.solve(XTX, X.T @ y_train)
    return w[0], w[1:]

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def evaluate_linear_regression(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    fill_value: float,
    r: float = 0.0,
) -> float:
    X_train = train_df[CAR_COLUMNS].fillna(fill_value).to_numpy(dtype=float)
    y_train = train_df[TARGET].to_numpy(dtype=float)

    X_val = val_df[CAR_COLUMNS].fillna(fill_value).to_numpy(dtype=float)
    y_val = val_df[TARGET].to_numpy(dtype=float)

    w0, w = train_linear_regression(X_train, y_train, r=r)
    y_pred = w0 + X_val @ w
    return rmse(y_val, y_pred)

# Prepare and split dataset with seed 42
df_train, df_val, df_test = split_with_lecture_shuffle(df, seed=42)

# Question 3: Filling NAs (With 0 vs With mean)
rmse_zero = evaluate_linear_regression(df_train, df_val, fill_value=0.0)
mean_hp_train = df_train["horsepower"].mean()
rmse_mean = evaluate_linear_regression(df_train, df_val, fill_value=mean_hp_train)

print("\nQ3. Filling NAs evaluation:")
print(f"  - RMSE with 0:    {rmse_zero:.4f} -> rounded(3): {round(rmse_zero, 3)}")
print(f"  - RMSE with mean: {rmse_mean:.4f} -> rounded(3): {round(rmse_mean, 3)}")

if round(rmse_zero, 3) < round(rmse_mean, 3):
    q3_ans = "With 0"
elif round(rmse_mean, 3) < round(rmse_zero, 3):
    q3_ans = "With mean"
else:
    q3_ans = "Both are equally good"
print(f"  Answer: {q3_ans}")

# Question 4: Regularization r tuning
reg_candidates = [0, 0.01, 0.1, 1, 5, 10, 100]
print("\nQ4. Regularized linear regression (filled with 0):")
reg_scores = {}
for r in reg_candidates:
    score = evaluate_linear_regression(df_train, df_val, fill_value=0.0, r=r)
    reg_scores[r] = score
    print(f"  - r = {r:<4}: RMSE = {score:.5f} (round 4: {round(score, 4)})")

best_r = min(reg_candidates, key=lambda r: (round(reg_scores[r], 4), r))
print(f"  Best r: {best_r}")

# Question 5: Seed stability (RMSE standard deviation)
seeds = list(range(10))
seed_scores = []
for s in seeds:
    s_train, s_val, _ = split_with_lecture_shuffle(df, seed=s)
    score = evaluate_linear_regression(s_train, s_val, fill_value=0.0, r=0.0)
    seed_scores.append(score)

std_rmse = np.std(seed_scores)
print("\nQ5. Seed stability across seeds [0..9]:")
print(f"  Scores: {[round(s, 4) for s in seed_scores]}")
print(f"  Standard Deviation: {std_rmse:.4f} -> rounded(3): {round(std_rmse, 3)}")

# Question 6: Test evaluation with seed 9
train_9, val_9, test_9 = split_with_lecture_shuffle(df, seed=9)
combined_9 = pd.concat([train_9, val_9])
test_score = evaluate_linear_regression(combined_9, test_9, fill_value=0.0, r=0.001)
print(f"\nQ6. Test RMSE (seed=9, r=0.001): {test_score:.4f} -> rounded: {round(test_score, 3)}")
print("=" * 60 + "\n")
