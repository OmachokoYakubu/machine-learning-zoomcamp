"""
Homework 1: Introduction to Machine Learning
Machine Learning Zoomcamp 2026
Student: Omachoko Yakubu
"""

import numpy as np
import pandas as pd

# Download / Load dataset
DATA_URL = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
print(f"Loading data from {DATA_URL}...")
df = pd.read_csv(DATA_URL)

# Q1. Pandas version
print("\n" + "=" * 50)
print(f"Q1. Pandas version: {pd.__version__}")

# Q2. Records count
records_count = len(df)
print(f"Q2. Records count: {records_count}")

# Q3. Fuel types
fuel_types = df["fuel_type"].unique()
fuel_type_count = df["fuel_type"].nunique()
print(f"Q3. Fuel types presented: {fuel_type_count} ({fuel_types.tolist()})")

# Q4. Missing values
missing_per_col = df.isnull().sum()
missing_col_count = (missing_per_col > 0).sum()
print("Missing values per column:")
for col, cnt in missing_per_col.items():
    if cnt > 0:
        print(f"  - {col}: {cnt}")
print(f"Q4. Number of columns with missing values: {missing_col_count}")

# Q5. Max fuel efficiency
asia_cars = df[df["origin"] == "Asia"]
max_fuel_eff_asia = asia_cars["fuel_efficiency_mpg"].max()
print(f"Q5. Maximum fuel efficiency of cars from Asia: {max_fuel_eff_asia:.1f} (raw: {max_fuel_eff_asia})")

# Q6. Median value of horsepower
med_before = df["horsepower"].median()
mode_val = df["horsepower"].mode().iloc[0]
df_filled = df.copy()
df_filled["horsepower"] = df_filled["horsepower"].fillna(mode_val)
med_after = df_filled["horsepower"].median()

if med_after > med_before:
    change = "Yes, it increased"
elif med_after < med_before:
    change = "Yes, it decreased"
else:
    change = "No"

print(f"Q6. Horsepower median before: {med_before}, mode: {mode_val}, median after: {med_after}")
print(f"    Has it changed? {change}")

# Q7. Sum of weights
# 1. Select all the cars from Asia
# 2. Select only columns vehicle_weight and model_year
# 3. Select the first 7 values
asia_7 = asia_cars[["vehicle_weight", "model_year"]].head(7)
# 4. Get underlying numpy array X
X = asia_7.to_numpy()
# 5. Matrix-matrix multiplication between transpose of X and X (XTX)
XTX = X.T @ X
# 6. Invert XTX
XTX_inv = np.linalg.inv(XTX)
# 7. Create array y
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
# 8. Multiply inverse of XTX with transpose of X and y (w)
w = XTX_inv @ X.T @ y
sum_w = w.sum()
print(f"Q7. Weights vector w: {w}")
print(f"    Sum of all elements in w: {sum_w:.4f} -> rounded: {sum_w:.3f}")
print("=" * 50 + "\n")
