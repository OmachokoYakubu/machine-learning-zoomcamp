# Homework 6: Decision Trees and Ensemble Learning

* **Course Portal:** [ML Zoomcamp 2026 HW06 Submission Page](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw06)
* **Dataset:** [Car Fuel Efficiency 2026 (`car_fuel_efficiency_2026.csv`)](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv)
* **Solution Script:** [`homework_6.py`](./homework_6.py)
* **Notebook:** [`homework_6.ipynb`](./homework_6.ipynb)

---

## 📋 Problem Setup
* **Target:** `fuel_efficiency_mpg` (continuous regression)
* **Missing Value Imputation:** Missing values filled with zeros (`fillna(0)`).
* **Data Split:**
  ```python
  df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
  df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
  ```
* **Feature Matrix:** `DictVectorizer(sparse=True)` fitted on `train` dictionary records and transformed on `val`/`test`.

---

## 🔍 Verified Answers & Explanations

### Q1. Decision Tree Split Feature (`max_depth=1`)
* Trained `DecisionTreeRegressor(max_depth=1, random_state=1)` on `X_train`.
* The single decision stump split on `model_year <= 78.5`.
* Mapping back to the original column name yields `model_year`.
* **Selected Option:** **`'model_year'`**

### Q2. Random Forest Validation RMSE (`n_estimators=10`)
* Trained `RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1)` on `X_train`.
* Validation RMSE: **1.8371** ($\approx \mathbf{1.837}$).
* **Selected Option:** **1.837**

### Q3. Tuning `n_estimators` for Lowest Validation RMSE
* Evaluated `n_estimators` $\in [10, 50, 100, 150]$ with `random_state=1`:
  * $n = 10$: $\text{RMSE} = 1.8371$
  * $n = 50$: $\text{RMSE} = 1.7716$
  * $n = 100$: $\text{RMSE} = \mathbf{1.7681}$ (Lowest)
  * $n = 150$: $\text{RMSE} = 1.7688$
* **Selected Option:** **100**

### Q4. Tuning `max_depth` across Forest Sizes
* Evaluated `max_depth` $\in [10, 15, 20, 25]$ averaged over `n_estimators` $\in [10, 50, 100, 150]$:
  * $\text{depth} = 10$: Mean $\text{RMSE} = \mathbf{1.7554}$ (Best)
  * $\text{depth} = 15$: Mean $\text{RMSE} = 1.7840$
  * $\text{depth} = 20$: Mean $\text{RMSE} = 1.7882$
  * $\text{depth} = 25$: Mean $\text{RMSE} = 1.7863$
* **Selected Option:** **10**

### Q5. Most Important Feature
* Extracted `feature_importances_` from `RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1)`:
  * `vehicle_weight`: **0.2039** (Highest)
  * `horsepower`: 0.0780
  * `engine_displacement`: 0.0610
  * `acceleration`: 0.0552
* **Selected Option:** **`vehicle_weight`**

### Q6. XGBoost Learning Rate (`eta`) Tuning
* Trained XGBoost regressor for 100 rounds with `reg:squarederror`, `max_depth=6`, `min_child_weight=1`, `seed=1`:
  * $\eta = 0.3$: Validation $\text{RMSE} = 1.8265$
  * $\eta = 0.1$: Validation $\text{RMSE} = \mathbf{1.7248}$ (Best)
* $\eta = 0.1$ achieves the lower validation error.
* **Selected Option:** **0.1**

---

## 🚀 Running the Code

```bash
uv run --with pandas --with scikit-learn --with xgboost python homework_6.py
```
