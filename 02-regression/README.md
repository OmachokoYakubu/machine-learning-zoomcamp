# Homework 2: Machine Learning for Regression

* **Course Portal:** [ML Zoomcamp 2026 HW02 Submission Page](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw02)
* **Dataset:** [Car Fuel Efficiency 2026 Dataset (`car_fuel_efficiency_2026.csv`)](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv)
* **Solution Script:** [`homework_2.py`](./homework_2.py)

---

## 📋 Features & Columns Used
* `engine_displacement`
* `horsepower`
* `vehicle_weight`
* `model_year`
* **Target:** `fuel_efficiency_mpg`

---

## 🔍 Verified Answers & Explanations

### Q1. Missing Values
* `horsepower` has 877 missing entries out of 10,000 records.
* **Selected Option:** **horsepower** (`radio-659-2`)

### Q2. Median for Horsepower
```python
df['horsepower'].median()  # 254.0
```
* **Selected Option:** **254** (`radio-660-2`)

### Q3. Filling NAs (With 0 vs With Mean)
* **RMSE with 0:** `2.205`
* **RMSE with mean:** `2.202`
* Imputing with the training mean yields lower error on the validation set.
* **Selected Option:** **With mean** (`radio-661-2`)

### Q4. Best Regularization ($r$)
* Tested $r \in [0, 0.01, 0.1, 1, 5, 10, 100]$.
* Scores rounded to 4 decimals are identical (`2.2053`).
* Following the rule: *"If multiple options give the same best RMSE, select the smallest $r$"*.
* **Selected Option:** **0** (`radio-662-1`)

### Q5. RMSE Standard Deviation (Seed Stability)
* Tested seeds `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`.
* Standard deviation across all 10 folds: $\sigma = 0.02878 \dots \approx 0.029$.
* **Selected Option:** **0.029** (`radio-663-3`)

### Q6. Evaluation on Test Dataset
* Seed: `9`
* Train + Validation combined, evaluated on Test.
* Regularization: $r = 0.001$, NAs filled with 0.
* $\text{RMSE}_{\text{test}} = 2.2358 \dots \approx 2.236$.
* **Selected Option:** **2.236** (`radio-664-2`)
