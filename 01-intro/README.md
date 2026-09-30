# Homework 1: Introduction to Machine Learning

* **Course Portal:** [ML Zoomcamp 2026 HW01 Submission Page](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw01)
* **Dataset:** [Car Fuel Efficiency 2026 Dataset (`car_fuel_efficiency_2026.csv`)](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv)
* **Solution Script:** [`homework_1.py`](./homework_1.py)

---

## 📋 Verified Answers & Solutions

### Q1. Pandas Version
```python
import pandas as pd
pd.__version__
```
* **Answer:** `2.2.3`

### Q2. Records Count
```python
len(df)
```
* **Result:** `10000`
* **Selected Option:** **10000** (`radio-653-3`)

### Q3. Fuel Types
```python
df['fuel_type'].nunique()  # ['Gasoline', 'Diesel', 'Hybrid']
```
* **Result:** `3`
* **Selected Option:** **3** (`radio-654-3`)

### Q4. Missing Values
Columns with missing values: `horsepower` (877) and `acceleration` (264).
```python
(df.isnull().sum() > 0).sum()
```
* **Result:** `2`
* **Selected Option:** **2** (`radio-655-3`)

### Q5. Max Fuel Efficiency (Asia)
```python
df[df['origin'] == 'Asia']['fuel_efficiency_mpg'].max()
```
* **Result:** `41.2`
* **Selected Option:** **41.2** (`radio-656-3`)

### Q6. Median Value of Horsepower
* Median before filling: `254.0`
* Mode value: `252.0`
* Median after filling with mode: `252.0`
```python
# 254.0 -> 252.0
```
* **Result:** Decreased
* **Selected Option:** **Yes, it decreased** (`radio-657-2`)

### Q7. Sum of Weights (Linear Regression)
```python
asia_7 = df[df['origin'] == 'Asia'][['vehicle_weight', 'model_year']].head(7)
X = asia_7.to_numpy()
XTX = X.T @ X
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = np.linalg.inv(XTX) @ X.T @ y
w.sum()  # 0.3692
```
* **Result:** `0.369`
* **Selected Option:** **0.369** (`radio-658-2`)
