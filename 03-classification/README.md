# Homework 3: Machine Learning for Classification

* **Course Portal:** [ML Zoomcamp 2026 HW03 Submission Page](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw03)
* **Dataset:** [Course Lead Scoring 2026 Dataset (`course_lead_scoring_2026.csv`)](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv)
* **Solution Script:** [`homework_3.py`](./homework_3.py)
* **Notebook:** [`homework_3.ipynb`](./homework_3.ipynb)

---

## 📋 Features & Preprocessing
* **Target:** `converted` (binary: 0 or 1)
* **Categorical Features:** `lead_source`, `industry`, `employment_status`, `location` (missing values imputed with `'NA'`)
* **Numerical Features:** `annual_income`, `number_of_courses_viewed`, `interaction_count`, `lead_score` (missing values imputed with `0.0`)
* **Data Split:** 60% train, 20% validation, 20% test with `random_state=42`.

---

## 🔍 Verified Answers & Explanations

### Q1. Mode for `industry`
```python
df['industry'].mode()[0]  # 'technology' (1,173 observations)
```
* **Selected Option:** **technology** (`radio-665-2`)

### Q2. Features with the Biggest Correlation
* Correlation matrix for numerical features:
  * `corr(interaction_count, lead_score) = 0.9157`
  * `corr(number_of_courses_viewed, lead_score) = 0.7572`
  * `corr(number_of_courses_viewed, interaction_count) = 0.7216`
  * `corr(annual_income, interaction_count) = 0.1228`
* **Selected Option:** **interaction_count and lead_score** (`radio-666-1`)

### Q3. Feature with the Highest Mutual Information
* Evaluated against `converted` on the training set:
  * `lead_source`: **0.0343** (rounded: `0.03`)
  * `employment_status`: **0.0190** (rounded: `0.02`)
  * `industry`: **0.0020** (rounded: `0.00`)
  * `location`: **0.0010** (rounded: `0.00`)
* **Selected Option:** **lead_source** (`radio-667-3`)

### Q4. Model Validation Accuracy
* Trained `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)` with `DictVectorizer(sparse=False)`.
* Validation accuracy: $0.6450 \rightarrow \mathbf{0.65}$.
* **Selected Option:** **0.65** (`radio-668-2`)

### Q5. Feature Elimination (Smallest Difference)
* Baseline accuracy: `0.6450`
  * Exclude `number_of_courses_viewed`: accuracy `0.6430` (difference: `+0.0020`)
  * Exclude `lead_source`: accuracy `0.6420` (difference: `+0.0030`)
  * Exclude `interaction_count`: accuracy `0.6010` (difference: `+0.0440`)
* **Selected Option:** **number_of_courses_viewed** (`radio-669-2`)

### Q6. Regularization Parameter $C$
* Evaluated $C \in [0.000001, 0.00001, 0.0001, 0.001]$:
  * $C = 0.000001$: accuracy = `0.598`
  * $C = 0.00001$: accuracy = `0.598`
  * $C = 0.0001$: accuracy = `0.613`
  * $C = 0.001$: accuracy = **`0.645`**
* **Selected Option:** **0.001** (`radio-670-4`)
