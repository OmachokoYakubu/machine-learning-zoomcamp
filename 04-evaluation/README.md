# Homework 4: Evaluation Metrics for Classification

* **Course Portal:** [ML Zoomcamp 2026 HW04 Submission Page](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw04)
* **Dataset:** [Course Lead Scoring 2026 Dataset (`course_lead_scoring_2026.csv`)](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv)
* **Solution Script:** [`homework_4.py`](./homework_4.py)
* **Notebook:** [`homework_4.ipynb`](./homework_4.ipynb)

---

## 📋 Problem Setup
* **Target:** `converted` (binary: 0 or 1)
* **Missing Value Imputation:** Categorical features imputed with `'NA'`, numerical features with `0.0`.
* **Data Split:**
  * Full train (80%) / Test (20%) with `random_state=1`
  * Train (60%) / Validation (20%) with `random_state=1`

---

## 🔍 Verified Answers & Explanations

### Q1. ROC AUC Feature Importance (Numerical Variables)
* Computed ROC AUC of each numerical feature against `converted` on the training dataset (inverting if $\text{AUC} < 0.5$):
  * `lead_score`: **0.7882**
  * `interaction_count`: **0.7649**
  * `number_of_courses_viewed`: **0.7230**
  * `annual_income`: **0.6082**
* **Selected Option:** **lead_score** (`radio-671-1`)

### Q2. Model Training and Validation AUC
* Trained `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)` with `DictVectorizer(sparse=False)`.
* Validation ROC AUC: $0.7318 \dots \approx \mathbf{0.732}$.
* **Selected Option:** **0.732** (`radio-672-3`)

### Q3. Precision and Recall Curve Intersection
* Evaluated precision and recall for thresholds $t \in [0.0, 1.0]$ with step $0.01$.
* Minimum absolute difference $|P - R|$ occurs at threshold $t = 0.63$ ($P = 0.7328, R = 0.7253$, diff $= 0.0075$).
* **Selected Option:** **0.63** (`radio-673-2`)

### Q4. Maximal F1 Score Threshold
* Computed $F_1 = 2 \cdot \frac{P \cdot R}{P + R}$ across all thresholds.
* Peak $F_1$ score is achieved at threshold $t = 0.41$ ($F_1 = 0.7576$).
* **Selected Option:** **0.41** (`radio-674-2`)

### Q5. 5-Fold Cross-Validation Standard Deviation
* Evaluated 5-fold CV (`KFold(n_splits=5, shuffle=True, random_state=1)`) on `df_full_train`.
* Standard deviation of ROC AUC scores across the 5 folds: $\sigma = 0.0069 \dots \approx \mathbf{0.007}$.
* **Selected Option:** **0.007** (`radio-675-2`)

### Q6. Hyperparameter Tuning Across $C$
* Evaluated $C \in [0.000001, 0.001, 1.0]$ using 5-fold CV:
  * $C = 0.000001$: Mean $\text{AUC} = 0.617$, Std $= 0.019$
  * $C = 0.001$: Mean $\text{AUC} = \mathbf{0.749}$, Std $= 0.010$
  * $C = 1.0$: Mean $\text{AUC} = 0.732$, Std $= 0.007$
* The highest mean ROC AUC is achieved with $C = 0.001$.
* **Selected Option:** **0.001** (`radio-676-2`)
