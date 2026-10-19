# Machine Learning Zoomcamp 2026: Master Course Playbook

**Student:** Omachoko Yakubu  
**Repository:** [github.com/OmachokoYakubu/machine-learning-zoomcamp](https://github.com/OmachokoYakubu/machine-learning-zoomcamp)  
**Course Portal:** [DataTalks.Club Machine Learning Zoomcamp 2026](https://courses.datatalks.club/ml-zoomcamp-2026/)  

---

## 📅 Cohort 2026 Timeline & Schedule

| Week / Module | Topic | Deadline | Status | Deliverables |
| :--- | :--- | :--- | :---: | :--- |
| **Module 01** | **Introduction to Machine Learning** | 2026-09-28 | **Completed** | Pandas, NumPy, Fuel Efficiency dataset, Matrix Inverse |
| **Module 02** | **Machine Learning for Regression** | 2026-10-05 | **Completed** | Train/Val/Test split, Linear Regression, RMSE, Regularization |
| **Module 03** | **Machine Learning for Classification** | 2026-10-12 | **Completed** | Lead scoring, Logistic Regression, Mutual info, Feature elimination |
| **Module 04** | **Evaluation Metrics for Classification** | 2026-10-19 | **Completed** | ROC AUC, Precision/Recall intersection, F1 max, 5-Fold CV |
| **Module 05** | **Deploying Machine Learning Models** | 2026-10-26 | **Completed** | Pinned artifact inference, FastAPI service, Docker container |
| **Module 06** | **Decision Trees & Ensemble Learning** | 2026-11-02 | Upcoming | Decision trees, Random Forests, XGBoost |
| **Midterm** | **Midterm Project** | 2026-11-16 | Upcoming | End-to-end ML project submission & peer review |
| **Module 08** | **Deep Learning & Neural Networks** | 2026-11-30 | Upcoming | Computer vision, CNNs, image classification |
| **Module 09** | **Serverless Deep Learning** | 2026-12-07 | Upcoming | Serverless deployment, AWS Lambda, Docker |
| **Module 10** | **Kubernetes & Model Serving** | 2026-12-14 | Upcoming | Kubernetes clusters, ingress, horizontal pod autoscaling |
| **Capstone 1** | **Capstone Project 1** | 2027-01-11 | Upcoming | Capstone project submission & peer review |

---

## 📝 Verified Homework Submission Solutions

### Homework 1: Introduction to Machine Learning
* **Portal Link:** [courses.datatalks.club/ml-zoomcamp-2026/homework/hw01](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw01)
* **Dataset:** `car_fuel_efficiency_2026.csv` (10,000 records)

| Question | Topic | Solution / Output | Selected Option | Form ID |
| :--- | :--- | :--- | :--- | :--- |
| **Q1** | **Pandas Version** | `2.2.3` | Text input: `2.2.3` | `answer_652` |
| **Q2** | **Records Count** | `10000` | **10000** | `radio-653-3` |
| **Q3** | **Fuel Types** | `3` (`['Gasoline', 'Diesel', 'Hybrid']`) | **3** | `radio-654-3` |
| **Q4** | **Missing Values Columns** | `2` (`horsepower` [877], `acceleration` [264]) | **2** | `radio-655-3` |
| **Q5** | **Max Fuel Efficiency (Asia)** | `41.2` mpg | **41.2** | `radio-656-3` |
| **Q6** | **Median Horsepower after Mode Fill** | Median decreased (254.0 $\rightarrow$ 252.0, mode=252.0) | **Yes, it decreased** | `radio-657-2` |
| **Q7** | **Sum of Weights (Linear Regression)** | $w = [0.1364, 0.2327]$, $\sum w = 0.3692$ | **0.369** | `radio-658-2` |

---

### Homework 2: Machine Learning for Regression
* **Portal Link:** [courses.datatalks.club/ml-zoomcamp-2026/homework/hw02](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw02)
* **Dataset:** `car_fuel_efficiency_2026.csv` (`fuel_efficiency_mpg` target)

| Question | Topic | Solution / Output | Selected Option | Form ID |
| :--- | :--- | :--- | :--- | :--- |
| **Q1** | **Missing Values Column** | `horsepower` (877 missing out of 10,000) | **horsepower** | `radio-659-2` |
| **Q2** | **Median for Horsepower** | `254` | **254** | `radio-660-2` |
| **Q3** | **Filling NAs Comparison** | Mean imputation ($\text{RMSE}=2.202$) is slightly better than Zero ($\text{RMSE}=2.205$) | **With mean** | `radio-661-2` |
| **Q4** | **Best Regularization ($r$)** | Regularization values give identical 4-digit RMSE (2.2053), select smallest $r$ | **0** | `radio-662-1` |
| **Q5** | **RMSE Standard Deviation** | $\sigma = 0.02878 \approx 0.029$ across 10 random seeds `[0..9]` | **0.029** | `radio-663-3` |
| **Q6** | **Test Evaluation ($r=0.001$, seed 9)** | $\text{RMSE}_{\text{test}} = 2.2358 \approx 2.236$ | **2.236** | `radio-664-2` |

---

### Homework 3: Machine Learning for Classification
* **Portal Link:** [courses.datatalks.club/ml-zoomcamp-2026/homework/hw03](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw03)
* **Dataset:** `course_lead_scoring_2026.csv` (`converted` target)

| Question | Topic | Solution / Output | Selected Option | Form ID |
| :--- | :--- | :--- | :--- | :--- |
| **Q1** | **Mode for `industry`** | `technology` (1,173 records) | **technology** | `radio-665-2` |
| **Q2** | **Biggest Numerical Correlation** | `interaction_count` and `lead_score` ($r = 0.9157$) | **interaction_count and lead_score** | `radio-666-1` |
| **Q3** | **Highest Mutual Information** | `lead_source` ($\text{MI} = 0.0343$) | **lead_source** | `radio-667-3` |
| **Q4** | **Logistic Regression Accuracy** | Validation accuracy $= 0.6450 \approx 0.65$ | **0.65** | `radio-668-2` |
| **Q5** | **Feature Elimination (Smallest Diff)** | Without `number_of_courses_viewed`, accuracy diff is smallest ($+0.0020$) | **number_of_courses_viewed** | `radio-669-2` |
| **Q6** | **Regularization Parameter $C$** | $C = 0.001$ yields highest validation accuracy ($0.645$) | **0.001** | `radio-670-4` |

---

### Homework 4: Evaluation Metrics for Classification
* **Portal Link:** [courses.datatalks.club/ml-zoomcamp-2026/homework/hw04](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw04)
* **Dataset:** `course_lead_scoring_2026.csv` (`converted` target)

| Question | Topic | Solution / Output | Selected Option | Form ID |
| :--- | :--- | :--- | :--- | :--- |
| **Q1** | **ROC AUC Feature Importance** | `lead_score` ($\text{AUC} = 0.7882$) | **lead_score** | `radio-671-1` |
| **Q2** | **Model ROC AUC on Validation** | Validation $\text{ROC AUC} = 0.7318 \approx 0.732$ | **0.732** | `radio-672-3` |
| **Q3** | **Precision/Recall Intersection** | Intersection occurs at threshold $t = 0.63$ ($|P - R| = 0.0075$) | **0.63** | `radio-673-2` |
| **Q4** | **Maximal F1 Score Threshold** | Maximal $F_1 = 0.7576$ achieved at threshold $t = 0.41$ | **0.41** | `radio-674-2` |
| **Q5** | **5-Fold Cross-Validation Std** | Standard deviation across 5 folds $\sigma = 0.0069 \approx 0.007$ | **0.007** | `radio-675-2` |
| **Q6** | **Hyperparameter Tuning ($C$)** | $C = 0.001$ gives best mean $\text{ROC AUC} = 0.749$ | **0.001** | `radio-676-2` |

---

### Homework 5: Deploying Machine Learning Models
* **Portal Link:** [courses.datatalks.club/ml-zoomcamp-2026/homework/hw05](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw05)
* **Artifact:** `pipeline.bin` (SHA256: `1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1`)

| Question | Topic | Solution / Output | Selected Option / Value | Form ID |
| :--- | :--- | :--- | :--- | :--- |
| **Q1** | **uv version** | Local setup check | Text: `0.9.16` | `answer_677` |
| **Q2** | **Scikit-Learn Locked Version** | From metadata and pyproject.toml | **1.7.2** | `radio-678-2` |
| **Q3** | **Conversion Probability (Loaded Model)** | Loaded pipeline on Q3 lead: $0.532994 \approx 0.533$ | Text: `0.533` | `answer_679` |
| **Q4** | **Conversion Probability (API)** | FastAPI service on Q4 lead: $0.769799 \approx 0.770$ | Text: `0.770` | `answer_680` |
| **Q5** | **Python Base Image** | Declared in Dockerfile | **python:3.11.15-slim-bookworm** | `radio-681-2` |
| **Q6** | **Conversion Probability (Container)** | Container request on Q4 lead: $0.769799 \approx 0.770$ | Text: `0.770` | `answer_682` |

