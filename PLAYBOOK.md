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
| **Module 03** | **Machine Learning for Classification** | 2026-10-12 | Upcoming | Churn prediction, Logistic Regression, Feature importance |
| **Module 04** | **Evaluation Metrics for Classification** | 2026-10-19 | Upcoming | Confusion matrix, ROC curves, K-Fold CV |
| **Module 05** | **Deploying Machine Learning Models** | 2026-10-26 | Upcoming | Model serialization, web service API, Docker container |
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
