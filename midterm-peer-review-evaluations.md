# Machine Learning Zoomcamp 2026 — Midterm Project Peer-Review Evaluations

**Reviewer:** Omachoko Yakubu  
**Review Date:** 22 November 2026  
**Course Portal:** [DataTalks.Club ML Zoomcamp 2026 — Midterm Project Evaluations](https://courses.datatalks.club/ml-zoomcamp-2026/project/midterm/eval)  
**Review Deadline:** Wednesday, 25 November 2026, 12:00  
**Rubric Source:** Midterm Project Peer Evaluation Form (9 Criteria, 16 Points Maximum)

---

## 1. Assigned Submissions Summary

| Assignment | Peer Pseudonym | Project / Repository | Evaluated Commit | Portal Link | Score |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **#14893** | **Peaceful Raman** | [`daler-rahimov/customer_churn_prediction_ml`](https://github.com/daler-rahimov/customer_churn_prediction_ml) *(HTTP 404 — Private / Not Found)* | `1b76f3f` | [Eval #14893](https://courses.datatalks.club/ml-zoomcamp-2026/project/midterm/eval/14893) | **0 / 16** |
| **#14894** | **Stoic Hugle** | [`tishapereira20/flight_price_prediction`](https://github.com/tishapereira20/flight_price_prediction) *(HTTP 404 — Private / Not Found)* | `07d6d5c` | [Eval #14894](https://courses.datatalks.club/ml-zoomcamp-2026/project/midterm/eval/14894) | **0 / 16** |
| **#14895** | **Eager Kepler** | [**NFL Q3 Win Probability** (`VegasKevin/NFL_Game_ML_Model_mid`)](https://github.com/VegasKevin/NFL_Game_ML_Model_mid) | `f5165f9` | [Eval #14895](https://courses.datatalks.club/ml-zoomcamp-2026/project/midterm/eval/14895) | **11 / 16** |

---

## 2. Comparative Score Matrix (9 Criteria)

| # | Criterion | Max Pts | #14893 (`customer_churn`) | #14894 (`flight_price`) | #14895 (`NFL_Game_ML_Model_mid`) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **Problem description** | 2 | 0 | 0 | **2** |
| **2** | **EDA** | 2 | 0 | 0 | **1** |
| **3** | **Model training** | 3 | 0 | 0 | **3** |
| **4** | **Exporting notebook to script** | 1 | 0 | 0 | **1** |
| **5** | **Reproducibility** | 1 | 0 | 0 | **1** |
| **6** | **Model deployment** | 1 | 0 | 0 | **0** |
| **7** | **Dependency and environment management** | 2 | 0 | 0 | **1** |
| **8** | **Containerization** | 2 | 0 | 0 | **2** |
| **9** | **Cloud deployment** | 2 | 0 | 0 | **0** |
| | **Total Score** | **16** | **0 / 16** | **0 / 16** | **11 / 16** |

---

## 3. Detailed Evaluation #14893 — `customer_churn_prediction_ml` (Peaceful Raman)

- **Repository URL:** `https://github.com/daler-rahimov/customer_churn_prediction_ml`
- **Commit ID:** `1b76f3f`
- **Accessibility Check:** Accessing `https://github.com/daler-rahimov/customer_churn_prediction_ml` via browser, `git clone`, and the GitHub REST API returns **HTTP 404 (Repository not found)**.
- **Note to Peer:**
  > Unfortunately, `https://github.com/daler-rahimov/customer_churn_prediction_ml` returns HTTP 404 (Repository not found) when accessed publicly or via `git clone` during the peer evaluation window. Please set the GitHub repository visibility to Public so reviewers can inspect your Midterm Project.

---

## 4. Detailed Evaluation #14894 — `flight_price_prediction` (Stoic Hugle)

- **Repository URL:** `https://github.com/tishapereira20/flight_price_prediction`
- **Commit ID:** `07d6d5c`
- **Accessibility Check:** Accessing `https://github.com/tishapereira20/flight_price_prediction` via browser, `git clone`, and the GitHub REST API returns **HTTP 404 (Repository not found)**.
- **Note to Peer:**
  > Unfortunately, `https://github.com/tishapereira20/flight_price_prediction` returns HTTP 404 (Repository not found) when accessed publicly or via `git clone` during the peer evaluation window. Please ensure the repository is Public and the GitHub URL is accessible.

---

## 5. Detailed Evaluation #14895 — `NFL_Game_ML_Model_mid` (Eager Kepler)

- **Repository URL:** `https://github.com/VegasKevin/NFL_Game_ML_Model_mid`
- **Commit ID:** `f5165f9`

### Criterion-by-Criterion Breakdown
1. **Problem description (2/2 — `radio-217-3`):** `README.md` clearly defines the problem: predicting an NFL team's win probability at the end of Q3 using `nflverse` play-by-play data (2016–2025 regular season and playoffs), why two rows per game must be grouped by `game_id` across splits to prevent leakage, and how the pipeline operates.
2. **EDA (1/2 — `radio-218-2`):** `Data_Import.ipynb` and `Clean_Data_1.py` inspect dataset shapes, missing-value counts before and after cleaning, domain-specific imputation (`previous_season.py` for Week 1 openers and 0-filling third-down conversion rates), and export a hyperparameter ROC AUC heatmap (`tree_auc_heatmeap.png`). Expanding target distribution plots and feature-correlation charts would elevate this to 2/2.
3. **Model training (3/3 — `radio-219-4`):** `Clean_Data_1.py` splits by unique `game_id` into 60/20/20 Train/Val/Test sets, compares `DecisionTreeClassifier` and `LogisticRegression`, and systematically grid-searches 11 `max_depth` values (`[2..15, None]`) × 10 `min_samples_split` values (`[2..500]`), selecting `max_depth=4, min_samples_split=500` (`ROC AUC = 0.8637`).
4. **Exporting notebook to script (1/1 — `radio-220-2`):** All data ingestion, feature engineering, imputation, and model training logic are cleanly modularized into `pipeline.py`, `previous_season.py`, and `Clean_Data_1.py` (with unit tests in `tests/test_pipeline.py`).
5. **Reproducibility (1/1 — `radio-221-2`):** `pipeline.py` automatically downloads and caches the 2016–2025 play-by-play CSV releases from `github.com/nflverse/nflverse-data` and writes `data/processed/team_games_q3_2016_2025.parquet`.
6. **Model deployment (0/1 — `radio-222-1`):** As noted in `README.md` (*"Model scores are exploratory validation results, not a deployed prediction service"*), no Flask/FastAPI/BentoML prediction endpoint is included.
7. **Dependency and environment management (1/2 — `radio-223-2`):** Pinned `requirements.txt` (`pandas==2.2.3`, `numpy==2.2.6`, `pyarrow==19.0.1`, `scikit-learn==1.6.1`, `seaborn==0.13.2`, `matplotlib==3.11.2`) is provided, though local virtual environment creation/activation instructions are omitted from `README.md`.
8. **Containerization (2/2 — `radio-224-3`):** `Dockerfile` (`python:3.11.11-slim`) and `compose.yaml` are provided with clear build and execution commands (`docker compose build`, `docker compose run --rm pipeline`, `docker compose run --rm pipeline Clean_Data_1.py`).
9. **Cloud deployment (0/2 — `radio-225-1`):** No cloud or Kubernetes deployment configuration is included.

- **Note to Peer:**
  > Thoughtful domain feature engineering in `pipeline.py` and `previous_season.py` (especially grouping train/val/test splits by `game_id` so opponent rows never leak across splits, and backfilling Week 1 opener stats from the prior season)! For the Capstone project, consider serializing the trained champion model (`pickle`/`joblib`) and serving it via a FastAPI/Flask `/predict` endpoint inside your Docker container so it earns full deployment and cloud points.

