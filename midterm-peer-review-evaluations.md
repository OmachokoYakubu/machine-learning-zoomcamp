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
| **#14893** | **Peaceful Raman** | [**Stellar Classification (SDSS17)** (`Sanyaman58/zoomcamp2026_midterm_project`)](https://github.com/Sanyaman58/zoomcamp2026_midterm_project) | `a37ee87` | [Eval #14893](https://courses.datatalks.club/ml-zoomcamp-2026/project/midterm/eval/14893) | **14 / 16** |
| **#14894** | **Stoic Hugle** | [**Shopping Purchase Prediction** (`salexyun/shopping-purchase-prediction`)](https://github.com/salexyun/shopping-purchase-prediction) | `87c14e5` | [Eval #14894](https://courses.datatalks.club/ml-zoomcamp-2026/project/midterm/eval/14894) | **16 / 16** |
| **#14895** | **Eager Kepler** | [**NFL Q3 Win Probability** (`VegasKevin/NFL_Game_ML_Model_mid`)](https://github.com/VegasKevin/NFL_Game_ML_Model_mid) | `f5165f9` | [Eval #14895](https://courses.datatalks.club/ml-zoomcamp-2026/project/midterm/eval/14895) | **11 / 16** |

---

## 2. Comparative Score Matrix (9 Criteria)

| # | Criterion | Max Pts | #14893 (`Stellar SDSS17`) | #14894 (`Shopping Purchase`) | #14895 (`NFL Q3 Win Prob`) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **Problem description** | 2 | **2** | **2** | **2** |
| **2** | **EDA** | 2 | **2** | **2** | **1** |
| **3** | **Model training** | 3 | **3** | **3** | **3** |
| **4** | **Exporting notebook to script** | 1 | **1** | **1** | **1** |
| **5** | **Reproducibility** | 1 | **1** | **1** | **1** |
| **6** | **Model deployment** | 1 | **1** | **1** | **0** |
| **7** | **Dependency and environment management** | 2 | **2** | **2** | **1** |
| **8** | **Containerization** | 2 | **2** | **2** | **2** |
| **9** | **Cloud deployment** | 2 | **0** | **2** | **0** |
| | **Total Score** | **16** | **14 / 16** | **16 / 16** | **11 / 16** |

---

## 3. Detailed Evaluation #14893 — `zoomcamp2026_midterm_project` (Peaceful Raman)

- **Repository URL:** `https://github.com/Sanyaman58/zoomcamp2026_midterm_project`
- **Commit ID:** `a37ee87`

### Criterion-by-Criterion Breakdown
1. **Problem description (2/2 — `radio-217-3`):** `README.md` clearly describes the SDSS17 stellar object classification problem—predicting whether a celestial object is a `STAR`, `GALAXY`, or `QSO` (Quasar) from 8 photometric and redshift parameters—and explains the domain motivation.
2. **EDA (2/2 — `radio-218-3`):** `notebook.ipynb` conducts outlier/anomaly filtering (`g > -9999`), class distribution analysis, correlation matrix inspection, mutual information scoring, and feature importance comparison.
3. **Model training (3/3 — `radio-219-4`):** Trains and tunes multiple model families (`LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, and multi-class `XGBoost`), selecting tuned `XGBoost` (`eta=0.1, max_depth=6, min_child_weight=5`).
4. **Exporting notebook to script (1/1 — `radio-220-2`):** Model training pipeline is exported cleanly to `train.py`.
5. **Reproducibility (1/1 — `radio-221-2`):** `star_classification.csv` is committed directly in the repository, allowing `notebook.ipynb` and `train.py` to run end to end without external data setup.
6. **Model deployment (1/1 — `radio-222-2`):** `predict.py` serves the serialized `xgb_model_depth_6.bin` model via a Flask `/predict` endpoint (`port 9696`), paired with `predict-test.py`.
7. **Dependency and environment management (2/2 — `radio-223-3`):** `Pipfile` and `Pipfile.lock` are included with clear `pipenv install` and `pipenv run` commands in `README.md`.
8. **Containerization (2/2 — `radio-224-3`):** `Dockerfile` is provided and `README.md` documents `docker build -t stellar-classification .` and `docker run -it --rm -p 9696:9696 stellar-classification`.
9. **Cloud deployment (0/2 — `radio-225-1`):** No cloud or Kubernetes deployment is included.

- **Note to Peer:**
  > Very clean end-to-end astronomy classification project! Committing `star_classification.csv`, providing both `Pipfile`/`Pipfile.lock` and a `Dockerfile`, and comparing 4 distinct classifiers (`LogisticRegression`, `DecisionTree`, `RandomForest`, `XGBoost`) made reproduction effortless. In `train.py` line 116, consider passing `(dval, 'val')` instead of `(dtest, 'test')` to `evals` during training and using `gunicorn`/`waitress` in your `Dockerfile` entrypoint.

---

## 4. Detailed Evaluation #14894 — `shopping-purchase-prediction` (Stoic Hugle)

- **Repository URL:** `https://github.com/salexyun/shopping-purchase-prediction`
- **Commit ID:** `87c14e5`

### Criterion-by-Criterion Breakdown
1. **Problem description (2/2 — `radio-217-3`):** `README.md` frames the e-commerce session purchase ranking problem on the UCI Online Shoppers Purchasing Intention dataset (`12,330` sessions, `15.5%` positive rate), explains why leakage-prone exit features (`PageValues`, `BounceRates`, `ExitRates`) are excluded, and discusses deployment calibration limits.
2. **EDA (2/2 — `radio-218-3`):** `notebook.ipynb` inspects missing values, numerical ranges, duplicate predictor profiles, class imbalance, precision-recall curves, calibration curves, and permutation feature importance.
3. **Model training (3/3 — `radio-219-4`):** Compares a prior-only baseline, tuned `LogisticRegression`, `HistGradientBoostingClassifier`, and `RandomForestClassifier` with group-aware 3-fold cross-validation optimizing Average Precision (`0.371` CV AP, `0.321` holdout AP, `0.750` holdout ROC AUC).
4. **Exporting notebook to script (1/1 — `radio-220-2`):** Exported to `train.py`, writing `artifacts/model.joblib` and `artifacts/model.metrics.json`.
5. **Reproducibility (1/1 — `radio-221-2`):** `data/online_shoppers_intention.csv` (`SHA-256: b3055ee3...`) is committed in the repo with deterministic `uv sync --locked` instructions.
6. **Model deployment (1/1 — `radio-222-2`):** Deployed as a FastAPI service in `predict.py` with `/health`, `/predict`, and `/docs`.
7. **Dependency and environment management (2/2 — `radio-223-3`):** `pyproject.toml`, `uv.lock`, and `.python-version` (`3.12`) with clear `uv sync --locked` and `.venv` activation instructions in `README.md`.
8. **Containerization (2/2 — `radio-224-3`):** Multi-stage `Dockerfile` that trains and packages the artifact during image build, documented with `docker build`/`docker run` commands and `examples/docker-demo.png`.
9. **Cloud deployment (2/2 — `radio-225-3`):** Includes `render.yaml` Blueprint for Render Docker cloud deployment, step-by-step Render deployment instructions in `README.md`, and verification screenshot `examples/docker-demo.png`.

- **Note to Peer:**
  > Exemplary ML engineering! Keeping identical predictor profiles grouped across train/CV/holdout splits, excluding leakage-prone post-session features (`PageValues`, `BounceRates`, `ExitRates`), evaluating with Average Precision + calibration curves on an imbalanced target, and shipping a multi-stage `uv`-locked `Dockerfile` + `render.yaml` Blueprint makes this a model 16/16 submission.

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
