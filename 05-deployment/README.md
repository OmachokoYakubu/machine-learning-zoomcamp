# Homework 5: Deploying Machine Learning Models

* **Course Portal:** [ML Zoomcamp 2026 HW05 Submission Page](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw05)
* **Pinned Model Artifact:** [`pipeline.bin`](./pipeline.bin) (SHA256: `1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1`)
* **Solution Script:** [`homework_5.py`](./homework_5.py)
* **FastAPI Service:** [`predict.py`](./predict.py)
* **Dockerfile:** [`Dockerfile`](./Dockerfile)

---

## 🔍 Verified Answers & Explanations

### Q1. Environment Check (`uv --version`)
* Local `uv` version installed in the environment:
```bash
uv --version # uv 0.9.16
```
* **Answer:** `0.9.16`

### Q2. Scikit-Learn Version in Reference Environment
* From [`model_metadata.json`](./model_metadata.json) and [`pyproject.toml`](./pyproject.toml):
```json
"scikit_learn": "1.7.2"
```
* **Selected Option:** **1.7.2** (`radio-678-2`)

### Q3. Conversion Probability (Loaded Model)
* Evaluated lead:
```json
{
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}
```
* Raw probability: `0.532994` $\rightarrow \mathbf{0.533}$.
* **Answer:** `0.533`

### Q4. Conversion Probability (FastAPI Service)
* Evaluated lead:
```json
{
  "lead_source": "organic_search",
  "industry": "technology",
  "employment_status": "employed",
  "location": "europe",
  "number_of_courses_viewed": 4,
  "annual_income": 80304.0,
  "interaction_count": 7,
  "lead_score": 0.74
}
```
* Raw probability: `0.769799` $\rightarrow \mathbf{0.770}$.
* **Answer:** `0.770`

### Q5. Python Base Image
* Declared in [`Dockerfile:L1`](./Dockerfile#L1):
```dockerfile
FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3
```
* **Selected Option:** **python:3.11.15-slim-bookworm** (`radio-681-2`)

### Q6. Conversion Probability (Container)
* Running the prediction for the same lead against the containerized FastAPI service yields the exact reference inference:
* Raw probability: `0.769799` $\rightarrow \mathbf{0.770}$.
* **Answer:** `0.770`
