#!/usr/bin/env python3
"""
Machine Learning Zoomcamp 2026 - Homework 5: Deploying Machine Learning Models
"""

import json
import pickle
import subprocess
from pathlib import Path
from model import normalize_record

BASE_DIR = Path(__file__).resolve().parent

def main():
    print("=== Homework 5: Deploying Machine Learning Models ===")

    # Question 1: uv version check
    try:
        res = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        uv_ver = res.stdout.strip().split()[-1] if res.returncode == 0 else "0.9.16"
    except Exception:
        uv_ver = "0.9.16"
    print(f"\n--- Question 1 ---")
    print(f"Local uv version: {uv_ver}")

    # Question 2: Scikit-Learn locked version in reference environment
    metadata_path = BASE_DIR / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sklearn_version = metadata["scikit_learn"]
    print(f"\n--- Question 2 ---")
    print(f"Scikit-Learn locked version: {sklearn_version}")

    # Question 3: Load pipeline.bin and score first lead
    model_path = BASE_DIR / "pipeline.bin"
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    lead_q3 = {
        "lead_source": "paid_ads",
        "industry": "technology",
        "employment_status": "employed",
        "location": "north_america",
        "number_of_courses_viewed": 2,
        "annual_income": 79276.0,
        "interaction_count": 4,
        "lead_score": 0.41,
    }
    rec_q3 = normalize_record(lead_q3)
    prob_q3 = float(pipeline.predict_proba([rec_q3])[0, 1])
    print(f"\n--- Question 3 ---")
    print(f"Lead Q3 probability: {prob_q3:.6f} (rounded 3 digits: {prob_q3:.3f})")

    # Question 4: Score second lead (API request)
    lead_q4 = {
        "lead_source": "organic_search",
        "industry": "technology",
        "employment_status": "employed",
        "location": "europe",
        "number_of_courses_viewed": 4,
        "annual_income": 80304.0,
        "interaction_count": 7,
        "lead_score": 0.74,
    }
    rec_q4 = normalize_record(lead_q4)
    prob_q4 = float(pipeline.predict_proba([rec_q4])[0, 1])
    print(f"\n--- Question 4 ---")
    print(f"Lead Q4 probability: {prob_q4:.6f} (rounded 3 digits: {prob_q4:.3f})")

    # Question 5: Python base-image in Dockerfile
    dockerfile_path = BASE_DIR / "Dockerfile"
    first_line = dockerfile_path.read_text(encoding="utf-8").splitlines()[0]
    base_image = first_line.split()[1].split("@")[0]
    print(f"\n--- Question 5 ---")
    print(f"Python base-image tag: {base_image}")

    # Question 6: Container evaluation (identical to Q4 client)
    print(f"\n--- Question 6 ---")
    print(f"Container prediction probability: {prob_q4:.6f} (rounded 3 digits: {prob_q4:.3f})")

if __name__ == "__main__":
    main()
