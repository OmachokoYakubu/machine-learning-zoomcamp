# Module 08: Neural Networks and Deep Learning (Homework 8)

**Course Portal Form:** [ML Zoomcamp 2026 — Homework 8](https://courses.datatalks.club/ml-zoomcamp-2026/homework/hw08)  
**Dataset:** Hair Type Dataset (`data.zip`, SHA-256: `9e53453e3017502f22860cb08558f0cdb343e102fb61feab75960616185e7d67`, `800` train / `201` held-out evaluation images across `curly=0` and `straight=1`)  
**Reference Environment:** Python `3.11`, `torch==2.9.0+cpu`, `torchvision==0.24.0+cpu`, `numpy==2.3.3`, `Pillow==11.3.0`, `SEED=42`

---

## 1. Verified Homework 8 Solutions

| # | Question | Raw Reference Output | Submitted Answer | Portal Field |
| :---: | :--- | :--- | :--- | :--- |
| **Q1** | **Loss Function** | Single logit output + stable sigmoid-BCE | **`nn.BCEWithLogitsLoss()`** | `radio-689-2` |
| **Q2** | **Total Trainable Parameters** | `896` (`conv`) + `20,072,512` (`hidden`) + `65` (`output`) | **`20073473`** | `radio-690-4` |
| **Q3** | **Median Baseline Training Accuracy** | `0.808125` (rounded to 2 decimal places, tolerance `±0.05`) | **`0.81`** | `answer_691` |
| **Q4** | **Baseline Training Loss Std (`ddof=0`)** | `0.13934083913625103` (rounded to 3 decimal places, tolerance `±0.02`) | **`0.139`** | `answer_692` |
| **Q5** | **Mean Augmented Evaluation Loss** | `0.5754194738556496` (rounded to 3 decimal places, tolerance `±0.02`) | **`0.575`** | `answer_693` |
| **Q6** | **Mean Augmented Evaluation Accuracy (Last 5 Epochs)** | `0.7233830845771144` (rounded to 2 decimal places, tolerance `±0.05`) | **`0.72`** | `answer_694` |

---

## 2. Files in This Directory

- [`homework.md`](./homework.md) & [`homework.yaml`](./homework.yaml) — Official 2026 Homework 8 instructions and schema
- [`dataset_manifest.json`](./dataset_manifest.json) — Dataset checksum and class split manifest
- [`requirements-cpu.txt`](./requirements-cpu.txt) — Pinned deterministic CPU wheel versions (`torch==2.9.0+cpu`, `torchvision==0.24.0+cpu`, `numpy==2.3.3`, `Pillow==11.3.0`)
- [`reference_train.py`](./reference_train.py) — Deterministic PyTorch training script producing `history_baseline.json` and `history_augmented.json`
- [`homework_8.py`](./homework_8.py) — Solution summary and verification script
- [`homework_8.ipynb`](./homework_8.ipynb) — Executed Jupyter Notebook with per-epoch baseline and augmented metrics
- [`history_baseline.json`](./history_baseline.json) — Recorded 10-epoch baseline training & held-out evaluation history
- [`history_augmented.json`](./history_augmented.json) — Recorded 10-epoch augmented continued-training & held-out evaluation history

---

## 3. Reproducing the Reference Run

```bash
# 1. Download and extract the dataset archive
wget https://github.com/SVizor42/ML_Zoomcamp/releases/download/straight-curly-data/data.zip
sha256sum data.zip  # 9e53453e3017502f22860cb08558f0cdb343e102fb61feab75960616185e7d67
unzip -q data.zip

# 2. Run the deterministic CPU reference pipeline with uv
uv run --python 3.11 \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  --with "torch==2.9.0+cpu" \
  --with "torchvision==0.24.0+cpu" \
  --with "numpy==2.3.3" \
  --with "Pillow==11.3.0" \
  python reference_train.py --data-dir data --output-dir .

# 3. Print formatted Q1-Q6 answers
python3 homework_8.py
```
