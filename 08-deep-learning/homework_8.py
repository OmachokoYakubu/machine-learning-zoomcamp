"""ML Zoomcamp 2026 — Module 08: Neural Networks and Deep Learning (Homework 8).

Runs the deterministic PyTorch CPU reference training pipeline (if history JSONs
are not yet generated) and prints the verified solutions for Questions 1–6.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

from reference_train import HairModel, run


def main() -> None:
    here = Path(__file__).resolve().parent
    baseline_path = here / "history_baseline.json"
    augmented_path = here / "history_augmented.json"

    if not (baseline_path.exists() and augmented_path.exists()):
        run(here / "data", here)

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    augmented = json.loads(augmented_path.read_text(encoding="utf-8"))

    model = HairModel()
    total_params = sum(p.numel() for p in model.parameters())

    median_train_acc = statistics.median(baseline["train_accuracy"])
    mean_loss = statistics.mean(baseline["train_loss"])
    std_train_loss_pop = math.sqrt(
        sum((x - mean_loss) ** 2 for x in baseline["train_loss"])
        / len(baseline["train_loss"])
    )
    mean_aug_eval_loss = statistics.mean(augmented["evaluation_loss"])
    mean_aug_last5_eval_acc = statistics.mean(augmented["evaluation_accuracy"][-5:])

    print("=" * 72)
    print("ML Zoomcamp 2026 — Homework 8 Verified Solutions")
    print("=" * 72)
    print("Q1. Loss function:                      nn.BCEWithLogitsLoss()")
    print(f"Q2. Total trainable parameters:         {total_params:,} (20073473)")
    print(
        f"Q3. Median baseline train accuracy:     {median_train_acc:.6f} -> {median_train_acc:.2f}"
    )
    print(
        f"Q4. Std (ddof=0) baseline train loss:   {std_train_loss_pop:.6f} -> {std_train_loss_pop:.3f}"
    )
    print(
        f"Q5. Mean augmented evaluation loss:     {mean_aug_eval_loss:.6f} -> {mean_aug_eval_loss:.3f}"
    )
    print(
        f"Q6. Mean augmented eval acc (epochs 6-10): {mean_aug_last5_eval_acc:.6f} -> {mean_aug_last5_eval_acc:.2f}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
