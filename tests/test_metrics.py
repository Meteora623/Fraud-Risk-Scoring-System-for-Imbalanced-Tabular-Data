from __future__ import annotations

import numpy as np

from src.evaluation.metrics import compute_binary_metrics
from src.evaluation.thresholding import pick_thresholds, threshold_table


def test_metrics_include_imbalanced_fields() -> None:
    y_true = np.array([0, 0, 0, 1, 1])
    y_score = np.array([0.01, 0.2, 0.4, 0.6, 0.95])

    metrics = compute_binary_metrics(y_true, y_score, threshold=0.5)

    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "false_positives" in metrics
    assert metrics["recall"] > 0.0


def test_threshold_picker_returns_regimes_including_cost_sensitive() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_score = np.array([0.02, 0.1, 0.2, 0.4, 0.55, 0.7, 0.8, 0.9])
    table = threshold_table(y_true, y_score, step=0.1, false_positive_cost=1.0, false_negative_cost=10.0)

    selected = pick_thresholds(table, precision_floor=0.8, recall_floor=0.8)

    assert "balanced_f1" in selected
    assert "high_precision" in selected
    assert "high_recall" in selected
    assert "cost_sensitive" in selected
    assert "expected_cost" in table.columns
