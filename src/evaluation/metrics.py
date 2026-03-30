from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "log_loss": float(log_loss(y_true, np.clip(y_score, 1e-6, 1 - 1e-6))),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
    return metrics


def recall_at_precision(y_true: np.ndarray, y_score: np.ndarray, precision_floor: float) -> float:
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    mask = precision >= precision_floor
    if not np.any(mask):
        return 0.0
    return float(np.max(recall[mask]))


def precision_at_recall(y_true: np.ndarray, y_score: np.ndarray, recall_floor: float) -> float:
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    mask = recall >= recall_floor
    if not np.any(mask):
        return 0.0
    return float(np.max(precision[mask]))


def attach_service_metrics(metrics: Dict[str, Any], y_true: np.ndarray, y_score: np.ndarray, precision_floor: float, recall_floor: float) -> Dict[str, Any]:
    metrics = dict(metrics)
    metrics["recall_at_precision_floor"] = recall_at_precision(y_true, y_score, precision_floor)
    metrics["precision_at_recall_floor"] = precision_at_recall(y_true, y_score, recall_floor)
    return metrics
