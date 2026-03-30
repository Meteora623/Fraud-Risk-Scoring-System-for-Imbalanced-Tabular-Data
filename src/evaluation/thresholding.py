from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def threshold_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    step: float = 0.01,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 1.0,
) -> pd.DataFrame:
    thresholds = np.arange(step, 1.0, step)
    rows = []
    n = len(y_true)

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        expected_cost = fp * false_positive_cost + fn * false_negative_cost

        rows.append(
            {
                "threshold": round(float(threshold), 4),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "expected_cost": float(expected_cost),
                "cost_per_txn": float(expected_cost / n) if n else 0.0,
            }
        )

    return pd.DataFrame(rows)


def pick_thresholds(table: pd.DataFrame, precision_floor: float, recall_floor: float) -> Dict[str, float]:
    best_f1 = table.loc[table["f1"].idxmax()]
    best_cost = table.loc[table["expected_cost"].idxmin()]

    precision_candidates = table[table["precision"] >= precision_floor]
    if len(precision_candidates):
        high_precision = precision_candidates.sort_values(["recall", "threshold"], ascending=[False, False]).iloc[0]
    else:
        high_precision = table.sort_values(["precision", "recall"], ascending=[False, False]).iloc[0]

    recall_candidates = table[table["recall"] >= recall_floor]
    if len(recall_candidates):
        high_recall = recall_candidates.sort_values(["precision", "threshold"], ascending=[False, True]).iloc[0]
    else:
        high_recall = table.sort_values(["recall", "precision"], ascending=[False, False]).iloc[0]

    return {
        "balanced_f1": float(best_f1["threshold"]),
        "high_precision": float(high_precision["threshold"]),
        "high_recall": float(high_recall["threshold"]),
        "cost_sensitive": float(best_cost["threshold"]),
    }


def regime_descriptions() -> Dict[str, str]:
    return {
        "balanced_f1": "Balances precision and recall for general triage.",
        "high_precision": "Minimizes false positives for manual-review efficiency.",
        "high_recall": "Catches more fraud at the expense of higher alert volume.",
        "cost_sensitive": "Minimizes weighted decision cost using configured FP/FN costs.",
    }


def summarize_thresholds(table: pd.DataFrame, selected: Dict[str, float]) -> dict:
    summary: Dict[str, dict] = {}
    descriptions = regime_descriptions()
    for name, threshold in selected.items():
        row = table.iloc[(table["threshold"] - threshold).abs().argmin()].to_dict()
        payload = {k: float(v) if isinstance(v, (float, int)) else v for k, v in row.items()}
        payload["description"] = descriptions.get(name, "")
        summary[name] = payload
    return summary


def risk_band(score: float, thresholds: Dict[str, float]) -> str:
    hp = thresholds.get("high_precision", 0.8)
    bf1 = thresholds.get("balanced_f1", 0.5)
    if score >= hp:
        return "high"
    if score >= bf1:
        return "medium"
    return "low"
