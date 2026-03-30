from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.thresholding import regime_descriptions, risk_band
from src.explainability.feature_importance import pretty_top_factors
from src.explainability.local_explanations import linear_local_contributions
from src.utils.io import load_joblib, read_json


@dataclass
class InferenceBundle:
    model_name: str
    model: Any
    thresholds: dict
    schema: dict
    feature_importance: list[dict]


def _regime_thresholds(threshold_payload: dict) -> dict[str, float]:
    return {
        k: float(v["threshold"])
        for k, v in threshold_payload.items()
        if isinstance(v, dict) and "threshold" in v
    }


def _review_posture(regime: str) -> str:
    mapping = {
        "balanced_f1": "balanced triage",
        "high_precision": "strict alerting",
        "high_recall": "coverage-first",
        "cost_sensitive": "cost-optimized",
    }
    return mapping.get(regime, "balanced triage")


def _recommended_action(risk_band_value: str) -> str:
    if risk_band_value == "high":
        return "Escalate for immediate fraud review and consider transaction hold."
    if risk_band_value == "medium":
        return "Queue for analyst review with standard SLA."
    return "Auto-approve with passive monitoring."


class RiskScorer:
    def __init__(self, model_bundle_path: str | Path, threshold_path: str | Path):
        payload = load_joblib(model_bundle_path)
        self.bundle = InferenceBundle(
            model_name=payload["model_name"],
            model=payload["model"],
            thresholds=read_json(threshold_path),
            schema=payload["schema"],
            feature_importance=payload.get("feature_importance", []),
        )

    def score_record(self, record: dict[str, Any], threshold_key: str = "balanced_f1") -> dict[str, Any]:
        row_df = pd.DataFrame([record])
        if "type" in row_df.columns:
            row_df["type"] = row_df["type"].astype(str).str.upper()

        model = self.bundle.model

        threshold_map = _regime_thresholds(self.bundle.thresholds)
        if threshold_key not in threshold_map:
            threshold_key = "balanced_f1"
        if threshold_key not in threshold_map:
            threshold_key = next(iter(threshold_map))

        score = float(model.predict_proba(row_df)[:, 1][0])
        threshold = float(threshold_map[threshold_key])
        pred_class = int(score >= threshold)
        band = risk_band(score, threshold_map)

        local_linear = linear_local_contributions(model, row_df, top_n=5)
        if local_linear:
            explanation = [f"{r['feature']}: {r['contribution']:.4f}" for r in local_linear]
        else:
            explanation = pretty_top_factors(self.bundle.feature_importance, row_df, top_n=5)

        regime_payload = self.bundle.thresholds.get(threshold_key, {})
        regime_metrics = {
            k: float(v)
            for k, v in regime_payload.items()
            if isinstance(v, (float, int)) and k in {"precision", "recall", "f1", "expected_cost", "cost_per_txn", "threshold"}
        }

        meta = self.bundle.thresholds.get("_meta", {})
        cost_context = {
            "false_positive_cost": float(meta.get("false_positive_cost", 0.0)),
            "false_negative_cost": float(meta.get("false_negative_cost", 0.0)),
        }

        return {
            "model": self.bundle.model_name,
            "risk_score": score,
            "predicted_class": pred_class,
            "threshold_used": threshold,
            "risk_band": band,
            "decision_regime": threshold_key,
            "decision_note": regime_descriptions().get(threshold_key, ""),
            "review_posture": _review_posture(threshold_key),
            "recommended_action": _recommended_action(band),
            "regime_metrics": regime_metrics,
            "cost_context": cost_context,
            "top_factors": explanation,
        }
