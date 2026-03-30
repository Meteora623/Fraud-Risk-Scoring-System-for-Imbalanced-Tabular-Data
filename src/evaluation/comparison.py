from __future__ import annotations

from typing import Dict


def rank_models(metrics_by_model: Dict[str, dict], key: str = "pr_auc") -> list[dict]:
    ranked = []
    for model_name, metrics in metrics_by_model.items():
        ranked.append({"model": model_name, **metrics})
    return sorted(ranked, key=lambda x: x.get(key, 0.0), reverse=True)
