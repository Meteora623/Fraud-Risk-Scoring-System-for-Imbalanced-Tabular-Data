from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _extract_model_and_preprocessor(model: Any):
    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        return model.named_steps.get("model"), model.named_steps["preprocessor"]
    return model, None


def _humanize_feature_name(feature_name: str) -> str:
    cleaned = feature_name.replace("num__", "").replace("cat__", "")
    if "_" in cleaned and cleaned.split("_", 1)[0] == "type":
        value = cleaned.split("_", 1)[1]
        return f"transaction_type={value}"
    return cleaned


def global_feature_importance(model: Any, top_n: int = 20) -> list[dict]:
    estimator, preprocessor = _extract_model_and_preprocessor(model)

    if preprocessor is None:
        return []

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return []

    importances = None
    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        if coef.ndim == 2:
            coef = coef[0]
        importances = np.abs(coef)

    if importances is None or len(importances) != len(feature_names):
        return []

    order = np.argsort(importances)[::-1][:top_n]
    output = []
    for idx in order:
        raw_name = str(feature_names[idx])
        output.append(
            {
                "feature": _humanize_feature_name(raw_name),
                "raw_feature": raw_name,
                "importance": float(importances[idx]),
            }
        )
    return output


def pretty_top_factors(importance_rows: list[dict], record: pd.DataFrame | None = None, top_n: int = 5) -> list[str]:
    top = importance_rows[:top_n]
    factors = []
    for row in top:
        name = row.get("feature") or row.get("raw_feature", "")
        if record is not None and name in record.columns:
            factors.append(f"{name}={record.iloc[0][name]}")
        elif "transaction_type=" in name:
            factors.append(name)
        else:
            base = name.split("_")[0] if "_" in name else name
            if record is not None and base in record.columns:
                factors.append(f"{base}={record.iloc[0][base]}")
            else:
                factors.append(name)
    return factors
