from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.explainability.feature_importance import _humanize_feature_name


def linear_local_contributions(model: Any, row_df: pd.DataFrame, top_n: int = 5) -> list[dict]:
    if not hasattr(model, "named_steps"):
        return []

    preprocessor = model.named_steps.get("preprocessor")
    estimator = model.named_steps.get("model")

    if preprocessor is None or estimator is None or not hasattr(estimator, "coef_"):
        return []

    transformed = preprocessor.transform(row_df)
    coefs = estimator.coef_[0]

    dense = transformed.toarray()[0] if hasattr(transformed, "toarray") else np.asarray(transformed)[0]
    contributions = dense * coefs

    feature_names = preprocessor.get_feature_names_out()
    order = np.argsort(np.abs(contributions))[::-1][:top_n]

    result = []
    for idx in order:
        result.append(
            {
                "feature": _humanize_feature_name(str(feature_names[idx])),
                "contribution": float(contributions[idx]),
            }
        )
    return result
