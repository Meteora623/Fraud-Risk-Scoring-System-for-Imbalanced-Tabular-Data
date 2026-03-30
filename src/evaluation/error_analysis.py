from __future__ import annotations

from typing import Dict

import pandas as pd


def analyze_errors(df: pd.DataFrame, target_col: str, score_col: str, threshold: float) -> Dict[str, dict]:
    y_true = df[target_col].astype(int)
    y_pred = (df[score_col] >= threshold).astype(int)

    fp_df = df[(y_true == 0) & (y_pred == 1)]
    fn_df = df[(y_true == 1) & (y_pred == 0)]

    feature_cols = [c for c in df.columns if c not in {target_col, score_col, "split"}]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    def numeric_shift(error_df: pd.DataFrame) -> dict:
        shifts = {}
        for col in numeric_cols[:20]:
            base_mean = float(df[col].mean())
            err_mean = float(error_df[col].mean()) if len(error_df) else 0.0
            shifts[col] = {"error_mean": err_mean, "overall_mean": base_mean, "delta": err_mean - base_mean}
        return shifts

    def top_categorical(error_df: pd.DataFrame) -> dict:
        patterns = {}
        for col in categorical_cols[:10]:
            vals = error_df[col].value_counts(normalize=True).head(5)
            patterns[col] = {str(k): float(v) for k, v in vals.items()}
        return patterns

    return {
        "counts": {
            "false_positives": int(len(fp_df)),
            "false_negatives": int(len(fn_df)),
        },
        "false_positive_patterns": {
            "numeric_shift": numeric_shift(fp_df),
            "categorical_top": top_categorical(fp_df),
        },
        "false_negative_patterns": {
            "numeric_shift": numeric_shift(fn_df),
            "categorical_top": top_categorical(fn_df),
        },
    }
