from __future__ import annotations

from dataclasses import asdict, dataclass
from io import StringIO
from typing import Dict, List

import pandas as pd

from src.data.schemas import DatasetSchema


@dataclass
class SplitSummary:
    name: str
    rows: int
    positive_count: int
    positive_rate: float


def _feature_type_summary(df: pd.DataFrame, target_col: str) -> Dict[str, List[str]]:
    features = [c for c in df.columns if c != target_col]
    numeric_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in features if c not in numeric_cols]
    return {"numeric": numeric_cols, "categorical": categorical_cols}


def _distribution_notes(df: pd.DataFrame) -> dict:
    notes: dict = {}
    if "type" in df.columns:
        notes["transaction_type_share"] = {
            str(k): float(v)
            for k, v in df["type"].value_counts(normalize=True, dropna=False).round(4).to_dict().items()
        }
    if "amount" in df.columns:
        amount_q = df["amount"].quantile([0.5, 0.9, 0.99, 0.999]).to_dict()
        notes["amount_quantiles"] = {str(k): float(v) for k, v in amount_q.items()}
    if {"oldbalanceOrg", "newbalanceOrig"}.issubset(df.columns):
        delta = df["oldbalanceOrg"] - df["newbalanceOrig"]
        notes["origin_balance_delta"] = {
            "mean": float(delta.mean()),
            "median": float(delta.median()),
        }
    if {"oldbalanceDest", "newbalanceDest"}.issubset(df.columns):
        delta = df["newbalanceDest"] - df["oldbalanceDest"]
        notes["destination_balance_delta"] = {
            "mean": float(delta.mean()),
            "median": float(delta.median()),
        }
    return notes


def build_audit_report(
    df: pd.DataFrame,
    schema: DatasetSchema,
    split_frames: Dict[str, pd.DataFrame],
) -> dict:
    target_col = schema.target_col
    class_counts = df[target_col].value_counts(dropna=False).to_dict()
    positive_rate = float(df[target_col].mean())

    missing_by_col = df.isna().sum().sort_values(ascending=False)
    missing_by_col = {k: int(v) for k, v in missing_by_col.items() if v > 0}

    split_summary = []
    for name, split_df in split_frames.items():
        pos_count = int(split_df[target_col].sum()) if len(split_df) else 0
        split_summary.append(
            asdict(
                SplitSummary(
                    name=name,
                    rows=len(split_df),
                    positive_count=pos_count,
                    positive_rate=float(split_df[target_col].mean()) if len(split_df) else 0.0,
                )
            )
        )

    describe_numeric = pd.read_json(StringIO(df.describe(include="number").to_json()), typ="frame").to_dict()
    duplicate_rows_post = int(df.duplicated().sum())

    report = {
        "dataset_metadata": {
            "dataset_name": schema.dataset_name,
            "dataset_source": schema.dataset_source,
            "provenance_url": schema.provenance_url,
            "is_synthetic": schema.is_synthetic,
            "raw_row_count": schema.raw_row_count,
            "normalized_row_count": int(len(df)),
            "sampling_applied": schema.sampling_applied,
            "sample_max_rows": schema.sample_max_rows,
            "sampled_row_count": schema.sampled_row_count,
        },
        "target_col": target_col,
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "class_balance": {str(k): int(v) for k, v in class_counts.items()},
        "positive_rate": positive_rate,
        "missing_values": missing_by_col,
        "duplicate_handling": {
            "policy": schema.duplicate_policy,
            "duplicates_removed": schema.duplicates_removed,
            "duplicates_remaining": duplicate_rows_post,
        },
        "leakage_risk_columns_dropped": schema.dropped_leakage_cols,
        "identifier_columns_dropped": schema.dropped_identifier_cols,
        "feature_types": _feature_type_summary(df, target_col),
        "split_summary": split_summary,
        "distribution_notes": _distribution_notes(df),
        "describe_numeric": describe_numeric,
    }
    return report
