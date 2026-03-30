from __future__ import annotations

import pandas as pd

from src.data.loaders import normalize_schema
from src.utils.config import ProjectConfig


def _cfg_dict() -> dict:
    return {
        "project_name": "loader_test",
        "random_seed": 42,
        "paths": {
            "raw_data_path": "data/raw/paysim.csv",
            "processed_data_path": "data/processed/x.parquet",
            "split_data_dir": "data/processed/splits",
            "audit_report_path": "data/artifacts/audit_report.json",
            "model_dir": "data/models",
            "artifact_dir": "data/artifacts",
            "comparison_metrics_path": "data/artifacts/model_metrics.json",
            "threshold_summary_path": "data/artifacts/threshold_summary.json",
        },
        "dataset": {
            "name": "paysim",
            "source": "kaggle",
            "provenance_url": "https://www.kaggle.com/datasets/ealaxi/paysim1",
            "kaggle_dataset": "ealaxi/paysim1",
            "is_synthetic": True,
            "target_aliases": ["isFraud", "fraud"],
            "drop_identifier_columns": ["nameOrig", "nameDest"],
            "leakage_risk_columns": ["isFlaggedFraud"],
            "duplicate_policy": "drop_exact",
        },
        "schema": {
            "target_col": "isFraud",
            "positive_class_value": 1,
            "id_cols": ["nameOrig", "nameDest"],
            "time_col": "step",
        },
        "split": {"strategy": "random_stratified", "train_size": 0.7, "val_size": 0.15, "test_size": 0.15},
        "preprocessing": {
            "numeric_imputer_strategy": "median",
            "categorical_imputer_strategy": "most_frequent",
            "scale_numeric_for_linear": True,
            "one_hot_min_frequency": 0.01,
        },
        "models": {
            "enabled": ["logistic_regression"],
            "primary_model": "auto",
            "logistic_regression": {},
        },
        "calibration": {"enabled": False, "method": "sigmoid"},
        "thresholding": {
            "default_threshold": 0.5,
            "precision_floor": 0.9,
            "recall_floor": 0.8,
            "false_positive_cost": 1,
            "false_negative_cost": 25,
        },
        "api": {"host": "0.0.0.0", "port": 8000},
        "logging": {"level": "INFO"},
    }


def test_normalize_paysim_schema_drops_leakage_and_ids() -> None:
    raw = pd.DataFrame(
        {
            "step": [1, 1],
            "type": ["TRANSFER", "TRANSFER"],
            "amount": [1000.0, 1000.0],
            "oldbalanceOrg": [5000.0, 5000.0],
            "newbalanceOrig": [4000.0, 4000.0],
            "oldbalanceDest": [0.0, 0.0],
            "newbalanceDest": [1000.0, 1000.0],
            "nameOrig": ["C123", "C123"],
            "nameDest": ["M987", "M987"],
            "isFlaggedFraud": [0, 0],
            "isFraud": [1, 1],
        }
    )

    cfg = ProjectConfig.model_validate(_cfg_dict())
    normalized, schema = normalize_schema(raw, cfg)

    assert "nameOrig" not in normalized.columns
    assert "nameDest" not in normalized.columns
    assert "isFlaggedFraud" not in normalized.columns
    assert len(normalized) == 1
    assert schema.duplicates_removed == 1
    assert schema.dataset_name == "paysim"
