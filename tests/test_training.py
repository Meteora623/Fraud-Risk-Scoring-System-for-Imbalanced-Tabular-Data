from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.data.schemas import DatasetSchema
from src.models.train import save_training_artifacts, train_all_models
from src.utils.config import ProjectConfig


def _build_test_config(tmp_path: Path) -> ProjectConfig:
    payload = {
        "project_name": "test_project",
        "random_seed": 42,
        "paths": {
            "raw_data_path": str(tmp_path / "raw.csv"),
            "processed_data_path": str(tmp_path / "processed.parquet"),
            "split_data_dir": str(tmp_path / "splits"),
            "audit_report_path": str(tmp_path / "audit.json"),
            "model_dir": str(tmp_path / "models"),
            "artifact_dir": str(tmp_path / "artifacts"),
            "comparison_metrics_path": str(tmp_path / "artifacts" / "model_metrics.json"),
            "threshold_summary_path": str(tmp_path / "artifacts" / "thresholds.json"),
        },
        "schema": {
            "target_col": "Class",
            "positive_class_value": 1,
            "id_cols": [],
            "time_col": None,
        },
        "split": {
            "strategy": "random_stratified",
            "train_size": 0.70,
            "val_size": 0.15,
            "test_size": 0.15,
        },
        "preprocessing": {
            "numeric_imputer_strategy": "median",
            "categorical_imputer_strategy": "most_frequent",
            "scale_numeric_for_linear": True,
            "one_hot_min_frequency": 0.01,
        },
        "models": {
            "enabled": ["logistic_regression", "random_forest"],
            "primary_model": "random_forest",
            "logistic_regression": {"C": 1.0, "max_iter": 200, "class_weight": "balanced"},
            "random_forest": {
                "n_estimators": 50,
                "max_depth": 6,
                "min_samples_leaf": 1,
                "class_weight": "balanced_subsample",
                "n_jobs": 1,
            },
            "xgboost": {"n_estimators": 20},
        },
        "calibration": {"enabled": True, "method": "sigmoid"},
        "thresholding": {"default_threshold": 0.5, "precision_floor": 0.8, "recall_floor": 0.7},
        "api": {"host": "0.0.0.0", "port": 8000},
        "logging": {"level": "INFO"},
    }
    return ProjectConfig.model_validate(payload)


def test_training_pipeline_produces_artifacts(tmp_path: Path) -> None:
    x, y = make_classification(
        n_samples=1200,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        weights=[0.97, 0.03],
        random_state=42,
    )
    cols = [f"f{i}" for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=cols)
    df["Class"] = y.astype(int)

    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["Class"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["Class"], random_state=42)

    cfg = _build_test_config(tmp_path)
    schema = DatasetSchema(target_col="Class", id_cols=[], time_col=None)

    result = train_all_models(train_df, val_df, test_df, schema, cfg)
    summary = save_training_artifacts(train_df, val_df, test_df, schema, cfg, result)

    assert "primary_model" in summary
    assert Path(cfg.paths.comparison_metrics_path).exists()
    assert Path(cfg.paths.threshold_summary_path).exists()
    assert (Path(cfg.paths.model_dir) / "serving_bundle.joblib").exists()

