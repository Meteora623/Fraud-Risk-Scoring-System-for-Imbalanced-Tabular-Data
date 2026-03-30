from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.api.app import create_app
from src.utils.io import dump_joblib, write_json


def _prepare_api_artifacts(tmp_path: Path) -> Path:
    x_df = pd.DataFrame(
        {
            "step": [1, 2, 3, 4, 5],
            "type": ["TRANSFER", "PAYMENT", "CASH_OUT", "DEBIT", "TRANSFER"],
            "amount": [1000, 150, 8000, 200, 12000],
            "oldbalanceOrg": [4000, 500, 9000, 800, 16000],
            "newbalanceOrig": [3000, 350, 1000, 600, 2000],
            "oldbalanceDest": [100, 2000, 0, 800, 200],
            "newbalanceDest": [1100, 2150, 8000, 1000, 12200],
        }
    )
    y = np.array([0, 0, 1, 0, 1])

    num_cols = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    cat_cols = ["type"]

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=200)),
        ]
    )
    model.fit(x_df, y)

    model_dir = tmp_path / "models"
    artifact_dir = tmp_path / "artifacts"
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model_name": "logistic_regression",
        "model": model,
        "schema": {"target_col": "isFraud", "id_cols": [], "time_col": "step"},
        "feature_importance": [],
    }
    dump_joblib(model_dir / "serving_bundle.joblib", bundle)

    write_json(
        artifact_dir / "thresholds.json",
        {
            "balanced_f1": {"threshold": 0.5},
            "high_precision": {"threshold": 0.8},
            "high_recall": {"threshold": 0.3},
            "cost_sensitive": {"threshold": 0.4},
        },
    )
    write_json(artifact_dir / "metrics.json", {"logistic_regression": {"test": {"pr_auc": 0.8}}})

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
project_name: api_test
random_seed: 42
paths:
  raw_data_path: {tmp_path / 'raw.csv'}
  processed_data_path: {tmp_path / 'processed.parquet'}
  split_data_dir: {tmp_path / 'splits'}
  audit_report_path: {tmp_path / 'audit.json'}
  model_dir: {model_dir}
  artifact_dir: {artifact_dir}
  comparison_metrics_path: {artifact_dir / 'metrics.json'}
  threshold_summary_path: {artifact_dir / 'thresholds.json'}
dataset:
  name: paysim
  source: kaggle
  provenance_url: null
  kaggle_dataset: ealaxi/paysim1
  is_synthetic: true
  target_aliases: [isFraud]
  drop_identifier_columns: [nameOrig, nameDest]
  leakage_risk_columns: [isFlaggedFraud]
  duplicate_policy: drop_exact
schema:
  target_col: isFraud
  positive_class_value: 1
  id_cols: [nameOrig, nameDest]
  time_col: step
split:
  strategy: random_stratified
  train_size: 0.7
  val_size: 0.15
  test_size: 0.15
preprocessing:
  numeric_imputer_strategy: median
  categorical_imputer_strategy: most_frequent
  scale_numeric_for_linear: true
  one_hot_min_frequency: 0.01
models:
  enabled: [logistic_regression]
  primary_model: logistic_regression
  logistic_regression: {{}}
calibration:
  enabled: false
  method: sigmoid
thresholding:
  default_threshold: 0.5
  precision_floor: 0.8
  recall_floor: 0.8
  false_positive_cost: 1.0
  false_negative_cost: 10.0
api:
  host: 0.0.0.0
  port: 8000
logging:
  level: INFO
""",
        encoding="utf-8",
    )
    return cfg_path


def test_api_health_endpoint(tmp_path: Path) -> None:
    cfg_path = _prepare_api_artifacts(tmp_path)
    app = create_app(str(cfg_path))
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True


def test_api_score_endpoint(tmp_path: Path) -> None:
    cfg_path = _prepare_api_artifacts(tmp_path)
    app = create_app(str(cfg_path))
    client = TestClient(app)

    response = client.post(
        "/score_transaction",
        json={
            "transaction": {
                "step": 5,
                "type": "TRANSFER",
                "amount": 7500.0,
                "oldbalanceOrg": 9500.0,
                "newbalanceOrig": 2000.0,
                "oldbalanceDest": 100.0,
                "newbalanceDest": 7600.0,
            },
            "threshold_mode": "cost_sensitive",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "risk_score" in body
    assert body["decision_regime"] == "cost_sensitive"
