from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from src.api.routes import router
from src.models.infer import RiskScorer
from src.utils.config import load_config
from src.utils.io import read_json


def create_app(config_path: str = "configs/config.yaml") -> FastAPI:
    cfg = load_config(config_path)

    app = FastAPI(title="Fraud Risk Scoring API", version="1.0.0")

    model_bundle_path = Path(cfg.paths.model_dir) / "serving_bundle.joblib"
    threshold_path = cfg.paths.threshold_summary_path
    metrics_path = cfg.paths.comparison_metrics_path

    if model_bundle_path.exists() and Path(threshold_path).exists():
        app.state.scorer = RiskScorer(model_bundle_path, threshold_path)
    else:
        app.state.scorer = None

    app.state.metrics_summary = read_json(metrics_path) if Path(metrics_path).exists() else {}

    app.include_router(router)
    return app


app = create_app()
