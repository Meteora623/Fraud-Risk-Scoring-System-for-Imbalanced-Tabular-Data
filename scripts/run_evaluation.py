from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.error_analysis import analyze_errors
from src.evaluation.plots import (
    plot_calibration,
    plot_confusion,
    plot_pr_curve,
    plot_roc_curve,
    plot_score_distribution,
    plot_threshold_tradeoff,
)
from src.evaluation.thresholding import threshold_table
from src.utils.config import load_config
from src.utils.io import load_joblib, read_json, write_json
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    cfg = load_config()
    setup_logging(cfg.logging.level)

    scored_path = Path(cfg.paths.artifact_dir) / "scored_holdout.csv"
    if not scored_path.exists():
        raise FileNotFoundError("Missing scored holdout file. Run scripts/train_models.py first.")

    scored = pd.read_csv(scored_path)
    target_col = cfg.schema.target_col
    score_col = "score"

    bundle = load_joblib(Path(cfg.paths.model_dir) / "serving_bundle.joblib")
    primary_model = bundle["model_name"]

    threshold_summary = read_json(cfg.paths.threshold_summary_path)
    threshold = float(threshold_summary["balanced_f1"]["threshold"])

    y_true = scored[target_col].astype(int).to_numpy()
    y_score = scored[score_col].to_numpy()
    y_pred = (y_score >= threshold).astype(int)

    eval_dir = Path(cfg.paths.artifact_dir) / "plots"
    plot_pr_curve(y_true, y_score, eval_dir, "pr_curve.png")
    plot_roc_curve(y_true, y_score, eval_dir, "roc_curve.png")
    plot_confusion(y_true, y_pred, eval_dir, "confusion_matrix.png")
    plot_calibration(y_true, y_score, eval_dir, "calibration_curve.png")
    plot_score_distribution(y_true, y_score, eval_dir, "score_distribution.png")

    table = threshold_table(
        y_true,
        y_score,
        false_positive_cost=cfg.thresholding.false_positive_cost,
        false_negative_cost=cfg.thresholding.false_negative_cost,
    )
    table.to_csv(Path(cfg.paths.artifact_dir) / "threshold_table.csv", index=False)
    plot_threshold_tradeoff(table, eval_dir, "threshold_tradeoff.png")

    error_report = analyze_errors(scored, target_col=target_col, score_col=score_col, threshold=threshold)
    error_report["primary_model"] = primary_model
    write_json(Path(cfg.paths.artifact_dir) / "error_analysis.json", error_report)

    logger.info("Evaluation outputs generated in: %s", cfg.paths.artifact_dir)


if __name__ == "__main__":
    main()
