from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.feature_pipeline import build_preprocessor
from src.data.preprocess import infer_feature_types
from src.data.schemas import DatasetSchema
from src.evaluation.metrics import attach_service_metrics, compute_binary_metrics
from src.evaluation.thresholding import pick_thresholds, summarize_thresholds, threshold_table
from src.explainability.feature_importance import global_feature_importance
from src.models.baselines import build_logistic_regression, build_random_forest
from src.models.boosted import build_boosted_model
from src.models.calibrate import fit_calibrator
from src.utils.config import ProjectConfig
from src.utils.io import dump_joblib, ensure_parent_dir, write_json

logger = logging.getLogger(__name__)


def _build_model(name: str, cfg: ProjectConfig, scale_pos_weight: float):
    random_seed = cfg.random_seed
    if name == "logistic_regression":
        model = build_logistic_regression(cfg.models.get("logistic_regression", {}))
        return model, True
    if name == "random_forest":
        model_cfg = dict(cfg.models.get("random_forest", {}))
        model_cfg["random_state"] = random_seed
        model = build_random_forest(model_cfg)
        return model, False
    if name == "xgboost":
        model = build_boosted_model(cfg.models.get("xgboost", {}), scale_pos_weight, random_seed)
        return model, False
    raise ValueError(f"Unsupported model: {name}")


def _split_xy(df: pd.DataFrame, schema: DatasetSchema):
    y = df[schema.target_col].astype(int).to_numpy()
    x = df.drop(columns=[schema.target_col])
    return x, y


def _score_model(model, x, y, threshold, cfg: ProjectConfig):
    scores = model.predict_proba(x)[:, 1]
    metrics = compute_binary_metrics(y, scores, threshold=threshold)
    return attach_service_metrics(
        metrics,
        y,
        scores,
        precision_floor=cfg.thresholding.precision_floor,
        recall_floor=cfg.thresholding.recall_floor,
    ), scores


def _extract_estimator_class(model: Any) -> str:
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return type(model.named_steps["model"]).__name__
    if hasattr(model, "estimator") and hasattr(model.estimator, "named_steps"):
        inner = model.estimator.named_steps.get("model")
        if inner is not None:
            return type(inner).__name__
    return type(model).__name__


def _model_backend_name(model_name: str, estimator_class: str) -> str:
    if model_name == "xgboost":
        if estimator_class == "XGBClassifier":
            return "xgboost"
        if estimator_class == "HistGradientBoostingClassifier":
            return "hist_gradient_boosting_fallback"
    return model_name


def train_all_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema: DatasetSchema,
    cfg: ProjectConfig,
) -> Dict[str, Any]:
    x_train, y_train = _split_xy(train_df, schema)
    x_val, y_val = _split_xy(val_df, schema)
    x_test, y_test = _split_xy(test_df, schema)

    feature_set = infer_feature_types(train_df, schema)
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = (negatives / max(positives, 1)) if positives else 1.0

    enabled = cfg.models.get("enabled", ["logistic_regression", "random_forest", "xgboost"])
    default_threshold = cfg.thresholding.default_threshold

    model_results: Dict[str, dict] = {}
    metrics_summary: Dict[str, dict] = {}

    for model_name in enabled:
        base_model, for_linear_model = _build_model(model_name, cfg, scale_pos_weight)
        preprocessor = build_preprocessor(feature_set, cfg.preprocessing, for_linear_model=for_linear_model)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", base_model),
            ]
        )

        estimator_class = _extract_estimator_class(pipeline)
        model_backend = _model_backend_name(model_name, estimator_class)

        logger.info("Training model: %s (estimator=%s)", model_name, estimator_class)
        pipeline.fit(x_train, y_train)

        val_metrics_raw, val_scores_raw = _score_model(pipeline, x_val, y_val, default_threshold, cfg)
        test_metrics_raw, test_scores_raw = _score_model(pipeline, x_test, y_test, default_threshold, cfg)

        selected_model = pipeline
        calibration_state = "not_applied"

        if cfg.calibration.enabled:
            calibrated = fit_calibrator(pipeline, x_val, y_val, method=cfg.calibration.method)
            c_val_metrics, c_val_scores = _score_model(calibrated, x_val, y_val, default_threshold, cfg)
            c_test_metrics, c_test_scores = _score_model(calibrated, x_test, y_test, default_threshold, cfg)

            if c_val_metrics["brier"] <= val_metrics_raw["brier"]:
                selected_model = calibrated
                calibration_state = "calibrated_selected"
                val_metrics = c_val_metrics
                val_scores = c_val_scores
                test_metrics = c_test_metrics
                test_scores = c_test_scores
            else:
                calibration_state = "calibration_rejected"
                val_metrics = val_metrics_raw
                val_scores = val_scores_raw
                test_metrics = test_metrics_raw
                test_scores = test_scores_raw
        else:
            val_metrics = val_metrics_raw
            val_scores = val_scores_raw
            test_metrics = test_metrics_raw
            test_scores = test_scores_raw

        selected_estimator_class = _extract_estimator_class(selected_model)

        model_results[model_name] = {
            "fitted_model": selected_model,
            "val_scores": val_scores,
            "test_scores": test_scores,
            "feature_importance": global_feature_importance(pipeline, top_n=30),
            "calibration_state": calibration_state,
            "estimator_class": estimator_class,
            "selected_estimator_class": selected_estimator_class,
            "model_backend": model_backend,
        }

        metrics_summary[model_name] = {
            "val": val_metrics,
            "test": test_metrics,
            "calibration_state": calibration_state,
            "estimator_class": estimator_class,
            "selected_estimator_class": selected_estimator_class,
            "model_backend": model_backend,
            "selected_for_serving": False,
        }

        dump_joblib(Path(cfg.paths.model_dir) / f"{model_name}.joblib", selected_model)

    return {
        "models": model_results,
        "metrics_summary": metrics_summary,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }


def _select_primary_model(metrics_summary: Dict[str, dict], configured_primary: str) -> tuple[str, str]:
    if configured_primary and configured_primary != "auto" and configured_primary in metrics_summary:
        return configured_primary, "configured_primary_model"

    selected = max(
        metrics_summary.keys(),
        key=lambda m: metrics_summary[m]["val"].get("pr_auc", 0.0),
    )
    return selected, "auto_best_val_pr_auc"


def _build_final_model_summary(
    cfg: ProjectConfig,
    schema: DatasetSchema,
    metrics_summary: Dict[str, dict],
    threshold_summary: Dict[str, dict],
    primary_model_name: str,
    selection_reason: str,
) -> dict:
    ranking = sorted(
        [
            {
                "model": model_name,
                "test_pr_auc": payload.get("test", {}).get("pr_auc", 0.0),
                "test_roc_auc": payload.get("test", {}).get("roc_auc", 0.0),
                "test_f1": payload.get("test", {}).get("f1", 0.0),
                "calibration_state": payload.get("calibration_state"),
                "model_backend": payload.get("model_backend"),
                "selected_for_serving": payload.get("selected_for_serving", False),
            }
            for model_name, payload in metrics_summary.items()
        ],
        key=lambda row: row["test_pr_auc"],
        reverse=True,
    )

    boosted_payload = metrics_summary.get("xgboost")
    boosted_note = "Boosted model not enabled in this run."
    if boosted_payload is not None:
        boosted_backend = boosted_payload.get("model_backend")
        boosted_selected = boosted_payload.get("selected_for_serving", False)
        boosted_pr_auc = boosted_payload.get("test", {}).get("pr_auc", 0.0)
        primary_pr_auc = metrics_summary[primary_model_name]["test"].get("pr_auc", 0.0)

        if boosted_selected:
            boosted_note = "Boosted model was selected for serving in this run."
        else:
            reasons = []
            if boosted_backend == "hist_gradient_boosting_fallback":
                reasons.append("requested xgboost backend fell back to HistGradientBoosting in this environment")
            if boosted_pr_auc < primary_pr_auc:
                reasons.append("lower test PR-AUC than selected model")
            reasons.append("single-run defaults were used without extensive hyperparameter tuning")
            boosted_note = "Boosted model not selected: " + "; ".join(reasons) + "."

    regime_rows = {k: v for k, v in threshold_summary.items() if isinstance(v, dict) and "threshold" in v}

    return {
        "project": {
            "name": cfg.project_name,
            "dataset_name": schema.dataset_name,
            "dataset_source": schema.dataset_source,
            "provenance_url": schema.provenance_url,
            "is_synthetic": schema.is_synthetic,
        },
        "primary_model": primary_model_name,
        "primary_model_selection_reason": selection_reason,
        "model_ranking_by_test_pr_auc": ranking,
        "boosted_model_observation": {
            "configured_entry": "xgboost",
            "actual_backend": boosted_payload.get("model_backend") if boosted_payload else None,
            "selected_for_serving": boosted_payload.get("selected_for_serving") if boosted_payload else False,
            "note": boosted_note,
        },
        "calibration_policy": {
            "enabled": cfg.calibration.enabled,
            "method": cfg.calibration.method,
            "selection_rule": "use calibrated model only if validation brier score improves",
        },
        "threshold_policy": {
            "default_regime": "balanced_f1",
            "precision_floor": cfg.thresholding.precision_floor,
            "recall_floor": cfg.thresholding.recall_floor,
            "false_positive_cost": cfg.thresholding.false_positive_cost,
            "false_negative_cost": cfg.thresholding.false_negative_cost,
            "regimes": regime_rows,
        },
        "interview_safety_notes": [
            "PaySim is synthetic data; discuss this explicitly in interviews.",
            "Model selection is based on holdout metrics from this snapshot, not guaranteed universal superiority.",
            "Cost-sensitive thresholding reflects configured FP/FN costs and should be aligned with business context.",
        ],
    }


def save_training_artifacts(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema: DatasetSchema,
    cfg: ProjectConfig,
    train_output: Dict[str, Any],
) -> Dict[str, Any]:
    metrics_summary = train_output["metrics_summary"]

    configured_primary = str(cfg.models.get("primary_model", "auto"))
    primary_model_name, selection_reason = _select_primary_model(metrics_summary, configured_primary)
    primary = train_output["models"][primary_model_name]

    for name in metrics_summary:
        metrics_summary[name]["selected_for_serving"] = name == primary_model_name

    table = threshold_table(
        train_output["y_val"],
        primary["val_scores"],
        step=0.01,
        false_positive_cost=cfg.thresholding.false_positive_cost,
        false_negative_cost=cfg.thresholding.false_negative_cost,
    )
    selected = pick_thresholds(
        table,
        precision_floor=cfg.thresholding.precision_floor,
        recall_floor=cfg.thresholding.recall_floor,
    )
    threshold_summary = summarize_thresholds(table, selected)
    threshold_summary["_meta"] = {
        "false_positive_cost": cfg.thresholding.false_positive_cost,
        "false_negative_cost": cfg.thresholding.false_negative_cost,
        "primary_model": primary_model_name,
        "primary_model_selection_reason": selection_reason,
        "default_serving_regime": "balanced_f1",
    }

    ensure_parent_dir(cfg.paths.comparison_metrics_path)
    write_json(cfg.paths.comparison_metrics_path, metrics_summary)
    write_json(cfg.paths.threshold_summary_path, threshold_summary)

    table.to_csv(Path(cfg.paths.artifact_dir) / "threshold_table.csv", index=False)

    val_out = val_df.copy()
    val_out["score"] = primary["val_scores"]
    val_out["split"] = "val"

    test_out = test_df.copy()
    test_out["score"] = primary["test_scores"]
    test_out["split"] = "test"

    pd.concat([val_out, test_out], axis=0).to_csv(
        Path(cfg.paths.artifact_dir) / "scored_holdout.csv", index=False
    )

    serving_bundle = {
        "model_name": primary_model_name,
        "model": primary["fitted_model"],
        "schema": {
            "target_col": schema.target_col,
            "id_cols": schema.id_cols,
            "time_col": schema.time_col,
            "dataset_name": schema.dataset_name,
            "dataset_source": schema.dataset_source,
            "provenance_url": schema.provenance_url,
            "is_synthetic": schema.is_synthetic,
        },
        "feature_importance": primary["feature_importance"],
    }
    dump_joblib(Path(cfg.paths.model_dir) / "serving_bundle.joblib", serving_bundle)

    final_summary = _build_final_model_summary(
        cfg=cfg,
        schema=schema,
        metrics_summary=metrics_summary,
        threshold_summary=threshold_summary,
        primary_model_name=primary_model_name,
        selection_reason=selection_reason,
    )
    write_json(Path(cfg.paths.artifact_dir) / "final_model_summary.json", final_summary)

    claims_md = Path(cfg.paths.artifact_dir) / "final_project_claims.md"
    claims_md.write_text(
        "\n".join(
            [
                "# Final Project Claims",
                "",
                "- This repository is a risk scoring and decision-threshold project, not just a binary classifier demo.",
                "- Dataset defaults to PaySim and is synthetic; this is explicitly documented.",
                f"- Current serving model: `{primary_model_name}` (`{selection_reason}`).",
                "- Calibration is applied only when validation Brier score improves.",
                "- Threshold regimes include balanced, precision-focused, recall-focused, and cost-sensitive modes.",
                "- Cost-sensitive regime uses configured false-positive and false-negative costs from config.",
                "- Boosted-model results are reported transparently, including backend fallback when applicable.",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "primary_model": primary_model_name,
        "primary_model_selection_reason": selection_reason,
        "threshold_summary": threshold_summary,
        "metrics_summary": metrics_summary,
    }
