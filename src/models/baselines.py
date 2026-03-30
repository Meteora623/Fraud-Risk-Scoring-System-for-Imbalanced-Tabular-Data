from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_logistic_regression(cfg: dict) -> LogisticRegression:
    return LogisticRegression(
        C=cfg.get("C", 1.0),
        max_iter=cfg.get("max_iter", 1000),
        class_weight=cfg.get("class_weight", "balanced"),
        solver="lbfgs",
        n_jobs=None,
    )


def build_random_forest(cfg: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.get("n_estimators", 300),
        max_depth=cfg.get("max_depth", 12),
        min_samples_leaf=cfg.get("min_samples_leaf", 2),
        class_weight=cfg.get("class_weight", "balanced_subsample"),
        n_jobs=cfg.get("n_jobs", 1),
        random_state=cfg.get("random_state", 42),
    )
