from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier


def build_boosted_model(cfg: dict, scale_pos_weight: float, random_seed: int) -> Any:
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=cfg.get("n_estimators", 500),
            learning_rate=cfg.get("learning_rate", 0.05),
            max_depth=cfg.get("max_depth", 6),
            subsample=cfg.get("subsample", 0.9),
            colsample_bytree=cfg.get("colsample_bytree", 0.9),
            reg_lambda=cfg.get("reg_lambda", 1.0),
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            random_state=random_seed,
            n_jobs=-1,
        )
    except Exception:
        return HistGradientBoostingClassifier(
            learning_rate=cfg.get("learning_rate", 0.05),
            max_depth=cfg.get("max_depth", 6),
            max_iter=cfg.get("n_estimators", 500),
            random_state=random_seed,
        )
