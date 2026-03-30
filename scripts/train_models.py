from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loaders import load_and_normalize
from src.data.schemas import DatasetSchema
from src.data.split import split_dataset
from src.models.train import save_training_artifacts, train_all_models
from src.utils.config import load_config
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def _looks_like_configured_dataset(train_df: pd.DataFrame, cfg) -> bool:
    if cfg.schema.target_col not in train_df.columns:
        return False

    if cfg.dataset.name.lower() == "paysim":
        required = {"step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"}
        return required.issubset(set(train_df.columns))

    return True


def _load_or_create_splits(cfg):
    split_dir = Path(cfg.paths.split_data_dir)
    train_path = split_dir / "train.csv"
    val_path = split_dir / "val.csv"
    test_path = split_dir / "test.csv"

    if train_path.exists() and val_path.exists() and test_path.exists():
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        if _looks_like_configured_dataset(train_df, cfg):
            return train_df, val_df, test_df

        logger.warning("Existing split files do not match configured dataset/schema. Regenerating splits.")

    df, schema = load_and_normalize(cfg)
    splits = split_dataset(df, schema, cfg.split, cfg.random_seed)
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, split_df in splits.items():
        split_df.to_csv(split_dir / f"{name}.csv", index=False)
    return splits["train"], splits["val"], splits["test"]


def main() -> None:
    cfg = load_config()
    setup_logging(cfg.logging.level)

    train_df, val_df, test_df = _load_or_create_splits(cfg)
    schema = DatasetSchema(
        target_col=cfg.schema.target_col,
        id_cols=cfg.schema.id_cols,
        time_col=cfg.schema.time_col,
        dataset_name=cfg.dataset.name,
        dataset_source=cfg.dataset.source,
        provenance_url=cfg.dataset.provenance_url,
        is_synthetic=cfg.dataset.is_synthetic,
    )

    output = train_all_models(train_df, val_df, test_df, schema, cfg)
    artifact_summary = save_training_artifacts(train_df, val_df, test_df, schema, cfg, output)

    logger.info("Primary model: %s", artifact_summary["primary_model"])
    logger.info("Primary selection reason: %s", artifact_summary["primary_model_selection_reason"])
    logger.info("Model metrics summary: %s", cfg.paths.comparison_metrics_path)
    logger.info("Threshold summary: %s", cfg.paths.threshold_summary_path)
    logger.info("Final model summary: %s", Path(cfg.paths.artifact_dir) / "final_model_summary.json")


if __name__ == "__main__":
    main()
