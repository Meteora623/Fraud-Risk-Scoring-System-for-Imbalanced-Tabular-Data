from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.audit import build_audit_report
from src.data.loaders import load_and_normalize
from src.data.split import split_dataset
from src.utils.config import load_config
from src.utils.io import write_json
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    cfg = load_config()
    setup_logging(cfg.logging.level)

    df, schema = load_and_normalize(cfg)
    splits = split_dataset(df, schema, cfg.split, cfg.random_seed)

    Path(cfg.paths.split_data_dir).mkdir(parents=True, exist_ok=True)
    for name, split_df in splits.items():
        split_df.to_csv(Path(cfg.paths.split_data_dir) / f"{name}.csv", index=False)

    Path(cfg.paths.processed_data_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.paths.processed_data_path, index=False)

    report = build_audit_report(df, schema, splits)
    write_json(cfg.paths.audit_report_path, report)

    logger.info("Audit saved: %s", cfg.paths.audit_report_path)
    logger.info("Split files saved under: %s", cfg.paths.split_data_dir)


if __name__ == "__main__":
    main()
