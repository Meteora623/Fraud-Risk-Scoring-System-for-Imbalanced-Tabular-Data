from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def _find_largest_csv(root: Path) -> Path | None:
    csv_files = list(root.rglob("*.csv"))
    if not csv_files:
        return None
    return max(csv_files, key=lambda p: p.stat().st_size)


def main() -> None:
    cfg = load_config()
    setup_logging(cfg.logging.level)

    out_path = Path(cfg.paths.raw_data_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        logger.info("Raw dataset already exists at %s", out_path)
        return

    kaggle_ref = cfg.dataset.kaggle_dataset
    if not kaggle_ref:
        logger.error("No Kaggle dataset reference configured. Set dataset.kaggle_dataset in config.")
        return

    kaggle_cli = shutil.which("kaggle")
    if not kaggle_cli:
        logger.error(
            "Kaggle CLI not found. Install/configure Kaggle CLI or place PaySim CSV manually at %s",
            out_path,
        )
        return

    download_dir = out_path.parent / "_kaggle_download"
    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        cmd = [kaggle_cli, "datasets", "download", "-d", kaggle_ref, "-p", str(download_dir), "--unzip"]
        logger.info("Downloading PaySim dataset via Kaggle CLI: %s", kaggle_ref)
        subprocess.run(cmd, check=True)

        csv_path = _find_largest_csv(download_dir)
        if csv_path is None:
            raise FileNotFoundError("No CSV file found in downloaded Kaggle dataset.")

        out_path.write_bytes(csv_path.read_bytes())
        logger.info("Saved PaySim dataset to %s", out_path)
    except Exception as exc:
        logger.exception("Kaggle download failed: %s", exc)
        logger.error(
            "Manual fallback: place PaySim CSV at %s with columns such as step,type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest,isFraud",
            out_path,
        )


if __name__ == "__main__":
    main()
