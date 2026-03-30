from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config


def main() -> None:
    cfg = load_config()
    uvicorn.run("src.api.app:app", host=cfg.api.host, port=cfg.api.port, reload=False)


if __name__ == "__main__":
    main()
