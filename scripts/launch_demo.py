from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "streamlit", "run", str(root / "src" / "demo" / "streamlit_app.py")]
    raise SystemExit(subprocess.call(cmd, env=env, cwd=root))


if __name__ == "__main__":
    main()
