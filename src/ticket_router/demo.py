"""One-command demo: prepare data + baseline if missing, then launch the app."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run(module: str, *args: str) -> None:
    print(f"→ {module} {' '.join(args)}".rstrip())
    subprocess.run([sys.executable, "-m", module, *args], check=True)


def ensure_baseline() -> None:
    """Data split, TF-IDF model, and its predictions: about a minute from scratch."""
    from ticket_router.models import TFIDF_PATH
    from ticket_router.results import OUTPUT_DIR, predictions_path

    if not Path("data/test.csv").exists():
        _run("ticket_router.data")
    if not TFIDF_PATH.exists():
        _run("ticket_router.train", "--model", "tfidf")
    if not predictions_path("tfidf", "test", OUTPUT_DIR).exists():
        _run("ticket_router.predict", "--model", "tfidf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default=None, help="Default: 8501, or the next free port")
    ap.add_argument("--no-setup", action="store_true", help="Skip data/baseline checks")
    args = ap.parse_args()

    os.chdir(PROJECT_ROOT)
    if not args.no_setup:
        ensure_baseline()
    cmd = [sys.executable, "-m", "streamlit", "run", "app/Home.py", "--browser.gatherUsageStats", "false"]
    if args.port:
        cmd += ["--server.port", args.port]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
