from __future__ import annotations

import os
import sys
from pathlib import Path

from streamlit.web.cli import main as streamlit_main


def _resource_path(name: str) -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / name
    return Path(__file__).resolve().parent / name


def main() -> int:
    app_path = _resource_path("app.py")
    os.chdir(app_path.parent)
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.address=0.0.0.0",
        "--server.port=8501",
        "--server.headless=true",
    ]
    return int(streamlit_main())


if __name__ == "__main__":
    raise SystemExit(main())
