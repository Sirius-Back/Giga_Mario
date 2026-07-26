#!/usr/bin/env python3
"""Thin wrapper — delegates to src.train_viz (canonical package)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # …/scripts → project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_viz.viz import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
