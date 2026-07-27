#!/usr/bin/env python3
"""Thin wrapper — delegates to src.train_viz.compute_final_scores."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TARGET = ROOT / "src" / "train_viz" / "compute_final_scores.py"
if not TARGET.exists():
    print(f"ERROR: missing {TARGET}", file=sys.stderr)
    raise SystemExit(2)
sys.argv[0] = str(TARGET)
runpy.run_path(str(TARGET), run_name="__main__")
