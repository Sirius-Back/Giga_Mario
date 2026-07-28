#!/usr/bin/env python3
"""Adapt skill CLI — delegates to src/preprocessing.py (Locked CDS±10 kb path)."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
PREPROCESS = PROJECT_ROOT / "src" / "preprocessing.py"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--legacy":
        print(
            "ERROR: --legacy removed. Use src/preprocessing.py (CDS±10 kb Locked path). "
            "Archived ±200 bp code: scripts/_archive/adapt.py",
            file=sys.stderr,
        )
        return 2
    if not PREPROCESS.exists():
        print(f"ERROR: missing {PREPROCESS}", file=sys.stderr)
        return 2
    sys.argv = [str(PREPROCESS), *argv]
    runpy.run_path(str(PREPROCESS), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
