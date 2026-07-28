#!/usr/bin/env python3
"""Adapt skill CLI — delegates to src.pipeline.adapt."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    from src.pipeline.adapt import main as adapt_main
    return adapt_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
