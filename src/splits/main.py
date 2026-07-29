#!/usr/bin/env python3
"""Split dispatcher — run a strategy implemented under src/splits/.

Workflow (skill contract):
  1. Write / update strategy code in src/splits/<id>.py
  2. Exec this module
  3. Only then treat folds as produced for downstream (@caduceus, etc.)

Example:
  python -m src.splits.main --strategy random --raw raw --ready ready --seed 42
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


STRATEGY_RUNNERS = {
    "random": ("src.splits.random", "run_random_split"),
    # GC / SBS strategies are assigned via src.pipeline.split_predict (type=gc).
    # Legacy main dispatcher keeps random materialization for Caduceus-ready trees.
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--strategy",
        required=True,
        help="Split strategy id (must match splits/<id>.md and src/splits/<id>.py)",
    )
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--raw", type=Path, default=Path("raw"))
    ap.add_argument(
        "--ready",
        type=Path,
        default=None,
        help="Ready dir (default: ready/ or data_ready/)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output root (default: splits/<strategy>)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for smoke tests",
    )
    args = ap.parse_args(argv)

    strategy = args.strategy.strip().lower()
    if strategy not in STRATEGY_RUNNERS:
        known = ", ".join(sorted(STRATEGY_RUNNERS))
        print(
            f"ERROR: unknown strategy {strategy!r}. Implemented: {known}. "
            f"Add src/splits/{strategy}.py and register it in STRATEGY_RUNNERS.",
            file=sys.stderr,
        )
        return 2

    root = args.root.resolve()
    # Ensure project root is importable
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    mod_name, fn_name = STRATEGY_RUNNERS[strategy]
    mod = importlib.import_module(mod_name)
    runner = getattr(mod, fn_name)

    out = args.out
    if out is None:
        out = Path("splits") / strategy

    meta = runner(
        root,
        raw_dir=args.raw,
        ready_dir=args.ready,
        out_dir=out,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    print(json.dumps({"strategy": strategy, "status": "ok", "meta": meta}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
