#!/usr/bin/env python3
"""Adapt skill CLI — delegates raw→data_ready conversion to src/preprocessing.py.

Legacy gene±200bp / region_split path remains available via --legacy.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# scripts → adapt → skills → .cursor → project  ⇒ parents[3] == project root
PROJECT_ROOT = SCRIPT_DIR.parents[3]
PREPROCESS = PROJECT_ROOT / "src" / "preprocessing.py"
LEGACY = SCRIPT_DIR / "adapt_legacy.py"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Allow `adapt.py --legacy ...` to run the previous adapt implementation if archived.
    if argv and argv[0] == "--legacy":
        legacy = SCRIPT_DIR / "adapt_legacy.py"
        if not legacy.exists():
            # Fall back to in-place historical module name if present as adapt_legacy
            print(
                "ERROR: --legacy requested but adapt_legacy.py not found. "
                "Use src/preprocessing.py for the Locked 2026-07-27 pipeline.",
                file=sys.stderr,
            )
            return 2
        sys.argv = [str(legacy), *argv[1:]]
        runpy.run_path(str(legacy), run_name="__main__")
        return 0

    if not PREPROCESS.exists():
        print(f"ERROR: missing {PREPROCESS}", file=sys.stderr)
        return 2

    # Map skill-style flags onto preprocessing.py
    ap = argparse.ArgumentParser(
        description="Adapt: raw FNA/GTF/TPM → data_ready (CDS±10kb + non-coding match)"
    )
    ap.add_argument("--raw", type=Path, default=PROJECT_ROOT / "raw")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data_ready")
    ap.add_argument("--input", default=None, help="Alias for --raw (auto|path)")
    ap.add_argument("--flank", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--genomes", nargs="*", default=None)
    ap.add_argument("--max-genes", type=int, default=None)
    ap.add_argument("--window-size", type=int, default=None, help="Ignored (kept for CLI compat)")
    ap.add_argument("--config", type=Path, default=None, help="Ignored (kept for CLI compat)")
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args, _unknown = ap.parse_known_args(argv)

    raw = args.raw
    if args.input and args.input != "auto":
        raw = Path(args.input)
    if not raw.is_absolute():
        raw = (args.root / raw).resolve()
    out = args.out if args.out.is_absolute() else (args.root / args.out).resolve()

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.preprocessing import run  # noqa: WPS433

    return run(
        raw_dir=raw,
        out_dir=out,
        flank=args.flank,
        seed=args.seed,
        genomes=args.genomes,
        max_genes=args.max_genes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
