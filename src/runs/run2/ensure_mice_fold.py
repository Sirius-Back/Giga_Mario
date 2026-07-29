"""Ensure mice ZSV fold.csv exists for ready_legnet (idempotent)."""
from __future__ import annotations

from pathlib import Path

from src.pipeline.generate_fold import run_generate_fold

ROOT = Path(__file__).resolve().parents[3]
PANEL = ROOT / "ready_legnet"
MICE = "GCF_000001635.27"


def main() -> int:
    prep = PANEL / "prepare_fold.csv"
    prep.write_text(
        "identificator|column|fold\n"
        f"{MICE}|genome|zsv\n",
        encoding="utf-8",
    )
    out = run_generate_fold(PANEL / "ID.csv", prep, outdir=PANEL)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
