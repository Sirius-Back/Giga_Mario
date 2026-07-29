"""Ensure mice ZSV fold.csv exists for ready_caduceus (idempotent)."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from src.pipeline.generate_fold import run_generate_fold

ROOT = Path(__file__).resolve().parents[3]
PANEL = ROOT / "ready_caduceus"
MICE = "GCF_000001635.27"


def main() -> int:
    if not (PANEL / "ID.csv").is_file():
        raise FileNotFoundError(f"Missing panel ID.csv: {PANEL / 'ID.csv'}")
    prep = PANEL / "prepare_fold.csv"
    prep.write_text(
        "identificator|column|fold\n"
        f"{MICE}|genome|zsv\n",
        encoding="utf-8",
    )
    out = run_generate_fold(PANEL / "ID.csv", prep, outdir=PANEL)
    rows = list(csv.DictReader(out.open(), delimiter="|"))
    folds = Counter(r["fold"] for r in rows)
    mice = sum(1 for r in rows if r["genome"].startswith(MICE) and r["fold"] == "zsv")
    leak = sum(1 for r in rows if r["genome"].startswith(MICE) and r["fold"] != "zsv")
    print(out)
    print(f"fold_counts={dict(folds)} mice_zsv={mice} mice_non_zsv={leak}")
    if mice == 0 or leak:
        raise RuntimeError(f"Mice ZSV fold assignment failed: mice_zsv={mice} leak={leak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
