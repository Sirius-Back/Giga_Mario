#!/usr/bin/env python3
"""Filter MARKED_pangenome ∩ PARSED → MARKED_parsed (pangenome step 2).

Example:
  python -m src.splits.intersect_pangenome \\
    --marked-pangenome output/pg/MARKED_pangenome \\
    --parsed ready_legnet/PARSED \\
    --outdir output/pg \\
    --id-csv ready_legnet/ID.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.splits.pangenome import (
    filter_ids_to_parsed,
    intersect_pangenome,
    materialize_marked_subset,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--marked-pangenome", type=Path, required=True)
    p.add_argument("--parsed", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--id-csv", type=Path, default=None)
    p.add_argument("--genome", dest="genomes", action="append", default=None)
    p.add_argument("--max-ids", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to materialize MARKED_parsed",
    )
    args = p.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.id_csv is not None or args.genomes or args.max_ids is not None:
        kept = filter_ids_to_parsed(
            marked_dir=args.marked_pangenome,
            parsed_dir=args.parsed,
            id_csv=args.id_csv,
            genomes=args.genomes,
            max_ids=args.max_ids,
            seed=args.seed,
        )
    else:
        kept = intersect_pangenome(args.marked_pangenome, args.parsed)

    marked_parsed = materialize_marked_subset(
        args.marked_pangenome,
        outdir / "MARKED_parsed",
        kept,
        mode=args.mode,
    )
    meta = {
        "n_ids": len(kept),
        "marked_pangenome": str(args.marked_pangenome),
        "parsed": str(args.parsed),
        "marked_parsed": str(marked_parsed),
    }
    (outdir / "intersect_pangenome_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(marked_parsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
