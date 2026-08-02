"""Rebuild pairwise heatmaps from saved matrices (no distance recomputation).

Example::

  python -m src.embed.replot_pairwise \\
    --out results/embed_legnet/pairwise \\
    --label-fontsize 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.embed.pairwise import (
    DEFAULT_LAYERS,
    HEATMAP_SCORES,
    write_heatmaps_from_matrices,
)

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "embed_legnet" / "pairwise",
        help="Directory with matrix_*.npy and pairwise_compare.json",
    )
    p.add_argument(
        "--layers",
        type=str,
        default=",".join(DEFAULT_LAYERS),
    )
    p.add_argument(
        "--scores",
        type=str,
        default=",".join(HEATMAP_SCORES),
    )
    p.add_argument("--label-fontsize", type=float, default=20.0)
    args = p.parse_args(argv)

    out = args.out if args.out.is_absolute() else ROOT / args.out
    meta = out / "pairwise_compare.json"
    if not meta.is_file():
        raise SystemExit(f"missing {meta}")
    payload = json.loads(meta.read_text(encoding="utf-8"))
    keys = list(payload["runs"])
    layers = tuple(s.strip() for s in args.layers.split(",") if s.strip())
    scores = tuple(s.strip() for s in args.scores.split(",") if s.strip())
    written = write_heatmaps_from_matrices(
        out,
        keys,
        layers=layers,
        scores=scores,
        label_fontsize=float(args.label_fontsize),
    )
    print(f"[replot] wrote {len(written)} files under {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
