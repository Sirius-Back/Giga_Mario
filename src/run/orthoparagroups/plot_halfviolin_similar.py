"""CLI: presentation half-violin of similar lengths (ortho|para × thr 0.8↑ / 0.5↓).

Reads the existing ``table_similar_lengths.tsv`` from consensus viz (same metric
as Figure_03/04). Labels stripped for slide overlays; aspect 9:5.

Run from project root::

  python -m src.run.orthoparagroups.plot_halfviolin_similar \\
    --length-table mag/orthoparagroups_aligned/figures/table_similar_lengths.tsv \\
    --outdir mag/orthoparagroups_aligned/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.homology.align_consensus_viz import (
    HALFVIOLIN_THR_DOWN,
    HALFVIOLIN_THR_UP,
    plot_halfviolin_similar_lengths,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--length-table",
        type=Path,
        default=Path("mag/orthoparagroups_aligned/figures/table_similar_lengths.tsv"),
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("mag/orthoparagroups_aligned/figures"),
    )
    p.add_argument(
        "--metric",
        type=str,
        default="similar_length_total",
        choices=("similar_length_total", "similar_length_longest_run", "similar_fraction"),
    )
    p.add_argument("--thr-up", type=float, default=HALFVIOLIN_THR_UP)
    p.add_argument("--thr-down", type=float, default=HALFVIOLIN_THR_DOWN)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument(
        "--stem",
        type=str,
        default="Figure_11_halfviolin_ortho_para_thr0p8_up_0p5_down",
    )
    p.add_argument(
        "--trim",
        action="store_true",
        help="Trim violin KDE to data min/max (default: no trim, cut=2 bandwidths)",
    )
    args = p.parse_args(argv)
    if not args.length_table.is_file():
        raise FileNotFoundError(f"similar-length table missing: {args.length_table}")
    df = pd.read_csv(args.length_table, sep="\t")
    if df.empty:
        raise ValueError(f"Empty length table: {args.length_table}")
    written = plot_halfviolin_similar_lengths(
        df,
        args.outdir,
        metric=args.metric,
        thr_up=args.thr_up,
        thr_down=args.thr_down,
        dpi=args.dpi,
        stem=args.stem,
        trim=bool(args.trim),
        y_gap=0.0,
    )
    print(f"[halfviolin] wrote {len(written)} paths", flush=True)
    for path in written:
        print(f"  {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
