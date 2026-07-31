"""CLI: distribution figures for mag/orthoparagroups/clusters.tsv."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.homology.orthoparagroups_viz import plot_clusters_tsv


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--clusters",
        type=Path,
        default=Path("mag/orthoparagroups/clusters.tsv"),
        help="Path to clusters.tsv",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("mag/orthoparagroups/figures"),
        help="Output directory for PDF/SVG/PNG + Altair HTML/VL",
    )
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)
    written = plot_clusters_tsv(args.clusters, args.outdir, dpi=args.dpi)
    print(f"[orthoparagroups_viz] wrote {len(written)} files under {args.outdir}", flush=True)
    for path in written:
        print(f"  {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
