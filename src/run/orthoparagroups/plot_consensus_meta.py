"""CLI: meta-cluster consensus figures (coverage filter, fine bins, ATG)."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.homology.align_consensus_meta_viz import (
    DEFAULT_N_BINS,
    MAX_META_CLUSTERS,
    MIN_ALIGN_FRAC,
    MIN_META_CLUSTERS,
    run_meta_viz,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-dir", type=Path, default=Path("mag/orthoparagroups_aligned/metrics"))
    p.add_argument("--aln-dir", type=Path, default=Path("mag/orthoparagroups_aligned"))
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("mag/orthoparagroups_aligned/figures_meta"),
    )
    p.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    p.add_argument("--min-align-frac", type=float, default=MIN_ALIGN_FRAC)
    p.add_argument(
        "--min-k",
        type=int,
        default=MIN_META_CLUSTERS,
        help="Lower bound for silhouette search (default 10)",
    )
    p.add_argument(
        "--max-k",
        type=int,
        default=MAX_META_CLUSTERS,
        help="Upper bound for silhouette search / meta-clusters (default 20)",
    )
    p.add_argument(
        "--fixed-k",
        type=int,
        default=0,
        help="If >0, force this k (skip silhouette search)",
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--from-tables",
        action="store_true",
        help="Redraw figures from existing TSVs in --outdir (skip align/cluster)",
    )
    args = p.parse_args(argv)
    if args.from_tables:
        from src.homology.align_consensus_meta_viz import redraw_meta_figures_from_tables

        written = redraw_meta_figures_from_tables(args.outdir, dpi=args.dpi)
    else:
        written = run_meta_viz(
            args.metrics_dir,
            args.outdir,
            aln_dir=args.aln_dir,
            limit=args.limit,
            n_bins=args.n_bins,
            min_align_frac=args.min_align_frac,
            max_k=args.max_k,
            min_k=args.min_k,
            fixed_k=(args.fixed_k if args.fixed_k > 0 else None),
            dpi=args.dpi,
            seed=args.seed,
        )
    print(f"[meta_viz] wrote {len(written)} paths under {args.outdir}", flush=True)
    for path in written[:40]:
        print(f"  {path}", flush=True)
    if len(written) > 40:
        print(f"  ... ({len(written) - 40} more)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
