"""CLI: consensus-rate figures for orthoparagroups_aligned metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.homology.align_consensus_viz import DEFAULT_THRESHOLDS, plot_consensus_metrics


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("mag/orthoparagroups_aligned/metrics"),
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("mag/orthoparagroups_aligned/figures"),
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--limit", type=int, default=0, help="Use first N clusters (0=all)")
    p.add_argument(
        "--thresholds",
        type=str,
        default=",".join(str(t) for t in DEFAULT_THRESHOLDS),
        help="Comma-separated similarity rate thresholds",
    )
    args = p.parse_args(argv)
    thresholds = tuple(float(x) for x in args.thresholds.split(",") if x.strip())
    if not thresholds:
        raise ValueError("No thresholds parsed")
    written = plot_consensus_metrics(
        args.metrics_dir,
        args.outdir,
        limit=args.limit,
        dpi=args.dpi,
        thresholds=thresholds,
    )
    print(f"[consensus_viz] wrote {len(written)} paths under {args.outdir}", flush=True)
    for path in written[:30]:
        print(f"  {path}", flush=True)
    if len(written) > 30:
        print(f"  ... ({len(written) - 30} more)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
