"""Render full homology network (components with n>5).

Run from project root::

  python -m src.run.homology_graph.plot_full \\
    --edges mag/homology_graph/edges.tsv.gz \\
    --out mag/homology_graph/graph_network_full.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.homology.graph import load_edge_table
from src.homology.visualize import plot_network_full_png
from src.pipeline.mem_guard import ensure_allocation_fits


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edges", type=Path, default=Path("mag/homology_graph/edges.tsv.gz"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path("mag/homology_graph/graph_network_full.png"),
        help="Output stem/path (.png); PDF written alongside",
    )
    p.add_argument("--min-nodes", type=int, default=6, help="Keep components with n>=this")
    p.add_argument("--edge-alpha", type=float, default=0.06)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--peak-ram-gib", type=float, default=24.0)
    args = p.parse_args(argv)

    if not args.edges.is_file():
        raise FileNotFoundError(f"Edge table missing: {args.edges}")

    ensure_allocation_fits(
        int(args.peak_ram_gib * (1024**3)),
        max_used_fraction=0.95,
        timeout_sec=600.0,
        label="homology_graph_full_png",
    )

    edges = load_edge_table(args.edges)
    plot_network_full_png(
        edges,
        args.out,
        min_nodes=args.min_nodes,
        edge_alpha=args.edge_alpha,
        dpi=args.dpi,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
