"""Build mammals-11 ortholog/paralog graph from Ensembl Compara dumps.

Run from project root::

  python -m src.run.homology_graph.build_mammals11 \\
    --ensembl-data mag/ensembl/data \\
    --outdir mag/homology_graph
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.homology.graph import (
    PANEL_SPECIES,
    build_homology_graph,
    summarize_graph,
    write_edge_table,
    write_summary,
)
from src.homology.visualize import (
    compute_graph_stats,
    plot_characteristics_altair,
    plot_characteristics_cnsplots,
    plot_network_png,
    write_stats_tables,
)
from src.pipeline.mem_guard import ensure_allocation_fits


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ensembl-data",
        type=Path,
        default=Path("mag/ensembl/data"),
        help="Root with <species>/compara_homology/…tsv.gz",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("mag/homology_graph"),
        help="Output directory for edges, figures, summaries",
    )
    p.add_argument("--release", type=int, default=116)
    p.add_argument(
        "--high-confidence-only",
        action="store_true",
        help="Keep only Compara rows with is_high_confidence=1",
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write edge table only (no PNG / Altair / cnsplots)",
    )
    p.add_argument(
        "--peak-ram-gib",
        type=float,
        default=16.0,
        help="Declared extra RAM peak before launch (mem_guard)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ensembl_data = args.ensembl_data
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    if not ensembl_data.is_dir():
        raise FileNotFoundError(f"Ensembl data root missing: {ensembl_data}")

    ensure_allocation_fits(
        int(args.peak_ram_gib * (1024**3)),
        max_used_fraction=0.95,
        timeout_sec=600.0,
        poll_sec=15.0,
        label="homology_graph_mammals11",
    )

    result = build_homology_graph(
        ensembl_data,
        species=PANEL_SPECIES,
        release=args.release,
        high_confidence_only=args.high_confidence_only,
    )
    edges_sorted = result.sorted_edges()
    edge_path = outdir / "edges.tsv.gz"
    write_edge_table(edges_sorted, edge_path)

    summary = summarize_graph(edges_sorted)
    summary.update(
        {
            "panel_species": list(PANEL_SPECIES),
            "release": args.release,
            "high_confidence_only": args.high_confidence_only,
            "n_rows_read": result.n_rows_read,
            "n_rows_kept_raw": result.n_rows_kept,
            "n_self_loops_skipped": result.n_self_loops_skipped,
            "unique_relation_counts": dict(result.relation_counts),
            "source_files": result.per_species_files,
            "edge_table": str(edge_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "random_seed": args.seed,
        }
    )
    write_summary(summary, outdir / "summary.json")

    plot_paths: list[str] = []
    if not args.skip_plots:
        stats = compute_graph_stats(edges_sorted)
        write_stats_tables(stats, outdir / "stats")
        net = plot_network_png(
            edges_sorted,
            outdir / "graph_network.png",
            dpi=args.dpi,
            seed=args.seed,
        )
        plot_paths.append(str(net))
        plot_paths.extend(str(p) for p in plot_characteristics_cnsplots(stats, figdir, dpi=args.dpi))
        plot_paths.extend(str(p) for p in plot_characteristics_altair(stats, figdir))

    manifest = {
        "edge_table": str(edge_path),
        "summary": str(outdir / "summary.json"),
        "plots": plot_paths,
        "n_edges": summary["n_edges"],
        "n_nodes": summary["n_nodes"],
    }
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
