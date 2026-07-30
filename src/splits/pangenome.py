"""Pangenome / cactus-like repeat-graph split strategy.

Caption: ``splits/pangenome.md``. Wired into ``split-predict`` as ``type=pangenome``.

Pipeline (strategy-owned steps):
  1. Filter MARKED IDs to those present in PARSED (intersect).
  2. Build a k-mer contingency / repeat graph in C++ (no pairwise distances).
  3. Cluster regions by connected components of shared-k-mer contingency.
  4. Assign clusters → train/val/test (Caduceus-aligned ratios; ZSV held out).
  5. Render the region co-occurrence graph (connected nodes only).

Adapt (raw → MARKED) remains outside this module; panels such as
``ready_legnet`` already provide MARKED + PARSED.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from src.pipeline.common import read_csv
from src.pipeline.generate_fold import is_zsv_fold, normalize_fold_label
from src.splits.common import assign_folds_random, assign_folds_stratified
from src.splits.sbs.assign import (
    aggregate_stratification_per_fold,
    assignment_rows_to_split_csv,
    write_assignment_table,
)
from src.splits.sbs.fna_io import load_fna_directory

__all__ = (
    "SPLIT_ID",
    "intersect_pangenome",
    "filter_ids_to_parsed",
    "materialize_marked_pangenome",
    "build_contingency_clusters",
    "render_contingency_graph",
    "run_pangenome_split_assign",
)

SPLIT_ID = "pangenome"
DEFAULT_K = 21
DEFAULT_MIN_SHARED = 1


def intersect_pangenome(
    marked_dir: Path,
    parsed_dir: Path,
    ids: Sequence[str] | None = None,
) -> list[str]:
    """Return IDs that exist in both MARKED (``.fa``) and PARSED (``.ext``).

    This is the obligatory filter step before repeat-graph construction.
    """
    marked_dir = Path(marked_dir)
    parsed_dir = Path(parsed_dir)
    if not marked_dir.is_dir():
        raise FileNotFoundError(f"MARKED directory missing: {marked_dir}")
    if not parsed_dir.is_dir():
        raise FileNotFoundError(f"PARSED directory missing: {parsed_dir}")

    if ids is None:
        candidates = sorted(p.stem for p in marked_dir.glob("*.fa"))
    else:
        candidates = [str(i) for i in ids]

    kept: list[str] = []
    for rid in candidates:
        if not (marked_dir / f"{rid}.fa").is_file():
            continue
        if not (parsed_dir / f"{rid}.ext").is_file():
            continue
        kept.append(rid)
    if not kept:
        raise ValueError(
            f"no IDs present in both MARKED ({marked_dir}) and PARSED ({parsed_dir})"
        )
    return kept


def filter_ids_to_parsed(
    *,
    marked_dir: Path,
    parsed_dir: Path,
    id_csv: Path | None = None,
    genomes: Sequence[str] | None = None,
    max_ids: int | None = None,
    seed: int = 42,
) -> list[str]:
    """Intersect MARKED∩PARSED, optionally restrict by genome and/or ``max_ids``."""
    ids: list[str] | None = None
    if id_csv is not None:
        rows = read_csv(Path(id_csv))
        if not rows or "ID" not in rows[0]:
            raise ValueError(f"id_csv missing ID column: {id_csv}")
        if genomes:
            gset = {str(g) for g in genomes}
            if "genome" not in rows[0]:
                raise ValueError(
                    f"id_csv missing genome column required for genome filter: {id_csv}"
                )
            ids = [r["ID"].strip() for r in rows if r.get("genome", "").strip() in gset]
        else:
            ids = [r["ID"].strip() for r in rows]
        ids = [i for i in ids if i]

    kept = intersect_pangenome(marked_dir, parsed_dir, ids=ids)
    if max_ids is not None and len(kept) > int(max_ids):
        rng = random.Random(int(seed))
        kept = list(kept)
        rng.shuffle(kept)
        kept = sorted(kept[: int(max_ids)], key=lambda x: (len(x), x))
    return kept


def materialize_marked_pangenome(
    marked_dir: Path,
    out_dir: Path,
    ids: Sequence[str],
    *,
    mode: str = "symlink",
) -> Path:
    """Write filtered ``MARKED_pangenome`` as symlinks (default) or hard copies."""
    marked_dir = Path(marked_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode not in {"symlink", "copy"}:
        raise ValueError(f"mode must be symlink|copy; got {mode!r}")
    for rid in ids:
        src = marked_dir / f"{rid}.fa"
        if not src.is_file():
            raise FileNotFoundError(f"MARKED fasta missing for ID {rid}: {src}")
        dst = out_dir / f"{rid}.fa"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if mode == "symlink":
            dst.symlink_to(src.resolve())
        else:
            dst.write_bytes(src.read_bytes())
    return out_dir


def build_contingency_clusters(
    sequences: list[str],
    *,
    k: int = DEFAULT_K,
    min_shared: int = DEFAULT_MIN_SHARED,
    max_edges: int = 100_000,
    collect_edges: bool = True,
) -> Any:
    """C++ contingency clustering on sequences; returns ``ContingencyGraphResult``."""
    from src.splits.pangenome_native import get_native_graph

    return get_native_graph().contingency_clusters(
        sequences,
        k=k,
        min_shared=min_shared,
        max_edges=max_edges,
        collect_edges=collect_edges,
    )


def _load_fold_map(fold_csv: Path | None) -> dict[str, str]:
    if fold_csv is None:
        return {}
    rows = read_csv(Path(fold_csv))
    out: dict[str, str] = {}
    for row in rows:
        out[row["ID"].strip()] = normalize_fold_label(row["fold"])
    return out


def _load_strat_map(strat_csv: Path | None) -> dict[str, dict[str, str]]:
    if strat_csv is None:
        return {}
    rows = read_csv(Path(strat_csv))
    return {row["ID"].strip(): row for row in rows}


def _assign_folds_to_train_test(
    fold_ids: list[str],
    *,
    seed: int,
    fold_strata: dict[str, str] | None,
    ratios: tuple[float, float, float] | None,
) -> dict[str, str]:
    if not fold_ids:
        return {}
    if len(fold_ids) < 3:
        labels = ["train", "val", "test"]
        return {fid: labels[i % 3] for i, fid in enumerate(sorted(fold_ids))}
    rng = random.Random(seed)
    if fold_strata:
        strata = [fold_strata[f] for f in fold_ids]
        labels = assign_folds_stratified(fold_ids, strata, rng, ratios=ratios)
        return dict(zip(fold_ids, labels))
    order = list(fold_ids)
    rng.shuffle(order)
    labels = assign_folds_random(len(order), ratios=ratios)
    return {fid: lab for fid, lab in zip(order, labels)}


def assign_from_contingency(
    ids: Sequence[str],
    cluster_ids: Sequence[int],
    *,
    fold_csv: Path | None = None,
    stratification_csv: Path | None = None,
    seed: int = 42,
    ratios: tuple[float, float, float] | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Map contingency clusters → assignment rows (ZSV held out)."""
    if len(ids) != len(cluster_ids):
        raise ValueError("ids and cluster_ids length mismatch")
    fold_map = _load_fold_map(fold_csv)
    strat_map = _load_strat_map(stratification_csv)

    zsv_ids: list[str] = []
    assignable: list[str] = []
    cluster_by_id = {str(rid): int(cid) for rid, cid in zip(ids, cluster_ids)}
    for rid in ids:
        rid = str(rid)
        raw = fold_map.get(rid, "0")
        if is_zsv_fold(raw):
            zsv_ids.append(rid)
        else:
            assignable.append(rid)

    rows: list[dict[str, str]] = []
    meta: dict[str, Any] = {
        "n_total": len(ids),
        "n_zsv": len(zsv_ids),
        "n_assignable": len(assignable),
        "seed": seed,
        "method_used": "contingency_cc",
    }
    for rid in zsv_ids:
        rows.append(
            {
                "region": rid,
                "cluster": "zsv",
                "train_test": "zsv",
                "fold": "zsv",
                "additional": json.dumps({"method": "contingency_cc"}),
            }
        )

    if not assignable:
        return rows, meta

    fold_members: dict[str, list[str]] = defaultdict(list)
    region_fold: dict[str, str] = {}
    for rid in assignable:
        fold_label = str(cluster_by_id[rid])
        region_fold[rid] = fold_label
        fold_members[fold_label].append(rid)

    fold_strata = None
    if strat_map:
        missing = [rid for rid in assignable if rid not in strat_map]
        if missing:
            raise ValueError(
                f"stratification.csv missing ID {missing[0]!r} "
                "(required when stratification is set)"
            )
        fold_strata = aggregate_stratification_per_fold(dict(fold_members), strat_map)

    fold_to_tt = _assign_folds_to_train_test(
        sorted(fold_members),
        seed=seed,
        fold_strata=fold_strata,
        ratios=ratios,
    )
    meta["train_test_by_fold"] = fold_to_tt
    meta["n_clusters"] = len(fold_members)

    by_region: dict[str, dict[str, str]] = {r["region"]: r for r in rows}
    for rid in assignable:
        fold_label = region_fold[rid]
        by_region[rid] = {
            "region": rid,
            "cluster": fold_label,
            "train_test": fold_to_tt[fold_label],
            "fold": fold_label,
            "additional": json.dumps(
                {"method": "contingency_cc", "cluster": int(fold_label)},
                sort_keys=True,
            ),
        }
    ordered = [by_region[str(rid)] for rid in ids if str(rid) in by_region]
    return ordered, meta


def render_contingency_graph(
    ids: Sequence[str],
    edge_u: Sequence[int],
    edge_v: Sequence[int],
    edge_w: Sequence[int],
    cluster_ids: Sequence[int],
    outdir: Path,
    *,
    title: str = "pangenome contingency graph (connected nodes)",
) -> dict[str, Any]:
    """Render region co-occurrence graph; drop isolates (degree 0)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    n = len(ids)
    degree = [0] * n
    edges: list[tuple[str, str, int]] = []
    for u, v, w in zip(edge_u, edge_v, edge_w):
        ui, vi = int(u), int(v)
        if ui < 0 or vi < 0 or ui >= n or vi >= n or ui == vi:
            continue
        degree[ui] += 1
        degree[vi] += 1
        edges.append((str(ids[ui]), str(ids[vi]), int(w)))

    connected = {str(ids[i]) for i, d in enumerate(degree) if d > 0}
    edges = [e for e in edges if e[0] in connected and e[1] in connected]
    nodes = sorted(connected)
    cid_map = {str(rid): int(cid) for rid, cid in zip(ids, cluster_ids)}

    graph_json = {
        "title": title,
        "n_nodes_connected": len(nodes),
        "n_nodes_total": n,
        "n_edges": len(edges),
        "nodes": [{"id": rid, "cluster": cid_map.get(rid, -1)} for rid in nodes],
        "edges": [{"source": a, "target": b, "weight": w} for a, b, w in edges],
    }

    json_path = outdir / "contingency_graph.json"
    json_path.write_text(json.dumps(graph_json, indent=2) + "\n", encoding="utf-8")

    # DOT for Graphviz consumers
    dot_path = outdir / "contingency_graph.dot"
    lines = ["graph G {", "  graph [overlap=false];", "  node [shape=circle, fontsize=8];"]
    for rid in nodes:
        lines.append(f'  "{rid}" [label="{rid}\\nc{cid_map.get(rid, -1)}"];')
    for a, b, w in edges:
        lines.append(f'  "{a}" -- "{b}" [label="{w}"];')
    lines.append("}")
    dot_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    png_path = outdir / "contingency_graph.png"
    pdf_path = outdir / "contingency_graph.pdf"
    plotted = False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        if nodes and edges:
            # Spring layout without networkx dependency
            idx = {rid: i for i, rid in enumerate(nodes)}
            m = len(nodes)
            rng = np.random.default_rng(42)
            pos = rng.normal(size=(m, 2))
            for _ in range(40):
                disp = np.zeros_like(pos)
                for a, b, w in edges:
                    i, j = idx[a], idx[b]
                    delta = pos[i] - pos[j]
                    dist = max(float(np.linalg.norm(delta)), 1e-3)
                    force = (dist - 1.0) * 0.05
                    direction = delta / dist
                    disp[i] -= direction * force
                    disp[j] += direction * force
                pos += disp
            fig, ax = plt.subplots(figsize=(8, 8))
            for a, b, w in edges:
                i, j = idx[a], idx[b]
                ax.plot(
                    [pos[i, 0], pos[j, 0]],
                    [pos[i, 1], pos[j, 1]],
                    color="#999999",
                    lw=max(0.5, min(3.0, w / 5.0)),
                    zorder=1,
                )
            colors = [cid_map.get(rid, 0) for rid in nodes]
            ax.scatter(pos[:, 0], pos[:, 1], c=colors, cmap="tab20", s=40, zorder=2)
            ax.set_title(title)
            ax.set_axis_off()
            fig.tight_layout()
            fig.savefig(pdf_path, bbox_inches="tight")
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            plotted = True
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No connected nodes", ha="center", va="center")
            ax.set_axis_off()
            fig.savefig(pdf_path, bbox_inches="tight")
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            plotted = True
    except Exception as exc:  # noqa: BLE001
        (outdir / "contingency_graph_plot_error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )

    return {
        "json": str(json_path),
        "dot": str(dot_path),
        "png": str(png_path) if plotted and png_path.is_file() else None,
        "pdf": str(pdf_path) if plotted and pdf_path.is_file() else None,
        "n_nodes_connected": len(nodes),
        "n_edges": len(edges),
    }


def run_pangenome_split_assign(
    *,
    outdir: Path,
    marked: Path,
    parsed: Path | None = None,
    id_csv: Path | None = None,
    fold_csv: Path | None = None,
    stratification_csv: Path | None = None,
    seed: int = 42,
    max_ids: int | None = None,
    ids: list[str] | None = None,
    genomes: Sequence[str] | None = None,
    k: int = DEFAULT_K,
    min_shared: int = DEFAULT_MIN_SHARED,
    ratios: tuple[float, float, float] | None = None,
    plot: bool = True,
    max_edges: int = 100_000,
    materialize_marked: bool = True,
) -> dict[str, Any]:
    """Filter → C++ contingency graph → cluster assign → ``split.csv`` (+ render)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    marked = Path(marked)
    parsed_dir = Path(parsed) if parsed is not None else marked.parent / "PARSED"

    if ids is None:
        kept = filter_ids_to_parsed(
            marked_dir=marked,
            parsed_dir=parsed_dir,
            id_csv=id_csv,
            genomes=genomes,
            max_ids=max_ids,
            seed=seed,
        )
    else:
        kept = intersect_pangenome(marked, parsed_dir, ids=ids)
        if max_ids is not None and len(kept) > int(max_ids):
            rng = random.Random(int(seed))
            kept = list(kept)
            rng.shuffle(kept)
            kept = sorted(kept[: int(max_ids)], key=lambda x: (len(x), x))

    marked_pg = outdir / "MARKED_pangenome"
    if materialize_marked:
        materialize_marked_pangenome(marked, marked_pg, kept, mode="symlink")
        seq_root = marked_pg
    else:
        seq_root = marked

    seq_map = load_fna_directory(seq_root, ids=kept)
    sequences = [seq_map[rid] for rid in kept]
    graph = build_contingency_clusters(
        sequences,
        k=k,
        min_shared=min_shared,
        max_edges=max_edges,
        collect_edges=plot,
    )

    rows, assign_meta = assign_from_contingency(
        kept,
        graph.cluster_ids.tolist(),
        fold_csv=fold_csv,
        stratification_csv=stratification_csv,
        seed=seed,
        ratios=ratios,
    )
    assign_path = write_assignment_table(rows, outdir / "pangenome_assignment.csv")
    split_csv = assignment_rows_to_split_csv(rows, outdir)

    plot_meta: dict[str, Any] | None = None
    if plot:
        plot_meta = render_contingency_graph(
            kept,
            graph.edge_u.tolist(),
            graph.edge_v.tolist(),
            graph.edge_w.tolist(),
            graph.cluster_ids.tolist(),
            outdir / "figures",
        )

    summary = {
        "split_id": SPLIT_ID,
        "seed": seed,
        "marked": str(marked),
        "parsed": str(parsed_dir),
        "marked_pangenome": str(marked_pg) if materialize_marked else None,
        "n_ids": len(kept),
        "k": k,
        "min_shared": min_shared,
        "n_clusters": int(graph.n_clusters),
        "n_edges": int(len(graph.edge_u)),
        "split_csv": str(split_csv),
        "assignment_csv": str(assign_path),
        "assign_meta": assign_meta,
        "plot": plot_meta,
        "genomes": list(genomes) if genomes else None,
    }
    (outdir / "pangenome_split_meta.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary
