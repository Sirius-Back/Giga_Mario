"""Pangenome / cactus-like repeat-graph split strategy.

Caption: ``splits/pangenome.md``. Wired into ``split-predict`` as ``type=pangenome``.

Pipeline:
  1. **Adapt (A2A)** — ``raw`` → ``MARKED_pangenome`` via ``src.pipeline.adapt``
     with the **pangenome window** (may differ from panel ``MARKED`` used for
     LegNet/Caduceus). Invoke ``@preprocess`` / ``adapt`` when that tree is
     missing; do not silently reuse panel ``MARKED`` unless
     ``reuse_panel_marked=True``.
  2. **Filter** — ``MARKED_pangenome`` ∩ ``PARSED`` → ``MARKED_parsed``.
  3. Build C++ **hash-graph**: k-mer hashes as nodes; keep repeat hashes
     (``min_df≥2``); UF on hash nodes via per-sequence co-occurrence.
  4. Per sequence: majority hash-cluster → fold; then train/val/test (+ ZSV).
  5. Render connected nodes only (capped region–region edges for viz).
"""
from __future__ import annotations

import json
import random
import shutil
import warnings
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
    "A2A_ADAPT_HINT",
    "PangenomeAdaptRequiredError",
    "intersect_pangenome",
    "filter_ids_to_parsed",
    "materialize_marked_subset",
    "materialize_marked_pangenome",
    "adapt_pangenome_from_raw",
    "ensure_marked_pangenome",
    "build_contingency_clusters",
    "build_hash_majority_clusters",
    "refine_large_components_by_modularity",
    "save_contingency_graph",
    "load_contingency_graph",
    "render_contingency_graph",
    "plot_pangenome_contingency_from_artifacts",
    "plot_fold_size_distribution",
    "run_pangenome_split_assign",
)

DEFAULT_MAX_FOLD_SIZE = 1000

# Okabe–Ito (colorblind-safe) for train_test categorical panels
_TRAIN_TEST_COLORS = {
    "train": "#0072B2",
    "val": "#E69F00",
    "validation": "#E69F00",
    "test": "#009E73",
    "zsv": "#CC79A7",
    "zeroshotvalidation": "#CC79A7",
}
_FOLD_PALETTE = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#D55E00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)

SPLIT_ID = "pangenome"
DEFAULT_K = 21
DEFAULT_MIN_SHARED = 1
DEFAULT_MIN_DF = 2
DEFAULT_CLUSTER_METHOD = "hash_majority"

A2A_ADAPT_HINT = (
    "A2A: pangenome windows may differ from panel MARKED. "
    "Invoke @preprocess / src.pipeline.adapt with the pangenome "
    "environment+window to write MARKED_pangenome from raw (GTF+FNA+ID.csv), "
    "then re-run type=pangenome. "
    "Only pass reuse_panel_marked=True when the panel MARKED window is "
    "intentionally identical to the pangenome window."
)


class PangenomeAdaptRequiredError(FileNotFoundError):
    """Raised when MARKED_pangenome is missing and adapt inputs are not provided."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or A2A_ADAPT_HINT)


def intersect_pangenome(
    marked_dir: Path,
    parsed_dir: Path,
    ids: Sequence[str] | None = None,
) -> list[str]:
    """Return IDs present in both MARKED_pangenome (``.fa``) and PARSED (``.ext``).

    Filter step before contingency-graph construction (produces ``MARKED_parsed``).
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


def materialize_marked_subset(
    marked_dir: Path,
    out_dir: Path,
    ids: Sequence[str],
    *,
    mode: str = "symlink",
) -> Path:
    """Write a subset directory of MARKED ``*.fa`` (symlinks by default)."""
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


def materialize_marked_pangenome(
    marked_dir: Path,
    out_dir: Path,
    ids: Sequence[str],
    *,
    mode: str = "symlink",
) -> Path:
    """Back-compat alias for ``materialize_marked_subset``."""
    return materialize_marked_subset(marked_dir, out_dir, ids, mode=mode)


def adapt_pangenome_from_raw(
    *,
    outdir: Path,
    gtf_dir: Path,
    fna_dir: Path,
    id_csv: Path,
    environment: str,
    window: dict[str, int],
    genomes: Sequence[str] | None = None,
    max_window: int | None = None,
    seed: int = 42,
) -> dict[str, Path]:
    """A2A adapt: raw GTF+FNA+ID.csv → ``outdir/MARKED_pangenome`` (+ intersect).

    Reuses ``src.pipeline.adapt.run_adapt`` so pangenome windows can differ from
    the panel ``MARKED`` produced for LegNet/Caduceus.
    """
    from src.pipeline.adapt import run_adapt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stage = outdir / "_adapt_pangenome_stage"
    if stage.exists():
        shutil.rmtree(stage)
    result = run_adapt(
        Path(gtf_dir),
        Path(fna_dir),
        outdir=stage,
        id_csv=Path(id_csv),
        environment=environment,
        window=window,
        max_window=max_window,
        genomes=list(genomes) if genomes else None,
        seed=seed,
    )
    marked_src = Path(result["marked_dir"])
    marked_pg = outdir / "MARKED_pangenome"
    if marked_pg.exists() or marked_pg.is_symlink():
        if marked_pg.is_dir() and not marked_pg.is_symlink():
            shutil.rmtree(marked_pg)
        else:
            marked_pg.unlink()
    marked_src.rename(marked_pg)
    intersect_src = Path(result["intersect_csv"])
    intersect_dst = outdir / "intersect_pangenome.csv"
    if intersect_src.is_file():
        intersect_src.replace(intersect_dst)
    if stage.is_dir():
        shutil.rmtree(stage, ignore_errors=True)
    return {"marked_pangenome": marked_pg, "intersect_csv": intersect_dst}


def ensure_marked_pangenome(
    *,
    outdir: Path,
    marked_pangenome: Path | None = None,
    panel_marked: Path | None = None,
    reuse_panel_marked: bool = False,
    gtf_dir: Path | None = None,
    fna_dir: Path | None = None,
    id_csv: Path | None = None,
    environment: str | None = None,
    window: dict[str, int] | None = None,
    genomes: Sequence[str] | None = None,
    max_window: int | None = None,
    seed: int = 42,
) -> tuple[Path, dict[str, Any]]:
    """Resolve ``MARKED_pangenome`` (adapt from raw, reuse, or fail with A2A hint)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    meta: dict[str, Any] = {"source": None}

    if marked_pangenome is not None:
        mp = Path(marked_pangenome)
        if not mp.is_dir():
            raise FileNotFoundError(f"marked_pangenome missing: {mp}")
        if not any(mp.glob("*.fa")):
            raise FileNotFoundError(f"marked_pangenome has no *.fa: {mp}")
        meta["source"] = "marked_pangenome"
        return mp, meta

    default_mp = outdir / "MARKED_pangenome"
    if default_mp.is_dir() and any(default_mp.glob("*.fa")):
        meta["source"] = "outdir/MARKED_pangenome"
        return default_mp, meta

    can_adapt = (
        gtf_dir is not None
        and fna_dir is not None
        and id_csv is not None
        and environment is not None
        and window is not None
    )
    if can_adapt:
        paths = adapt_pangenome_from_raw(
            outdir=outdir,
            gtf_dir=Path(gtf_dir),
            fna_dir=Path(fna_dir),
            id_csv=Path(id_csv),
            environment=str(environment),
            window=window,
            genomes=genomes,
            max_window=max_window,
            seed=seed,
        )
        meta["source"] = "adapt_from_raw"
        meta["window"] = dict(window)
        meta["environment"] = environment
        meta["intersect_csv"] = str(paths["intersect_csv"])
        return paths["marked_pangenome"], meta

    if reuse_panel_marked:
        if panel_marked is None:
            raise ValueError(
                "reuse_panel_marked=True requires panel_marked=… "
                "(existing panel MARKED whose window matches pangenome)"
            )
        pm = Path(panel_marked)
        if not pm.is_dir():
            raise FileNotFoundError(f"panel_marked missing: {pm}")
        warnings.warn(
            "reuse_panel_marked=True: using panel MARKED as MARKED_pangenome. "
            "Only valid when the pangenome window equals the panel adapt window.",
            UserWarning,
            stacklevel=2,
        )
        meta["source"] = "reuse_panel_marked"
        meta["panel_marked"] = str(pm)
        return pm, meta

    raise PangenomeAdaptRequiredError(
        "MARKED_pangenome not found and adapt inputs incomplete "
        "(need gtf_dir, fna_dir, id_csv, environment, window). "
        + A2A_ADAPT_HINT
    )


def build_hash_majority_clusters(
    sequences: list[str],
    *,
    k: int = DEFAULT_K,
    min_df: int = DEFAULT_MIN_DF,
    max_edges: int = 100_000,
    collect_edges: bool = True,
) -> Any:
    """C++ hash-graph clustering: repeat k-mers → UF → majority fold per sequence.

    1. Extract ACGT k-mer hashes per sequence.
    2. Keep hashes with document frequency ≥ ``min_df`` (default 2).
    3. Count hash–hash co-occurrence across sequences; union-find unite pairs
       seen together in ≥2 sequences.
    4. Assign each sequence the majority hash-cluster (ties → smaller id);
       sequences without repeat hashes get a singleton fold.
    """
    from src.splits.pangenome_native import get_native_graph

    return get_native_graph().hash_majority_clusters(
        sequences,
        k=k,
        min_df=min_df,
        max_edges=max_edges,
        collect_edges=collect_edges,
    )


def build_contingency_clusters(
    sequences: list[str],
    *,
    k: int = DEFAULT_K,
    min_shared: int = DEFAULT_MIN_SHARED,
    max_edges: int = 100_000,
    collect_edges: bool = True,
    method: str = DEFAULT_CLUSTER_METHOD,
    min_df: int = DEFAULT_MIN_DF,
) -> Any:
    """Build pangenome fold labels (default: hash-majority).

    ``method='hash_majority'`` (default) — see :func:`build_hash_majority_clusters`.
    ``method='region_contingency'`` — legacy region UF on shared k-mers.
    ``min_shared`` applies only to the legacy path / edge emission threshold
    documentation; hash-majority uses ``min_df``.
    """
    from src.splits.pangenome_native import get_native_graph

    return get_native_graph().contingency_clusters(
        sequences,
        k=k,
        min_shared=min_shared,
        max_edges=max_edges,
        collect_edges=collect_edges,
        method=method,
        min_df=min_df,
    )


def refine_large_components_by_modularity(
    ids: Sequence[str],
    sequences: Sequence[str],
    cluster_ids: Sequence[int],
    *,
    k: int,
    min_shared: int = DEFAULT_MIN_SHARED,
    max_fold_size: int = DEFAULT_MAX_FOLD_SIZE,
    max_edges: int = 2_000_000,
    seed: int = 42,
    resolution: float | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Split oversized contingency CCs with Louvain modularity.

    For each connected component with size > ``max_fold_size``, rebuild a
    k-mer co-occurrence subgraph on that subset and run NetworkX Louvain
    (``nx.community.louvain_communities``). Smaller components are kept.

    Returns renumbered cluster labels (0..n-1) and a refinement meta dict.
    """
    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "networkx is required for modularity refinement of large pangenome "
            "components; install networkx in the active environment"
        ) from exc

    if len(ids) != len(sequences) or len(ids) != len(cluster_ids):
        raise ValueError("ids, sequences, and cluster_ids length mismatch")
    if max_fold_size <= 1:
        raise ValueError(f"max_fold_size must be > 1; got {max_fold_size}")

    labels = [int(c) for c in cluster_ids]
    by_cc: dict[int, list[int]] = defaultdict(list)
    for i, cid in enumerate(labels):
        by_cc[cid].append(i)

    large = {cid: members for cid, members in by_cc.items() if len(members) > max_fold_size}
    meta: dict[str, Any] = {
        "method": "louvain_modularity",
        "max_fold_size": int(max_fold_size),
        "k": int(k),
        "min_shared": int(min_shared),
        "seed": int(seed),
        "n_large_components": len(large),
        "large_component_sizes": {str(cid): len(m) for cid, m in sorted(large.items(), key=lambda kv: -len(kv[1]))[:20]},
        "refined": [],
    }
    if not large:
        # Renumber for stable 0..n_clusters-1
        remap = {old: new for new, old in enumerate(sorted(by_cc))}
        return [remap[c] for c in labels], meta

    next_label = max(labels) + 1 if labels else 0
    for cid, members in sorted(large.items(), key=lambda kv: -len(kv[1])):
        sub_seqs = [sequences[i] for i in members]
        sub_graph = build_contingency_clusters(
            sub_seqs,
            k=int(k),
            min_shared=int(min_shared),
            max_edges=int(max_edges),
            collect_edges=True,
        )
        g = nx.Graph()
        g.add_nodes_from(range(len(members)))
        for u, v, w in zip(
            sub_graph.edge_u.tolist(),
            sub_graph.edge_v.tolist(),
            sub_graph.edge_w.tolist(),
        ):
            ui, vi, wi = int(u), int(v), float(w)
            if ui == vi or ui < 0 or vi < 0:
                continue
            if g.has_edge(ui, vi):
                g[ui][vi]["weight"] += wi
            else:
                g.add_edge(ui, vi, weight=wi)

        res = float(resolution) if resolution is not None else max(
            1.0, len(members) / float(max_fold_size)
        )
        if g.number_of_edges() == 0:
            # No pairwise edges collected — keep original CC (cannot modularize).
            meta["refined"].append(
                {
                    "original_cluster": int(cid),
                    "size": len(members),
                    "n_communities": 1,
                    "resolution": res,
                    "status": "skipped_no_edges",
                }
            )
            continue

        communities = louvain_communities(
            g, weight="weight", resolution=res, seed=int(seed)
        )
        communities = [sorted(c) for c in communities]
        # If still one oversized community, raise resolution once more.
        if len(communities) == 1 and len(members) > max_fold_size:
            res = res * 2.0
            communities = louvain_communities(
                g, weight="weight", resolution=res, seed=int(seed)
            )
            communities = [sorted(c) for c in communities]

        if len(communities) <= 1:
            meta["refined"].append(
                {
                    "original_cluster": int(cid),
                    "size": len(members),
                    "n_communities": 1,
                    "resolution": res,
                    "n_subgraph_edges": int(g.number_of_edges()),
                    "status": "unchanged",
                }
            )
            continue

        for j, comm in enumerate(communities):
            lab = int(cid) if j == 0 else int(next_label)
            if j > 0:
                next_label += 1
            for local_i in comm:
                labels[members[local_i]] = lab

        meta["refined"].append(
            {
                "original_cluster": int(cid),
                "size": len(members),
                "n_communities": len(communities),
                "community_sizes": [len(c) for c in communities],
                "resolution": res,
                "n_subgraph_edges": int(g.number_of_edges()),
                "status": "split",
            }
        )

    # Compact labels to 0..n_clusters-1
    unique = sorted(set(labels))
    remap = {old: new for new, old in enumerate(unique)}
    out = [remap[c] for c in labels]
    meta["n_clusters_after"] = len(unique)
    meta["n_clusters_before"] = len(by_cc)
    return out, meta


def save_contingency_graph(
    outdir: Path,
    ids: Sequence[str],
    graph: Any,
    *,
    k: int,
    min_shared: int = DEFAULT_MIN_SHARED,
    min_df: int = DEFAULT_MIN_DF,
    max_edges: int = 100_000,
    seed: int | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist the contingency / hash-majority graph for later reload.

    Layout under ``{outdir}/graph/``::

      contingency_graph.npz   # cluster_ids, edge_u, edge_v, edge_w (int32)
      ids.txt                 # one region ID per line (order = array index)
      nodes.tsv               # ID|cluster
      edges.tsv               # source|target|weight  (capped co-occurrence edges)
      contingency_graph_meta.json

    Reload with :func:`load_contingency_graph`.
    """
    import numpy as np

    outdir = Path(outdir)
    graph_dir = outdir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    ids_list = [str(x) for x in ids]
    n = len(ids_list)
    cluster_ids = np.asarray(graph.cluster_ids, dtype=np.int32)
    if cluster_ids.shape != (n,):
        raise ValueError(
            f"cluster_ids shape {cluster_ids.shape} != (n_ids={n},)"
        )
    edge_u = np.asarray(graph.edge_u, dtype=np.int32)
    edge_v = np.asarray(graph.edge_v, dtype=np.int32)
    edge_w = np.asarray(graph.edge_w, dtype=np.int32)
    if not (len(edge_u) == len(edge_v) == len(edge_w)):
        raise ValueError("edge_u/v/w length mismatch")

    npz_path = graph_dir / "contingency_graph.npz"
    np.savez_compressed(
        npz_path,
        cluster_ids=cluster_ids,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
    )
    ids_path = graph_dir / "ids.txt"
    ids_path.write_text("\n".join(ids_list) + ("\n" if ids_list else ""), encoding="utf-8")

    nodes_path = graph_dir / "nodes.tsv"
    with nodes_path.open("w", encoding="utf-8") as fh:
        fh.write("ID|cluster\n")
        for rid, cid in zip(ids_list, cluster_ids.tolist()):
            fh.write(f"{rid}|{int(cid)}\n")

    edges_path = graph_dir / "edges.tsv"
    with edges_path.open("w", encoding="utf-8") as fh:
        fh.write("source|target|weight\n")
        for u, v, w in zip(edge_u.tolist(), edge_v.tolist(), edge_w.tolist()):
            ui, vi = int(u), int(v)
            if ui < 0 or vi < 0 or ui >= n or vi >= n:
                continue
            fh.write(f"{ids_list[ui]}|{ids_list[vi]}|{int(w)}\n")

    method = str(getattr(graph, "method", DEFAULT_CLUSTER_METHOD) or DEFAULT_CLUSTER_METHOD)
    if method == "hash_majority":
        clustering = "hash_uf_majority"
        clustering_note = (
            "Graph nodes = ACGT k-mer hashes. Keep repeat hashes with "
            "document frequency ≥ min_df ({int(min_df)}). Union-find on hash "
            "nodes: unite hash pairs that co-occur in ≥2 sequences; each "
            "sequence fold = majority hash-cluster (ties → smaller id). "
            "Sequences without repeat hashes get a singleton fold. "
            "Not Louvain/Leiden/MCL."
        )
        edges_note = (
            "edges.tsv are capped region–region edges weighted by shared "
            "repeat hashes (≤ max_edges) for visualization. Fold labels come "
            "from hash-majority, not from this capped edge list alone."
        )
    else:
        clustering = "union_find_connected_components"
        clustering_note = (
            "Legacy: clusters = connected components via union-find on regions "
            "that share ≥ min_shared ACGT k-mers (bipartite region↔k-mer "
            "contingency). Not modularity (Louvain/Leiden), not Laplacian, not MCL."
        )
        edges_note = (
            "edges.tsv / edge_* arrays are a capped co-occurrence edge list for "
            "visualization and figure rebuild (≤ max_edges). CC labels in "
            "cluster_ids use the full streaming contingency, not only these edges."
        )

    meta = {
        "format": "gigamario_pangenome_contingency_graph_v1",
        "clustering": clustering,
        "method": method,
        "clustering_note": clustering_note,
        "edges_note": edges_note,
        "k": int(k),
        "min_shared": int(min_shared),
        "min_df": int(min_df),
        "max_edges": int(max_edges),
        "seed": seed,
        "n_ids": n,
        "n_clusters": int(getattr(graph, "n_clusters", len(set(cluster_ids.tolist())))),
        "n_edges": int(len(edge_u)),
        "paths": {
            "npz": str(npz_path),
            "ids": str(ids_path),
            "nodes_tsv": str(nodes_path),
            "edges_tsv": str(edges_path),
        },
    }
    if extra_meta:
        meta["extra"] = dict(extra_meta)
    meta_path = graph_dir / "contingency_graph_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    meta["paths"]["meta"] = str(meta_path)
    return meta


def load_contingency_graph(graph_dir: Path) -> dict[str, Any]:
    """Load a graph saved by :func:`save_contingency_graph`.

    Returns dict with ``ids``, ``cluster_ids``, ``edge_u``, ``edge_v``, ``edge_w``,
    ``n_clusters``, ``meta``.
    """
    import numpy as np

    graph_dir = Path(graph_dir)
    if graph_dir.name != "graph" and (graph_dir / "graph").is_dir():
        graph_dir = graph_dir / "graph"
    npz_path = graph_dir / "contingency_graph.npz"
    ids_path = graph_dir / "ids.txt"
    meta_path = graph_dir / "contingency_graph_meta.json"
    if not npz_path.is_file():
        raise FileNotFoundError(f"contingency graph npz missing: {npz_path}")
    if not ids_path.is_file():
        raise FileNotFoundError(f"contingency graph ids missing: {ids_path}")

    ids = [ln.strip() for ln in ids_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    with np.load(npz_path) as data:
        cluster_ids = np.asarray(data["cluster_ids"], dtype=np.int32)
        edge_u = np.asarray(data["edge_u"], dtype=np.int32)
        edge_v = np.asarray(data["edge_v"], dtype=np.int32)
        edge_w = np.asarray(data["edge_w"], dtype=np.int32)
    if len(ids) != len(cluster_ids):
        raise ValueError(
            f"ids.txt length {len(ids)} != cluster_ids length {len(cluster_ids)}"
        )
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n_clusters = int(meta.get("n_clusters") or len(set(cluster_ids.tolist())))
    return {
        "ids": ids,
        "cluster_ids": cluster_ids,
        "edge_u": edge_u,
        "edge_v": edge_v,
        "edge_w": edge_w,
        "n_clusters": n_clusters,
        "meta": meta,
        "graph_dir": str(graph_dir),
    }


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
    method: str = DEFAULT_CLUSTER_METHOD,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Map contingency / majority clusters → assignment rows (ZSV held out)."""
    if len(ids) != len(cluster_ids):
        raise ValueError("ids and cluster_ids length mismatch")
    fold_map = _load_fold_map(fold_csv)
    strat_map = _load_strat_map(stratification_csv)
    method_used = str(method or DEFAULT_CLUSTER_METHOD)

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
        "method_used": method_used,
    }
    for rid in zsv_ids:
        rows.append(
            {
                "region": rid,
                "cluster": "zsv",
                "train_test": "zsv",
                "fold": "zsv",
                "additional": json.dumps({"method": method_used}),
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
                {"method": method_used, "cluster": int(fold_label)},
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
    train_test: Sequence[str] | None = None,
    fold: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Render region co-occurrence graph; drop isolates (degree 0).

    Always writes ``contingency_graph.{json,dot}`` and a cluster-coloured
    scatter. When ``train_test`` / ``fold`` are provided (aligned with
    ``ids``), also writes a two-panel figure coloured by train/test/val/zsv
    and by fold (connected nodes only).
    """
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
    tt_map: dict[str, str] = {}
    fold_map: dict[str, str] = {}
    if train_test is not None:
        if len(train_test) != n:
            raise ValueError(
                f"train_test length {len(train_test)} != n_ids {n}"
            )
        tt_map = {str(rid): str(lab) for rid, lab in zip(ids, train_test)}
    if fold is not None:
        if len(fold) != n:
            raise ValueError(f"fold length {len(fold)} != n_ids {n}")
        fold_map = {str(rid): str(lab) for rid, lab in zip(ids, fold)}

    graph_json = {
        "title": title,
        "n_nodes_connected": len(nodes),
        "n_nodes_total": n,
        "n_edges": len(edges),
        "nodes": [
            {
                "id": rid,
                "cluster": cid_map.get(rid, -1),
                **({"train_test": tt_map[rid]} if rid in tt_map else {}),
                **({"fold": fold_map[rid]} if rid in fold_map else {}),
            }
            for rid in nodes
        ],
        "edges": [{"source": a, "target": b, "weight": w} for a, b, w in edges],
    }

    json_path = outdir / "contingency_graph.json"
    json_path.write_text(json.dumps(graph_json, indent=2) + "\n", encoding="utf-8")

    dot_path = outdir / "contingency_graph.dot"
    lines = [
        "graph G {",
        "  graph [overlap=false];",
        "  node [shape=circle, fontsize=8];",
    ]
    for rid in nodes:
        lines.append(f'  "{rid}" [label="{rid}\\nc{cid_map.get(rid, -1)}"];')
    for a, b, w in edges:
        lines.append(f'  "{a}" -- "{b}" [label="{w}"];')
    lines.append("}")
    dot_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    png_path = outdir / "contingency_graph.png"
    pdf_path = outdir / "contingency_graph.pdf"
    split_png = outdir / "Figure_pangenome_contingency_fold_train_test.png"
    split_pdf = outdir / "Figure_pangenome_contingency_fold_train_test.pdf"
    split_svg = outdir / "Figure_pangenome_contingency_fold_train_test.svg"
    plotted = False
    split_plotted = False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        pos: Any = None
        idx: dict[str, int] = {}
        if nodes and edges:
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
                    alpha=0.35,
                )
            colors = [cid_map.get(rid, 0) for rid in nodes]
            ax.scatter(pos[:, 0], pos[:, 1], c=colors, cmap="tab20", s=18, zorder=2)
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

        if pos is not None and (tt_map or fold_map):
            n_panels = int(bool(tt_map)) + int(bool(fold_map))
            fig, axes = plt.subplots(
                1, n_panels, figsize=(7.2 * n_panels, 6.5), squeeze=False
            )
            ax_i = 0

            def _draw_edges(ax: Any) -> None:
                for a, b, w in edges:
                    i, j = idx[a], idx[b]
                    ax.plot(
                        [pos[i, 0], pos[j, 0]],
                        [pos[i, 1], pos[j, 1]],
                        color="#BBBBBB",
                        lw=max(0.4, min(2.5, w / 6.0)),
                        zorder=1,
                        alpha=0.3,
                    )

            if tt_map:
                ax = axes[0, ax_i]
                ax_i += 1
                _draw_edges(ax)
                labels = [tt_map.get(rid, "other") for rid in nodes]
                unique = sorted(set(labels), key=lambda x: (x not in _TRAIN_TEST_COLORS, x))
                color_of = {
                    lab: _TRAIN_TEST_COLORS.get(lab.lower(), "#999999") for lab in unique
                }
                for lab in unique:
                    mask = [lb == lab for lb in labels]
                    pts = pos[[i for i, msk in enumerate(mask) if msk]]
                    if len(pts) == 0:
                        continue
                    ax.scatter(
                        pts[:, 0],
                        pts[:, 1],
                        c=color_of[lab],
                        s=16,
                        zorder=2,
                        label=f"{lab} (n={sum(mask)})",
                        edgecolors="none",
                    )
                ax.set_title("Connected nodes by train / test / val / zsv")
                ax.set_axis_off()
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    frameon=False,
                    fontsize=8,
                )

            if fold_map:
                ax = axes[0, ax_i]
                _draw_edges(ax)
                labels = [fold_map.get(rid, "other") for rid in nodes]
                # Stable palette index from fold string; legend = top folds by count.
                from collections import Counter

                counts = Counter(labels)
                top = [f for f, _ in counts.most_common(12)]
                top_set = set(top)
                fold_color = {
                    f: _FOLD_PALETTE[i % len(_FOLD_PALETTE)] for i, f in enumerate(top)
                }
                other_c = "#CCCCCC"
                # Draw "other" first so top folds sit on top.
                for lab, pts_idx in (
                    ("__other__", [i for i, lb in enumerate(labels) if lb not in top_set]),
                    *[
                        (f, [i for i, lb in enumerate(labels) if lb == f])
                        for f in top
                    ],
                ):
                    if not pts_idx:
                        continue
                    pts = pos[pts_idx]
                    if lab == "__other__":
                        ax.scatter(
                            pts[:, 0],
                            pts[:, 1],
                            c=other_c,
                            s=10,
                            zorder=2,
                            label=f"other folds (n={len(pts_idx)})",
                            edgecolors="none",
                        )
                    else:
                        ax.scatter(
                            pts[:, 0],
                            pts[:, 1],
                            c=fold_color[lab],
                            s=16,
                            zorder=3,
                            label=f"fold {lab} (n={counts[lab]})",
                            edgecolors="none",
                        )
                ax.set_title("Connected nodes by fold (top 12 + other)")
                ax.set_axis_off()
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    frameon=False,
                    fontsize=7,
                )

            fig.suptitle(
                "Pangenome contingency graph (connected nodes only)",
                fontsize=11,
                y=1.02,
            )
            fig.tight_layout()
            fig.savefig(split_pdf, bbox_inches="tight")
            fig.savefig(split_png, dpi=300, bbox_inches="tight")
            fig.savefig(split_svg, bbox_inches="tight")
            plt.close(fig)
            split_plotted = True
    except Exception as exc:  # noqa: BLE001
        (outdir / "contingency_graph_plot_error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )

    return {
        "json": str(json_path),
        "dot": str(dot_path),
        "png": str(png_path) if plotted and png_path.is_file() else None,
        "pdf": str(pdf_path) if plotted and pdf_path.is_file() else None,
        "split_png": str(split_png) if split_plotted and split_png.is_file() else None,
        "split_pdf": str(split_pdf) if split_plotted and split_pdf.is_file() else None,
        "split_svg": str(split_svg) if split_plotted and split_svg.is_file() else None,
        "n_nodes_connected": len(nodes),
        "n_edges": len(edges),
    }


def plot_fold_size_distribution(
    assignment_or_split_csv: Path,
    outdir: Path,
    *,
    title: str | None = None,
    exclude_zsv: bool = True,
    bins: int = 60,
) -> dict[str, Any]:
    """Histogram of contingency fold sizes on a log10(size + 1) scale.

    Reads ``pangenome_assignment.csv`` (``fold`` / ``cluster``) or ``split.csv``
    (``fold``). Writes publication PDF/PNG/SVG under ``outdir``.
    """
    from collections import Counter

    import numpy as np

    assignment_or_split_csv = Path(assignment_or_split_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(assignment_or_split_csv)
    if not rows:
        raise ValueError(f"empty table: {assignment_or_split_csv}")
    fold_key = "fold" if "fold" in rows[0] else "cluster"
    if fold_key not in rows[0]:
        raise ValueError(
            f"need fold or cluster column in {assignment_or_split_csv}; "
            f"have {list(rows[0])}"
        )
    counts: Counter[str] = Counter()
    for row in rows:
        lab = str(row[fold_key]).strip()
        if exclude_zsv and (lab.lower() in {"zsv", "zeroshotvalidation"} or is_zsv_fold(lab)):
            continue
        counts[lab] += 1
    if not counts:
        raise ValueError("no folds left after ZSV filter")

    sizes = np.asarray(sorted(counts.values()), dtype=np.float64)
    log_sizes = np.log10(sizes + 1.0)
    n_folds = int(len(sizes))
    n_regions = int(sizes.sum())
    n_singletons = int((sizes == 1).sum())

    stats = {
        "n_folds": n_folds,
        "n_regions": n_regions,
        "n_singletons": n_singletons,
        "singleton_fraction": float(n_singletons / n_folds),
        "size_min": int(sizes.min()),
        "size_median": float(np.median(sizes)),
        "size_mean": float(sizes.mean()),
        "size_max": int(sizes.max()),
        "exclude_zsv": exclude_zsv,
        "source": str(assignment_or_split_csv),
    }

    pdf_path = outdir / "Figure_pangenome_fold_size_log10.pdf"
    png_path = outdir / "Figure_pangenome_fold_size_log10.png"
    svg_path = outdir / "Figure_pangenome_fold_size_log10.svg"
    csv_path = outdir / "fold_size_distribution_stats.json"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(
        log_sizes,
        bins=bins,
        color="#0072B2",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xlabel(r"$\log_{10}(\mathrm{fold\ size} + 1)$")
    ax.set_ylabel("Number of folds")
    ax.set_title(
        title
        or (
            f"Pangenome contingency fold sizes "
            f"(n_folds={n_folds:,}; n_regions={n_regions:,})"
        )
    )
    # Annotate key quantiles on the log10(size+1) axis
    for label, val in (
        ("median", float(np.median(sizes))),
        ("max", float(sizes.max())),
    ):
        x = float(np.log10(val + 1.0))
        ax.axvline(x, color="#D55E00", ls="--", lw=1.0, alpha=0.85)
        ax.text(
            x,
            ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] else 1.0,
            f"{label}={int(val)}",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color="#D55E00",
        )
    note = (
        f"singletons={n_singletons:,} ({100 * n_singletons / n_folds:.1f}% of folds)"
        + ("; ZSV excluded" if exclude_zsv else "")
    )
    ax.text(
        0.98,
        0.98,
        note,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#333333",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    csv_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    stats["pdf"] = str(pdf_path)
    stats["png"] = str(png_path)
    stats["svg"] = str(svg_path)
    stats["stats_json"] = str(csv_path)
    return stats


def plot_pangenome_contingency_from_artifacts(
    *,
    marked_parsed: Path,
    split_csv: Path,
    outdir: Path,
    k: int = DEFAULT_K,
    min_shared: int = DEFAULT_MIN_SHARED,
    max_edges: int = 100_000,
    seed: int = 42,
    prefer_saved_graph: bool = True,
) -> dict[str, Any]:
    """Rebuild contingency figure from saved ``graph/`` or recompute from MARKED.

    Prefer ``{outdir}/graph/contingency_graph.npz`` when present so figures can
    be regenerated without re-streaming sequences. Falls back to C++ rebuild
    from ``MARKED_parsed`` + labels from ``split.csv``.
    """
    marked_parsed = Path(marked_parsed)
    split_csv = Path(split_csv)
    outdir = Path(outdir)
    if not split_csv.is_file():
        raise FileNotFoundError(f"split.csv missing: {split_csv}")

    rows = read_csv(split_csv)
    if not rows or "ID" not in rows[0]:
        raise ValueError(f"split.csv missing ID column: {split_csv}")
    split_tt = {r["ID"].strip(): r.get("train_test", "").strip() for r in rows}
    split_fold = {r["ID"].strip(): r.get("fold", "").strip() for r in rows}

    loaded: dict[str, Any] | None = None
    graph_dir = outdir / "graph"
    if prefer_saved_graph and (graph_dir / "contingency_graph.npz").is_file():
        loaded = load_contingency_graph(graph_dir)
        ids = list(loaded["ids"])
        cluster_ids = loaded["cluster_ids"].tolist()
        edge_u = loaded["edge_u"].tolist()
        edge_v = loaded["edge_v"].tolist()
        edge_w = loaded["edge_w"].tolist()
        n_clusters = int(loaded["n_clusters"])
        source = "saved_graph"
        meta_k = int((loaded.get("meta") or {}).get("k", k))
        meta_min = int((loaded.get("meta") or {}).get("min_shared", min_shared))
    else:
        if not marked_parsed.is_dir():
            raise FileNotFoundError(f"MARKED_parsed missing: {marked_parsed}")
        ids = [r["ID"].strip() for r in rows]
        seq_map = load_fna_directory(marked_parsed, ids=ids)
        missing = [rid for rid in ids if rid not in seq_map]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} split IDs missing from MARKED_parsed "
                f"(e.g. {missing[:5]})"
            )
        sequences = [seq_map[rid] for rid in ids]
        graph = build_contingency_clusters(
            sequences,
            k=int(k),
            min_shared=int(min_shared),
            max_edges=int(max_edges),
            collect_edges=True,
        )
        save_contingency_graph(
            outdir,
            ids,
            graph,
            k=int(k),
            min_shared=int(min_shared),
            max_edges=int(max_edges),
            seed=int(seed),
            extra_meta={"source": "recomputed_for_plot"},
        )
        cluster_ids = graph.cluster_ids.tolist()
        edge_u = graph.edge_u.tolist()
        edge_v = graph.edge_v.tolist()
        edge_w = graph.edge_w.tolist()
        n_clusters = int(graph.n_clusters)
        source = "recomputed"
        meta_k, meta_min = int(k), int(min_shared)

    train_test = [split_tt.get(rid, "") for rid in ids]
    fold = [split_fold.get(rid, "") for rid in ids]
    figs = outdir / "figures"
    plot_meta = render_contingency_graph(
        ids,
        edge_u,
        edge_v,
        edge_w,
        cluster_ids,
        figs,
        train_test=train_test,
        fold=fold,
    )
    summary = {
        "marked_parsed": str(marked_parsed),
        "split_csv": str(split_csv),
        "graph_source": source,
        "k": meta_k,
        "min_shared": meta_min,
        "max_edges": int(max_edges),
        "seed": int(seed),
        "n_ids": len(ids),
        "n_clusters": n_clusters,
        "n_edges": len(edge_u),
        "graph_dir": str(outdir / "graph"),
        "plot": plot_meta,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "pangenome_graph_plot_meta.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary


def run_pangenome_split_assign(
    *,
    outdir: Path,
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
    min_df: int = DEFAULT_MIN_DF,
    cluster_method: str = DEFAULT_CLUSTER_METHOD,
    ratios: tuple[float, float, float] | None = None,
    plot: bool = True,
    max_edges: int = 100_000,
    save_graph: bool = True,
    modularity_refine: bool = False,
    max_fold_size: int = DEFAULT_MAX_FOLD_SIZE,
    modularity_max_edges: int = 2_000_000,
    # MARKED_pangenome resolution (A2A adapt vs reuse)
    marked_pangenome: Path | None = None,
    panel_marked: Path | None = None,
    marked: Path | None = None,  # alias for panel_marked
    reuse_panel_marked: bool = False,
    gtf_dir: Path | None = None,
    fna_dir: Path | None = None,
    environment: str | None = None,
    window: dict[str, int] | None = None,
    max_window: int | None = None,
) -> dict[str, Any]:
    """Adapt/resolve MARKED_pangenome → MARKED_parsed → contingency → ``split.csv``."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    panel = panel_marked or marked

    marked_pg, source_meta = ensure_marked_pangenome(
        outdir=outdir,
        marked_pangenome=marked_pangenome,
        panel_marked=panel,
        reuse_panel_marked=reuse_panel_marked,
        gtf_dir=gtf_dir,
        fna_dir=fna_dir,
        id_csv=id_csv,
        environment=environment,
        window=window,
        genomes=genomes,
        max_window=max_window,
        seed=seed,
    )

    parsed_dir: Path
    if parsed is not None:
        parsed_dir = Path(parsed)
    elif panel is not None and (Path(panel).parent / "PARSED").is_dir():
        parsed_dir = Path(panel).parent / "PARSED"
    else:
        parsed_dir = marked_pg.parent / "PARSED"

    if ids is None:
        kept = filter_ids_to_parsed(
            marked_dir=marked_pg,
            parsed_dir=parsed_dir,
            id_csv=id_csv,
            genomes=genomes,
            max_ids=max_ids,
            seed=seed,
        )
    else:
        kept = intersect_pangenome(marked_pg, parsed_dir, ids=ids)
        if max_ids is not None and len(kept) > int(max_ids):
            rng = random.Random(int(seed))
            kept = list(kept)
            rng.shuffle(kept)
            kept = sorted(kept[: int(max_ids)], key=lambda x: (len(x), x))

    marked_parsed = materialize_marked_subset(
        marked_pg, outdir / "MARKED_parsed", kept, mode="symlink"
    )
    # If we adapted into outdir and source was panel reuse, also expose a
    # local MARKED_pangenome pointer for consumers (symlink tree of kept IDs).
    if source_meta.get("source") == "reuse_panel_marked":
        materialize_marked_subset(
            marked_pg, outdir / "MARKED_pangenome", kept, mode="symlink"
        )
        marked_pg_out = outdir / "MARKED_pangenome"
    else:
        marked_pg_out = marked_pg

    seq_map = load_fna_directory(marked_parsed, ids=kept)
    sequences = [seq_map[rid] for rid in kept]
    # Always collect edges when persisting / modularizing / plotting.
    collect_edges = bool(save_graph) or bool(plot) or bool(modularity_refine)
    graph = build_contingency_clusters(
        sequences,
        k=k,
        min_shared=min_shared,
        max_edges=max_edges,
        collect_edges=collect_edges,
        method=cluster_method,
        min_df=min_df,
    )

    import numpy as np
    from dataclasses import replace

    cluster_ids = [int(c) for c in graph.cluster_ids.tolist()]
    modularity_meta: dict[str, Any] | None = None
    if modularity_refine:
        cluster_ids, modularity_meta = refine_large_components_by_modularity(
            kept,
            sequences,
            cluster_ids,
            k=int(k),
            min_shared=int(min_shared),
            max_fold_size=int(max_fold_size),
            max_edges=int(modularity_max_edges),
            seed=int(seed),
        )
        graph = replace(
            graph,
            cluster_ids=np.asarray(cluster_ids, dtype=np.int32),
            n_clusters=int(len(set(cluster_ids))),
        )

    method_used = str(getattr(graph, "method", cluster_method) or cluster_method)
    graph_meta: dict[str, Any] | None = None
    if save_graph:
        graph_meta = save_contingency_graph(
            outdir,
            kept,
            graph,
            k=int(k),
            min_shared=int(min_shared),
            min_df=int(min_df),
            max_edges=int(max_edges),
            seed=int(seed),
            extra_meta={
                "marked_parsed": str(marked_parsed),
                "modularity_refine": bool(modularity_refine),
                "modularity": modularity_meta,
                "cluster_method": method_used,
            },
        )

    rows, assign_meta = assign_from_contingency(
        kept,
        cluster_ids,
        fold_csv=fold_csv,
        stratification_csv=stratification_csv,
        seed=seed,
        ratios=ratios,
        method=method_used,
    )
    if modularity_meta is not None:
        assign_meta = {**assign_meta, "modularity_refine": modularity_meta}
    assign_path = write_assignment_table(rows, outdir / "pangenome_assignment.csv")
    split_csv = assignment_rows_to_split_csv(rows, outdir)

    plot_meta: dict[str, Any] | None = None
    if plot:
        by_id = {
            str(r.get("region") or r.get("ID") or ""): r
            for r in rows
            if (r.get("region") or r.get("ID"))
        }
        tt_aligned = [
            str(by_id[str(rid)]["train_test"]) if str(rid) in by_id else ""
            for rid in kept
        ]
        fold_aligned = [
            str(by_id[str(rid)]["fold"]) if str(rid) in by_id else ""
            for rid in kept
        ]
        plot_meta = render_contingency_graph(
            kept,
            graph.edge_u.tolist(),
            graph.edge_v.tolist(),
            graph.edge_w.tolist(),
            cluster_ids,
            outdir / "figures",
            train_test=tt_aligned,
            fold=fold_aligned,
        )

    summary = {
        "split_id": SPLIT_ID,
        "seed": seed,
        "marked_pangenome": str(marked_pg_out),
        "marked_parsed": str(marked_parsed),
        "marked_source": source_meta,
        "parsed": str(parsed_dir),
        "panel_marked": str(panel) if panel else None,
        "n_ids": len(kept),
        "k": k,
        "min_shared": min_shared,
        "min_df": min_df,
        "cluster_method": method_used,
        "n_clusters": int(graph.n_clusters),
        "n_edges": int(len(graph.edge_u)),
        "split_csv": str(split_csv),
        "assignment_csv": str(assign_path),
        "assign_meta": assign_meta,
        "modularity_refine": bool(modularity_refine),
        "modularity": modularity_meta,
        "max_fold_size": int(max_fold_size) if modularity_refine else None,
        "graph": graph_meta,
        "graph_dir": str(outdir / "graph") if graph_meta else None,
        "plot": plot_meta,
        "genomes": list(genomes) if genomes else None,
        "a2a_adapt": A2A_ADAPT_HINT if source_meta.get("source") != "adapt_from_raw" else None,
    }
    (outdir / "pangenome_split_meta.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary
