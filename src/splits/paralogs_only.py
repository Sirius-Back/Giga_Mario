"""Orthogroup-representative train / paralog-remainder split.

Caption: ``splits/paralogs_only.md``.
Wired into ``split-predict`` as ``type=paralogs_only``.

Algorithm (C++ native):
  - Orthogroups = connected components on ortholog edges only
  - One representative gene per OG → train (max paralog_degree, seeded ties)
  - Remainder (other OG members + unmapped panel IDs) → 50/50 test/val
  - Unmapped panel IDs never enter train (Locked)
"""
from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any, Sequence

from src.pipeline.common import SPLIT_CSV_COLUMNS, ensure_dir, read_csv, write_csv
from src.pipeline.generate_fold import is_zsv_fold, normalize_fold_label
from src.splits.paralogs_only_native import ensure_built, get_native

__all__ = (
    "SPLIT_ID",
    "DEFAULT_HOMOLOGY_EDGES",
    "DEFAULT_HOMOLOGY_NODES",
    "run_paralogs_only_split_assign",
)

SPLIT_ID = "paralogs_only"
DEFAULT_HOMOLOGY_EDGES = Path("mag/homology_graph/edges.tsv.gz")
DEFAULT_HOMOLOGY_NODES = Path("mag/homology_graph/maps/nodes_extract.tsv")


def _load_ids(path: Path) -> list[str]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"id_csv is empty: {path}")
    if "ID" not in rows[0]:
        raise ValueError(f"id_csv missing column ['ID']; have {list(rows[0])}")
    ids: list[str] = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        identifier = row["ID"].strip()
        if not identifier:
            raise ValueError(f"id_csv has blank ID at row {row_number}")
        if identifier in seen:
            raise ValueError(f"id_csv has duplicate ID {identifier!r}")
        seen.add(identifier)
        ids.append(identifier)
    return ids


def _load_fold_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    rows = read_csv(Path(path))
    if not rows:
        raise ValueError(f"fold.csv is empty: {path}")
    if "ID" not in rows[0] or "fold" not in rows[0]:
        raise ValueError(f"fold.csv missing ID/fold; have {list(rows[0])}")
    out: dict[str, str] = {}
    for row in rows:
        rid = row["ID"].strip()
        if not rid:
            continue
        out[rid] = normalize_fold_label(row["fold"])
    return out


def _read_assignment_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def run_paralogs_only_split_assign(
    *,
    outdir: Path | str,
    id_csv: Path | str,
    homology_edges: Path | str | None = None,
    homology_nodes: Path | str | None = None,
    fold_csv: Path | str | None = None,
    seed: int = 42,
    max_ids: int | None = None,
) -> dict[str, Any]:
    """Run native assignment and write ``{outdir}/split.csv``.

    Returns a summary dict including ``split_csv`` path and native meta.
    """
    outdir = ensure_dir(Path(outdir))
    id_csv_p = Path(id_csv)
    if not id_csv_p.is_file():
        raise FileNotFoundError(f"id_csv missing: {id_csv_p}")

    edges = Path(homology_edges) if homology_edges else DEFAULT_HOMOLOGY_EDGES
    nodes = Path(homology_nodes) if homology_nodes else DEFAULT_HOMOLOGY_NODES
    if not edges.is_file():
        raise FileNotFoundError(
            f"homology edges missing: {edges} "
            "(rebuild via python -m src.run.homology_graph.build_mammals11)"
        )
    if not nodes.is_file():
        raise FileNotFoundError(f"homology nodes map missing: {nodes}")

    if fold_csv is None:
        warnings.warn("Warning: folds are not included", UserWarning, stacklevel=2)

    ids = _load_ids(id_csv_p)
    if max_ids is not None and max_ids > 0:
        ids = ids[: int(max_ids)]

    fold_map = _load_fold_map(Path(fold_csv) if fold_csv else None)
    id_set = set(ids)
    unknown = set(fold_map) - id_set
    if unknown:
        raise ValueError(
            f"fold.csv contains ID absent from id_csv: {sorted(unknown)[0]!r}"
        )

    zsv_ids: list[str] = []
    assignable: list[str] = []
    for i in ids:
        raw = fold_map.get(i, "0")
        lab = normalize_fold_label(raw)
        if is_zsv_fold(lab):
            zsv_ids.append(i)
        else:
            assignable.append(i)

    if not assignable:
        raise ValueError("no assignable IDs after ZSV holdout")

    ensure_built()
    native = get_native()
    graph_dir = ensure_dir(outdir / "graph")
    assignment_tsv = graph_dir / "assignment.tsv"
    meta_json = graph_dir / "paralogs_only_meta.json"
    native.assign(
        edges_path=edges,
        nodes_path=nodes,
        panel_ids=assignable,
        seed=int(seed),
        out_assignment_path=assignment_tsv,
        out_meta_json_path=meta_json,
    )

    assigned = _read_assignment_tsv(assignment_tsv)
    by_id = {row["ID"]: row for row in assigned}
    missing = [i for i in assignable if i not in by_id]
    if missing:
        raise RuntimeError(
            f"native assignment missing {len(missing)} IDs; e.g. {missing[0]!r}"
        )

    rows: list[dict[str, Any]] = []
    zsv_set = set(zsv_ids)
    for i in ids:
        if i in zsv_set:
            rows.append({"ID": i, "train_test": "zsv", "fold": "zsv"})
        else:
            row = by_id[i]
            rows.append(
                {
                    "ID": i,
                    "train_test": row["train_test"],
                    "fold": row["fold"],
                }
            )

    split_csv = outdir / "split.csv"
    write_csv(split_csv, rows, SPLIT_CSV_COLUMNS)

    meta: dict[str, Any] = {}
    if meta_json.is_file():
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
    counts = {
        "train": sum(1 for r in rows if r["train_test"] == "train"),
        "val": sum(1 for r in rows if r["train_test"] == "val"),
        "test": sum(1 for r in rows if r["train_test"] == "test"),
        "zsv": sum(1 for r in rows if r["train_test"] == "zsv"),
    }
    # Locked: no unmapped ID in train
    unmapped_train = [
        r["ID"]
        for r in rows
        if r["train_test"] == "train" and r["fold"] == "unmapped"
    ]
    if unmapped_train:
        raise RuntimeError(
            "unmapped IDs must not be train; "
            f"found {len(unmapped_train)} e.g. {unmapped_train[0]!r}"
        )

    summary = {
        "split_csv": str(split_csv),
        "assignment_tsv": str(assignment_tsv),
        "meta_json": str(meta_json),
        "counts": counts,
        "native_meta": meta,
        "seed": int(seed),
        "homology_edges": str(edges),
        "homology_nodes": str(nodes),
        "n_unmapped_in_split": sum(1 for r in rows if r["fold"] == "unmapped"),
    }
    (graph_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
