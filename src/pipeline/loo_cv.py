"""Leave-one-out CV helpers over reused homology clusters.

Maps pangenome ``fold`` (cluster) IDs into ``n_cv`` equal-mass CV folds, then
for each holdout round builds a train/val/test ``split.csv`` with ≈3:1:1 mass
(train = n_cv-2 folds, val = 1, test = 1) without reshuffling region IDs.
"""
from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.pipeline.common import SPLIT_CSV_COLUMNS, ensure_dir, read_csv, write_csv
from src.pipeline.generate_fold import is_zsv_fold

DEFAULT_N_CV = 5
CV_MASTER_COLUMNS = [
    "ID",
    "homology_fold",
    "cv_fold",
    "train_test",
    "cluster",
]


def load_pangenome_assignment(path: Path) -> list[dict[str, str]]:
    rows = read_csv(Path(path))
    if not rows:
        raise ValueError(f"empty assignment: {path}")
    # Accept region or ID as the region key.
    if "region" not in rows[0] and "ID" not in rows[0]:
        raise ValueError(
            f"assignment missing region/ID columns: {list(rows[0])}"
        )
    return rows


def _region_id(row: dict[str, str]) -> str:
    return str(row.get("region") or row.get("ID") or "").strip()


def _homology_fold(row: dict[str, str]) -> str:
    # Prefer fold; fall back to cluster (same grain for pangenome assignment).
    raw = str(row.get("fold") or row.get("cluster") or "").strip()
    return raw


def assign_clusters_to_cv_folds(
    rows: Sequence[dict[str, str]],
    *,
    n_cv: int = DEFAULT_N_CV,
    seed: int = 42,
) -> dict[str, int]:
    """Greedy largest-first packing of homology clusters into ``n_cv`` folds.

    Returns mapping ``homology_fold_id -> cv_fold_index`` (0..n_cv-1).
    ZSV clusters are omitted.
    """
    if n_cv < 3:
        raise ValueError(f"n_cv must be >= 3 for train/val/test LOO; got {n_cv}")
    sizes: Counter[str] = Counter()
    for row in rows:
        rid = _region_id(row)
        if not rid:
            continue
        hf = _homology_fold(row)
        if not hf or is_zsv_fold(hf) or is_zsv_fold(row.get("train_test", "")):
            continue
        sizes[hf] += 1
    if len(sizes) < n_cv:
        raise ValueError(
            f"need >= {n_cv} non-ZSV homology folds; got {len(sizes)}"
        )
    rng = random.Random(seed)
    # Stable order: size desc, then fold id; shuffle within equal sizes via rng key.
    items = list(sizes.items())
    rng.shuffle(items)
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    cv_mass = [0] * n_cv
    out: dict[str, int] = {}
    for hf, sz in items:
        j = min(range(n_cv), key=lambda i: (cv_mass[i], i))
        out[hf] = int(j)
        cv_mass[j] += int(sz)
    return out


def build_cv_master_rows(
    rows: Sequence[dict[str, str]],
    cluster_to_cv: dict[str, int],
) -> list[dict[str, str]]:
    """One master row per region with cv_fold index (or zsv)."""
    out: list[dict[str, str]] = []
    for row in rows:
        rid = _region_id(row)
        if not rid:
            continue
        hf = _homology_fold(row)
        tt = str(row.get("train_test") or "").strip().lower()
        if is_zsv_fold(hf) or is_zsv_fold(tt):
            out.append(
                {
                    "ID": rid,
                    "homology_fold": "zsv",
                    "cv_fold": "zsv",
                    "train_test": "zsv",
                    "cluster": str(row.get("cluster") or "zsv"),
                }
            )
            continue
        if hf not in cluster_to_cv:
            raise KeyError(f"homology fold {hf!r} missing from CV map (ID={rid})")
        cv = int(cluster_to_cv[hf])
        out.append(
            {
                "ID": rid,
                "homology_fold": hf,
                "cv_fold": str(cv),
                "train_test": "",  # filled per LOO round
                "cluster": str(row.get("cluster") or hf),
            }
        )
    return out


def split_rows_for_loo_round(
    master: Sequence[dict[str, str]],
    *,
    holdout: int,
    n_cv: int = DEFAULT_N_CV,
) -> list[dict[str, str]]:
    """Build ``split.csv`` rows for LOO round ``holdout``.

    test = cv_fold == holdout
    val = cv_fold == (holdout + 1) % n_cv
    train = remaining non-zsv
    """
    if not (0 <= holdout < n_cv):
        raise ValueError(f"holdout must be in [0, {n_cv}); got {holdout}")
    val_idx = (holdout + 1) % n_cv
    out: list[dict[str, str]] = []
    for row in master:
        rid = str(row["ID"]).strip()
        hf = str(row["homology_fold"])
        cv_raw = str(row["cv_fold"])
        if is_zsv_fold(cv_raw) or is_zsv_fold(row.get("train_test", "")):
            out.append({"ID": rid, "train_test": "zsv", "fold": "zsv"})
            continue
        cv = int(cv_raw)
        if cv == holdout:
            tt = "test"
        elif cv == val_idx:
            tt = "val"
        else:
            tt = "train"
        out.append({"ID": rid, "train_test": tt, "fold": hf})
    return out


def summarize_split_rows(rows: Sequence[dict[str, str]]) -> dict[str, Any]:
    counts = Counter(str(r["train_test"]).strip().lower() for r in rows)
    return {
        "n": len(rows),
        "train": int(counts.get("train", 0)),
        "test": int(counts.get("test", 0)),
        "val": int(counts.get("val", 0))
        + int(counts.get("validation", 0)),
        "zsv": int(counts.get("zsv", 0)),
    }


def stage_loo_cv_from_assignment(
    *,
    source_assignment: Path,
    out_root: Path,
    panel_root: Path,
    source_unif: Path | None = None,
    n_cv: int = DEFAULT_N_CV,
    seed: int = 42,
    copy_graph: bool = True,
) -> dict[str, Any]:
    """Write CV master + per-round split.csv; materialize each fold SPLIT (+ LegNet TSV)."""
    from src.pipeline.legnet_input import build_legnet_tsv
    from src.pipeline.split import run_split

    source_assignment = Path(source_assignment)
    out_root = ensure_dir(Path(out_root))
    panel_root = Path(panel_root)

    rows = load_pangenome_assignment(source_assignment)
    cluster_to_cv = assign_clusters_to_cv_folds(rows, n_cv=n_cv, seed=seed)
    master = build_cv_master_rows(rows, cluster_to_cv)

    # Stage provenance / graph sidecars (read-only reuse).
    shutil.copy2(source_assignment, out_root / "pangenome_assignment.csv")
    if source_unif is not None:
        src = Path(source_unif)
        for name in (
            "sbs_assignment.csv",
            "pangenome_split_meta.json",
            "intersect_pangenome.csv",
        ):
            p = src / name
            if p.is_file():
                shutil.copy2(p, out_root / name)
        if copy_graph and (src / "graph").is_dir():
            dest = out_root / "graph"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src / "graph", dest)

    write_csv(out_root / "cv_master.csv", master, CV_MASTER_COLUMNS)
    (out_root / "cv_cluster_map.json").write_text(
        json.dumps(
            {k: int(v) for k, v in sorted(cluster_to_cv.items(), key=lambda kv: kv[0])},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    cv_mass: Counter[str] = Counter()
    for row in master:
        if not is_zsv_fold(row["cv_fold"]):
            cv_mass[row["cv_fold"]] += 1

    round_meta: list[dict[str, Any]] = []
    # fold0 split.csv also promoted to out_root/split.csv for adversarial labels.
    for holdout in range(n_cv):
        fold_dir = ensure_dir(out_root / f"fold{holdout}")
        split_rows = split_rows_for_loo_round(master, holdout=holdout, n_cv=n_cv)
        split_csv = fold_dir / "split.csv"
        write_csv(split_csv, split_rows, SPLIT_CSV_COLUMNS)
        summary = summarize_split_rows(split_rows)
        print(
            f"LOO fold{holdout}: {json.dumps(summary, sort_keys=True)}",
            flush=True,
        )
        split_root = run_split(
            split_csv,
            parsed_target=panel_root / "PREDICT",
            parsed_data=panel_root / "PARSED",
            outdir=fold_dir,
            strategy="traintestval",
            intersect_allow=True,
            id_csv=panel_root / "ID.csv",
        )
        # Mirror ZSV trees to fold outdir root for zsv_eval (run_split writes there).
        tsv = None
        # LegNet only when panel sequences are 230 bp (ready_legnet).
        try:
            tsv = build_legnet_tsv(
                split_root=split_root,
                out_tsv=fold_dir / "legnet_input" / "all.tsv",
            )
        except ValueError as exc:
            # Caduceus panels are long windows — skip TSV.
            if "230" not in str(exc):
                raise
            print(f"fold{holdout}: skip LegNet TSV ({exc})", flush=True)
        meta = {
            "holdout": holdout,
            "val_fold": (holdout + 1) % n_cv,
            "summary": summary,
            "split_csv": str(split_csv),
            "split_root": str(split_root),
            "legnet_tsv": str(tsv) if tsv else None,
        }
        (fold_dir / "loo_round_meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        round_meta.append(meta)
        if holdout == 0:
            shutil.copy2(split_csv, out_root / "split.csv")

    manifest = {
        "n_cv": n_cv,
        "seed": seed,
        "source_assignment": str(source_assignment),
        "source_unif": str(source_unif) if source_unif else None,
        "panel_root": str(panel_root),
        "out_root": str(out_root),
        "cv_mass": dict(sorted(cv_mass.items(), key=lambda kv: int(kv[0]))),
        "n_clusters_mapped": len(cluster_to_cv),
        "rounds": round_meta,
        "staged_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_root / "loo_cv_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (out_root / "split_done.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "n_cv": n_cv,
                "split_csv": str(out_root / "split.csv"),
                "folds": [str(out_root / f"fold{i}") for i in range(n_cv)],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def rematerialize_loo_folds_for_panel(
    *,
    source_loo_root: Path,
    out_root: Path,
    panel_root: Path,
    n_cv: int = DEFAULT_N_CV,
    build_legnet: bool = False,
) -> dict[str, Any]:
    """Reuse CV split tables from a prior LOO run; rematerialize for another panel."""
    from src.pipeline.legnet_input import build_legnet_tsv
    from src.pipeline.split import run_split

    source_loo_root = Path(source_loo_root)
    out_root = ensure_dir(Path(out_root))
    panel_root = Path(panel_root)

    src_manifest = source_loo_root / "loo_cv_manifest.json"
    if not src_manifest.is_file():
        raise FileNotFoundError(src_manifest)
    # Copy CV tables / graph.
    for name in (
        "cv_master.csv",
        "cv_cluster_map.json",
        "loo_cv_manifest.json",
        "pangenome_assignment.csv",
        "sbs_assignment.csv",
        "pangenome_split_meta.json",
        "intersect_pangenome.csv",
        "split.csv",
    ):
        p = source_loo_root / name
        if p.is_file():
            shutil.copy2(p, out_root / name)
    if (source_loo_root / "graph").is_dir():
        dest = out_root / "graph"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source_loo_root / "graph", dest)

    round_meta: list[dict[str, Any]] = []
    for holdout in range(n_cv):
        src_split = source_loo_root / f"fold{holdout}" / "split.csv"
        if not src_split.is_file():
            raise FileNotFoundError(src_split)
        fold_dir = ensure_dir(out_root / f"fold{holdout}")
        split_csv = fold_dir / "split.csv"
        shutil.copy2(src_split, split_csv)
        split_root = run_split(
            split_csv,
            parsed_target=panel_root / "PREDICT",
            parsed_data=panel_root / "PARSED",
            outdir=fold_dir,
            strategy="traintestval",
            intersect_allow=True,
            id_csv=panel_root / "ID.csv",
        )
        tsv = None
        if build_legnet:
            tsv = build_legnet_tsv(
                split_root=split_root,
                out_tsv=fold_dir / "legnet_input" / "all.tsv",
            )
        summary = summarize_split_rows(read_csv(split_csv))
        meta = {
            "holdout": holdout,
            "summary": summary,
            "split_csv": str(split_csv),
            "split_root": str(split_root),
            "legnet_tsv": str(tsv) if tsv else None,
            "reused_from": str(src_split),
        }
        (fold_dir / "loo_round_meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        round_meta.append(meta)

    manifest = {
        "n_cv": n_cv,
        "source_loo_root": str(source_loo_root),
        "panel_root": str(panel_root),
        "out_root": str(out_root),
        "rounds": round_meta,
        "staged_at": datetime.now(timezone.utc).isoformat(),
        "mode": "rematerialize_from_loo",
    }
    (out_root / "loo_cv_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (out_root / "split_done.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "n_cv": n_cv,
                "split_csv": str(out_root / "split.csv"),
                "source_loo_root": str(source_loo_root),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def run_loo_direct_trains(
    *,
    out_root: Path,
    panel_root: Path,
    model: str,
    n_cv: int = DEFAULT_N_CV,
    seed: int = 42,
    epochs: int = 30,
    min_epochs: int = 15,
    early_stopping_patience: int = 10,
    batch_size: int,
    max_length: int | None = None,
    n_devices: int = 1,
    num_workers: int = 4,
    strategy_name: str = "pangenome",
) -> dict[str, Any]:
    """Train one direct model per LOO fold; returns per-fold summaries."""
    from src.pipeline.train import run_train

    out_root = Path(out_root)
    panel_root = Path(panel_root)
    results: list[dict[str, Any]] = []
    for holdout in range(n_cv):
        fold_dir = out_root / f"fold{holdout}"
        direct_out = fold_dir / "direct"
        if (direct_out / "best_model" / "best_meta.json").is_file():
            print(f"fold{holdout}: reuse existing direct {direct_out}", flush=True)
            results.append({"holdout": holdout, "status": "reused", "outdir": str(direct_out)})
            continue
        if direct_out.exists():
            raise FileExistsError(f"refusing overwrite: {direct_out}")

        if model == "legnet":
            folders = fold_dir / "legnet_input" / "all.tsv"
            if not folders.is_file():
                raise FileNotFoundError(folders)
            kw: dict[str, Any] = {"legnet_demo": True}
        elif model == "caduceus":
            folders = fold_dir / "SPLIT"
            if not folders.is_dir():
                raise FileNotFoundError(folders)
            kw = {"max_length": max_length or 256}
        else:
            raise ValueError(f"unsupported model: {model}")

        print(
            f"fold{holdout}: direct {model} epochs={epochs} "
            f"min={min_epochs} patience={early_stopping_patience}",
            flush=True,
        )
        run_train(
            model=model,
            type="regression",
            folders=folders,
            outdir=direct_out,
            strategy=strategy_name,
            smoke=False,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed + holdout,
            n_devices=n_devices,
            num_workers=num_workers,
            zsv_root=fold_dir,
            eval_zsv=True,
            checkpoint_every_n_epochs=10,
            early_stopping_patience=early_stopping_patience,
            min_epochs=min_epochs,
            **kw,
        )
        try:
            from src.pipeline.pipeline_viz import run_pipeline_viz_auto

            run_pipeline_viz_auto(
                out_root=fold_dir,
                panel_root=panel_root,
                train_dir=direct_out,
                run_id=f"{out_root.name}_fold{holdout}",
                seed=seed,
                plot_train=True,
                plot_sbs=False,
                include_split_compare=True,
                viz_conda_env="caduceus_env",
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"WARNING: viz fold{holdout} skipped: {type(exc).__name__}: {exc}",
                flush=True,
            )
        results.append({"holdout": holdout, "status": "COMPLETED", "outdir": str(direct_out)})

    summary = {
        "n_cv": n_cv,
        "model": model,
        "folds": results,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_root / "loo_direct_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
