"""Pipeline visualization stage: train monitor + SBS split PCA diagnostics.

Runs after /train (and can be re-invoked standalone). When the current
interpreter lacks plotting deps (e.g. LegNet env without matplotlib), re-exec
via ``viz_conda_env`` (default ``caduceus_env``).

Usage::

  conda run -n caduceus_env python -m src.pipeline.pipeline_viz \\
    --out-root runs/run2 --panel-root ready_legnet \\
    --train-dir runs/run2/direct --run-id run2
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def has_viz_deps() -> bool:
    try:
        import matplotlib  # noqa: F401
        import cnsplots  # noqa: F401
        import altair  # noqa: F401
    except Exception:
        return False
    return True


def resolve_viz_python(viz_conda_env: str | None = "caduceus_env") -> Path | None:
    """Return a Python that can plot, or None if only the current interpreter."""
    if has_viz_deps():
        return Path(sys.executable)
    env = (viz_conda_env or "").strip()
    if not env:
        return None
    candidates = [
        Path.home() / "miniconda3" / "envs" / env / "bin" / "python",
        Path("/home/User14/miniconda3/envs") / env / "bin" / "python",
    ]
    conda = shutil.which("conda")
    for cand in candidates:
        if cand.is_file():
            return cand
    if conda and env:
        # Fall back to `conda run` wrapper path discovery
        try:
            out = subprocess.check_output(
                [conda, "run", "-n", env, "which", "python"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            p = Path(out.splitlines()[-1])
            if p.is_file():
                return p
        except (subprocess.CalledProcessError, IndexError, OSError):
            pass
    return None


def assignment_rows_from_split_csv(split_csv: Path) -> list[dict[str, str]]:
    """Map ``ID|train_test|fold`` → SBS assignment rows for PCA diagnostics."""
    from src.pipeline.common import read_csv

    rows_in = read_csv(Path(split_csv))
    out: list[dict[str, str]] = []
    for row in rows_in:
        rid = str(row.get("ID", "")).strip()
        if not rid:
            continue
        fold = str(row.get("fold", "")).strip()
        tt = str(row.get("train_test", "")).strip() or "NA"
        out.append(
            {
                "region": rid,
                "train_test": tt,
                "fold": fold if fold else "NA",
                "cluster": fold if fold else "NA",
                "additional": "",
            }
        )
    if not out:
        raise ValueError(f"empty split assignment: {split_csv}")
    return out


def resolve_sequence_dir(
    panel_root: Path,
    out_root: Path | None = None,
) -> Path:
    """Prefer MARKED (.fa), then panel PARSED (.ext), then materialized SPLIT FASTA."""
    panel_root = Path(panel_root)
    candidates: list[Path] = [
        panel_root / "MARKED",
        panel_root / "PARSED",
    ]
    if out_root is not None:
        out_root = Path(out_root)
        candidates.extend(
            [
                out_root / "SPLIT" / "FASTA" / "TRAIN",
                out_root / "PARSED" / "zero-shot-validation",
            ]
        )
    for path in candidates:
        if path.is_dir() and any(path.iterdir()):
            return path
    raise FileNotFoundError(
        "No sequence directory for SBS viz "
        f"(tried MARKED/PARSED under {panel_root})"
    )


def run_sbs_split_viz(
    *,
    out_root: Path,
    panel_root: Path,
    split_csv: Path | None = None,
    sequence_dir: Path | None = None,
    seed: int = 42,
    max_ids: int | None = None,
    max_points: int | None = None,
) -> dict[str, Any]:
    """GC%/AAA% features + PCA diagnostics colored by the pipeline split."""
    from src.splits.gc import compute_gc_feature_table
    from src.splits.sbs.visualize import DEFAULT_PLOT_N, plot_sbs_pca_diagnostics

    out_root = Path(out_root)
    panel_root = Path(panel_root)
    split_csv = Path(split_csv) if split_csv else out_root / "split.csv"
    if not split_csv.is_file():
        raise FileNotFoundError(f"split.csv missing: {split_csv}")

    assign_rows = assignment_rows_from_split_csv(split_csv)
    ids = [r["region"] for r in assign_rows]
    if max_ids is not None and max_ids > 0:
        ids = ids[: int(max_ids)]
        id_set = set(ids)
        assign_rows = [r for r in assign_rows if r["region"] in id_set]

    seq_dir = Path(sequence_dir) if sequence_dir else resolve_sequence_dir(
        panel_root, out_root
    )
    fig_dir = out_root / "figures" / "sbs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    features = compute_gc_feature_table(
        seq_dir, mode="directory", ids=ids, max_ids=None
    )
    feat_csv = features.write_csv(fig_dir / "feature_table.csv")
    plot_meta = plot_sbs_pca_diagnostics(
        features,
        assign_rows,
        outdir=fig_dir,
        id_csv=panel_root / "ID.csv",
        seed=seed,
        max_points=int(max_points) if max_points else DEFAULT_PLOT_N,
    )
    summary = {
        "status": "ok",
        "split_csv": str(split_csv),
        "sequence_dir": str(seq_dir),
        "n_ids": features.n,
        "feature_table": str(feat_csv),
        "figures": str(fig_dir),
        "plot": plot_meta,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (fig_dir / "sbs_split_viz_manifest.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary


def run_train_viz(
    *,
    train_dir: Path,
    run_id: str = "",
    include_split_compare: bool = True,
) -> dict[str, Any]:
    """Learning curves + split_compare (+ TB export via train_monitor)."""
    from src.train_viz.train_monitor import refresh_train_monitor

    train_dir = Path(train_dir)
    label = run_id or train_dir.name
    return refresh_train_monitor(
        train_dir,
        model=f"{label}_direct" if run_id else train_dir.name,
        title=f"{label} — train monitor",
        include_split_compare=include_split_compare,
    )


def run_pipeline_viz(
    *,
    out_root: Path,
    panel_root: Path,
    train_dir: Path | None = None,
    run_id: str = "",
    seed: int = 42,
    plot_train: bool = True,
    plot_sbs: bool = True,
    include_split_compare: bool = True,
    max_ids: int | None = None,
    max_points: int | None = None,
) -> dict[str, Any]:
    """Run train monitor and/or SBS PCA diagnostics for a pipeline outdir."""
    out_root = Path(out_root)
    panel_root = Path(panel_root)
    train_dir = Path(train_dir) if train_dir else out_root / "direct"
    manifest: dict[str, Any] = {
        "out_root": str(out_root),
        "panel_root": str(panel_root),
        "train_dir": str(train_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if plot_train and train_dir.is_dir():
        manifest["train"] = run_train_viz(
            train_dir=train_dir,
            run_id=run_id,
            include_split_compare=include_split_compare,
        )
    elif plot_train:
        manifest["train"] = {"status": "skipped", "reason": f"missing {train_dir}"}

    if plot_sbs and (out_root / "split.csv").is_file():
        manifest["sbs"] = run_sbs_split_viz(
            out_root=out_root,
            panel_root=panel_root,
            seed=seed,
            max_ids=max_ids,
            max_points=max_points,
        )
    elif plot_sbs:
        manifest["sbs"] = {
            "status": "skipped",
            "reason": f"missing {out_root / 'split.csv'}",
        }

    statuses = []
    for key in ("train", "sbs"):
        block = manifest.get(key)
        if isinstance(block, dict):
            statuses.append(str(block.get("status", "ok")))
    if any(s in {"error", "failed"} for s in statuses):
        manifest["status"] = "failed"
    elif all(s in {"skipped", "no_metrics", "smoke_only"} for s in statuses) and statuses:
        manifest["status"] = "skipped"
    else:
        manifest["status"] = "ok"

    out_path = out_root / "pipeline_viz_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    manifest["manifest"] = str(out_path)
    return manifest


def run_pipeline_viz_auto(
    *,
    out_root: Path,
    panel_root: Path,
    train_dir: Path | None = None,
    run_id: str = "",
    seed: int = 42,
    plot_train: bool = True,
    plot_sbs: bool = True,
    include_split_compare: bool = True,
    max_ids: int | None = None,
    max_points: int | None = None,
    viz_conda_env: str | None = "caduceus_env",
) -> dict[str, Any]:
    """Inline if deps present; else subprocess with viz_conda_env Python."""
    out_root = Path(out_root)
    panel_root = Path(panel_root)
    train_dir = Path(train_dir) if train_dir else out_root / "direct"

    if has_viz_deps():
        return run_pipeline_viz(
            out_root=out_root,
            panel_root=panel_root,
            train_dir=train_dir,
            run_id=run_id,
            seed=seed,
            plot_train=plot_train,
            plot_sbs=plot_sbs,
            include_split_compare=include_split_compare,
            max_ids=max_ids,
            max_points=max_points,
        )

    viz_py = resolve_viz_python(viz_conda_env)
    if viz_py is None:
        return {
            "status": "failed",
            "error": (
                "matplotlib/cnsplots unavailable in current env and "
                f"viz_conda_env={viz_conda_env!r} python not found"
            ),
        }

    cmd = [
        str(viz_py),
        "-m",
        "src.pipeline.pipeline_viz",
        "--out-root",
        str(out_root),
        "--panel-root",
        str(panel_root),
        "--train-dir",
        str(train_dir),
        "--seed",
        str(seed),
    ]
    if run_id:
        cmd.extend(["--run-id", run_id])
    if not plot_train:
        cmd.append("--no-train")
    if not plot_sbs:
        cmd.append("--no-sbs")
    if not include_split_compare:
        cmd.append("--no-split-compare")
    if max_ids is not None:
        cmd.extend(["--max-ids", str(max_ids)])
    if max_points is not None:
        cmd.extend(["--max-points", str(max_points)])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    print(f"pipeline_viz via {viz_py}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False)
    manifest_path = out_root / "pipeline_viz_manifest.json"
    if manifest_path.is_file():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["subprocess_exit"] = proc.returncode
        data["viz_python"] = str(viz_py)
        return data
    return {
        "status": "failed" if proc.returncode else "ok",
        "subprocess_exit": proc.returncode,
        "viz_python": str(viz_py),
        "error": "pipeline_viz_manifest.json not written",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--panel-root", type=Path, required=True)
    p.add_argument("--train-dir", type=Path, default=None)
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-train", action="store_true")
    p.add_argument("--no-sbs", action="store_true")
    p.add_argument("--no-split-compare", action="store_true")
    p.add_argument("--max-ids", type=int, default=None)
    p.add_argument("--max-points", type=int, default=None)
    args = p.parse_args(argv)

    manifest = run_pipeline_viz(
        out_root=args.out_root,
        panel_root=args.panel_root,
        train_dir=args.train_dir,
        run_id=args.run_id,
        seed=args.seed,
        plot_train=not args.no_train,
        plot_sbs=not args.no_sbs,
        include_split_compare=not args.no_split_compare,
        max_ids=args.max_ids,
        max_points=args.max_points,
    )
    print(json.dumps({"status": manifest.get("status"), "manifest": manifest.get("manifest")}, indent=2))
    return 0 if manifest.get("status") in {"ok", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
