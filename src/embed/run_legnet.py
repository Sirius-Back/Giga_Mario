"""CLI: validate → extract LegNet layers → leakage L(τ).

Example::

  python -m src.embed.run_legnet \\
    --runs-root runs_unif/legnet \\
    --out results/embed_legnet \\
    --stages validate,extract,leakage
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.embed import DEFAULT_LAYERS, LAYER_DIMS, ROLE_NAMES
from src.embed.discover import LegNetRun, discover_legnet_runs
from src.embed.distances import METRICS
from src.embed.leakage import (
    DEFAULT_TAU0,
    plot_l_tau_curves,
    run_leakage_for_store,
    write_ranking_table,
)
from src.embed.legnet_extract import LegNetLayerExtractor, load_lit_model
from src.embed.store import (
    allocate_layer_memmap,
    role_code,
    run_out_dir,
    write_ids_roles,
    write_manifest,
)
from src.embed.validate import (
    load_split_roles,
    load_tsv_index,
    validate_all,
    write_validation_report,
)
from src.pipeline.job_queue import (
    CLASS_CPU_RAM_HEAVY,
    CLASS_GPU_TRAIN,
    append_queue_entry,
    queue_path,
)
from src.pipeline.mem_guard import ensure_allocation_fits, wait_for_ram_headroom

ROOT = Path(__file__).resolve().parents[2]


def _append_queue(**kwargs: Any) -> None:
    try:
        append_queue_entry(**kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: queue.md append failed: {exc}", flush=True)


def _update_queue(name: str, status: str, *, note: str = "") -> None:
    """Append a follow-up status block (do not rewrite history)."""
    try:
        p = queue_path()
        lines = [
            f"\n### {name} — {status}",
            f"- **update time:** {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        ]
        if note:
            lines.append(f"- **note:** {note}")
        lines.append(f"- **status:** {status}")
        with p.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: queue.md update failed: {exc}", flush=True)


def build_ordered_panel(
    run: LegNetRun,
) -> tuple[list[str], list[int], list[str]]:
    """Return (ids, roles, sequences) in train→test→val order."""
    roles_map = load_split_roles(run.split_csv)
    tsv_index, issues = load_tsv_index(run.legnet_tsv)
    if issues:
        bad = [i for i in issues if "len=" in i or "duplicate" in i]
        if bad:
            raise ValueError(f"{run.key}: TSV issues: {bad[:5]}")
    ids: list[str] = []
    roles: list[int] = []
    seqs: list[str] = []
    for role_name in ROLE_NAMES:
        for sid in sorted(roles_map[role_name], key=lambda x: (len(x), x)):
            if sid not in tsv_index:
                raise KeyError(f"{run.key}: ID {sid} missing from TSV")
            ids.append(sid)
            roles.append(role_code(role_name))
            seqs.append(tsv_index[sid])
    return ids, roles, seqs


def extract_one(
    run: LegNetRun,
    out_base: Path,
    *,
    layers: tuple[str, ...],
    batch_size: int,
    device: str,
    skip_existing: bool,
) -> Path:
    out_dir = run_out_dir(out_base, run)
    manifest_path = out_dir / "manifest.json"
    if skip_existing and manifest_path.is_file():
        ok = all((out_dir / f"layer_{k}.npy").is_file() for k in layers)
        if ok:
            print(f"[extract] skip existing {run.key}", flush=True)
            return out_dir

    wait_for_ram_headroom(label=f"extract_wait_{run.key}")
    ids, roles, seqs = build_ordered_panel(run)
    n = len(ids)
    n_by_role = {
        "train": sum(1 for r in roles if r == 0),
        "test": sum(1 for r in roles if r == 1),
        "val": sum(1 for r in roles if r == 2),
    }
    # Pre-allocate memmaps
    memmaps: dict[str, np.memmap] = {}
    total_bytes = sum(n * LAYER_DIMS[k] * 4 for k in layers)
    ensure_allocation_fits(total_bytes, label=f"alloc_{run.key}")
    for k in layers:
        memmaps[k] = allocate_layer_memmap(
            out_dir / f"layer_{k}.npy", n, LAYER_DIMS[k], label=f"{run.key}_{k}"
        )

    write_ids_roles(out_dir, ids, roles)

    lit = load_lit_model(run, map_location=device)
    model = lit.model
    with LegNetLayerExtractor(model, device=device, layers=layers) as ex:
        for start in range(0, n, batch_size):
            batch_seqs = seqs[start : start + batch_size]
            feats = ex.extract_batch(batch_seqs)
            for k in layers:
                arr = feats[k]
                if arr.shape[1] != LAYER_DIMS[k]:
                    raise RuntimeError(
                        f"{run.key} layer {k}: got dim {arr.shape[1]} "
                        f"expected {LAYER_DIMS[k]}"
                    )
                memmaps[k][start : start + len(batch_seqs)] = arr
            if (start // batch_size) % 20 == 0:
                print(
                    f"[extract] {run.key} {start}/{n}",
                    flush=True,
                )
    for mm in memmaps.values():
        mm.flush()

    write_manifest(
        out_dir,
        run=run,
        layers=layers,
        n_by_role=n_by_role,
        extra={"batch_size": batch_size, "device": device, "n_total": n},
    )
    print(f"[extract] wrote {out_dir} n={n}", flush=True)
    return out_dir


def stage_validate(runs: list[LegNetRun], out: Path) -> list[Any]:
    results = validate_all(runs, load_ckpt=False)
    path = write_validation_report(results, out / "validation_report.json")
    print(f"[validate] {path}", flush=True)
    ready = [r for r in results if r.status == "READY"]
    print(
        f"[validate] READY={len(ready)} SKIPPED="
        f"{sum(1 for r in results if r.status == 'SKIPPED')} "
        f"FAILED={sum(1 for r in results if r.status == 'FAILED')}",
        flush=True,
    )
    return results


def stage_extract(
    runs: list[LegNetRun],
    val_results: list[Any],
    out: Path,
    *,
    layers: tuple[str, ...],
    batch_size: int,
    device: str,
    skip_existing: bool,
) -> list[Path]:
    ready_keys = {r.key for r in val_results if r.status == "READY"}
    targets = [r for r in runs if r.key in ready_keys]
    job = f"embed_legnet_extract_{int(time.time())}"
    _append_queue(
        name=job,
        status="RUNNING",
        job=f"python -m src.embed.run_legnet extract n={len(targets)}",
        pid=os.getpid(),
        estimated_time=f"{max(1, len(targets)) * 45}m",
        job_class=CLASS_GPU_TRAIN if device.startswith("cuda") else CLASS_CPU_RAM_HEAVY,
        peak_ram_gib=32.0,
        gpus=(0,) if device.startswith("cuda") else (),
        log=str(out / "extract.log"),
    )
    written: list[Path] = []
    try:
        for run in targets:
            try:
                written.append(
                    extract_one(
                        run,
                        out,
                        layers=layers,
                        batch_size=batch_size,
                        device=device,
                        skip_existing=skip_existing,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[extract] FAILED {run.key}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
        _update_queue(job, "COMPLETED")
    except Exception:
        _update_queue(job, "FAILED")
        raise
    return written


def stage_leakage(
    out: Path,
    *,
    layers: tuple[str, ...],
    metrics: tuple[str, ...],
    tau0: float,
) -> Path:
    job = f"embed_legnet_leakage_{int(time.time())}"
    _append_queue(
        name=job,
        status="RUNNING",
        job="python -m src.embed.run_legnet leakage",
        pid=os.getpid(),
        estimated_time="6h",
        job_class=CLASS_CPU_RAM_HEAVY,
        peak_ram_gib=48.0,
        gpus=(),
        log=str(out / "leakage.log"),
    )
    all_rows: list[dict[str, Any]] = []
    try:
        store_dirs = sorted(
            p.parent
            for p in out.glob("**/manifest.json")
            if p.parent.name != "leakage"
        )
        for sd in store_dirs:
            print(f"[leakage] {sd.relative_to(out)}", flush=True)
            rows = run_leakage_for_store(
                sd, layers=layers, metrics=metrics, tau0=tau0
            )
            all_rows.extend(rows)
        ranking = write_ranking_table(all_rows, out / "leakage_ranking.tsv")
        fig_dir = out / "figures"
        for layer in layers:
            if layer == "pred":
                continue
            for metric in ("centered_cosine", "whitened_cosine"):
                if metric not in metrics:
                    continue
                plot_l_tau_curves(
                    out,
                    layer=layer,
                    metric=metric,
                    out_pdf=fig_dir / f"L_tau_{layer}_{metric}.pdf",
                )
        (out / "leakage_all_summaries.json").write_text(
            json.dumps(all_rows, indent=2) + "\n", encoding="utf-8"
        )
        _update_queue(job, "COMPLETED")
        print(f"[leakage] ranking → {ranking}", flush=True)
        return ranking
    except Exception:
        _update_queue(job, "FAILED")
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-root",
        type=Path,
        default=ROOT / "runs_unif" / "legnet",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "embed_legnet",
    )
    p.add_argument(
        "--stages",
        type=str,
        default="validate,extract,leakage",
        help="Comma list: validate,extract,leakage",
    )
    p.add_argument(
        "--layers",
        type=str,
        default=",".join(DEFAULT_LAYERS),
    )
    p.add_argument(
        "--metrics",
        type=str,
        default=",".join(METRICS),
    )
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--tau0", type=float, default=DEFAULT_TAU0)
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip extract when manifest+layers exist",
    )
    p.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Cap discovered runs (debug)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    layers = tuple(s.strip() for s in args.layers.split(",") if s.strip())
    metrics = tuple(s.strip() for s in args.metrics.split(",") if s.strip())
    for k in layers:
        if k not in LAYER_DIMS:
            raise SystemExit(f"unknown layer {k!r}")

    runs = discover_legnet_runs(args.runs_root)
    if args.max_runs is not None:
        runs = runs[: int(args.max_runs)]
    print(f"Discovered {len(runs)} LegNet units under {args.runs_root}", flush=True)

    val_results: list[Any] = []
    if "validate" in stages:
        val_results = stage_validate(runs, out)
    else:
        val_results = validate_all(runs, load_ckpt=False)

    if "extract" in stages:
        stage_extract(
            runs,
            val_results,
            out,
            layers=layers,
            batch_size=int(args.batch_size),
            device=str(args.device),
            skip_existing=bool(args.skip_existing),
        )

    if "leakage" in stages:
        stage_leakage(out, layers=layers, metrics=metrics, tau0=float(args.tau0))

    (out / "run_complete.json").write_text(
        json.dumps(
            {
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "stages": sorted(stages),
                "n_runs_discovered": len(runs),
                "layers": list(layers),
                "metrics": list(metrics),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
