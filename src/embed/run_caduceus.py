"""CLI: validate → extract Caduceus layers (phase-2).

Example::

  CUDA_VISIBLE_DEVICES=1 conda run -n caduceus_env --no-capture-output \\
    python -m src.embed.run_caduceus \\
      --runs-root runs_unif/caduceus \\
      --out results/embed_caduceus \\
      --stages validate,extract \\
      --loo-fold 0 \\
      --device cuda:0 \\
      --skip-existing
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

from src.embed import ROLE_NAMES
from src.embed.caduceus_extract import (
    CADUCEUS_DEFAULT_LAYERS,
    CADUCEUS_LAYER_DIMS,
    CaduceusLayerExtractor,
    load_caduceus_model,
)
from src.embed.discover_caduceus import CaduceusRun, discover_caduceus_runs
from src.embed.store import (
    allocate_layer_memmap,
    role_code,
    write_caduceus_manifest,
    write_ids_roles,
)
from src.embed.validate_caduceus import validate_all_caduceus, write_validation_report
from src.pipeline.job_queue import CLASS_GPU_TRAIN, append_queue_entry, queue_path
from src.pipeline.mem_guard import ensure_allocation_fits, wait_for_ram_headroom

ROOT = Path(__file__).resolve().parents[2]


def _append_queue(**kwargs: Any) -> None:
    try:
        append_queue_entry(**kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: queue.md append failed: {exc}", flush=True)


def _update_queue(name: str, status: str, *, note: str = "") -> None:
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


def _load_role_paths(
    splits_dir: Path, role: str, *, max_n: int | None
) -> tuple[list[str], list[Path]]:
    """Return (ids, seq_paths) for one role from caduceus_input."""
    labels = splits_dir / role / "labels.tsv"
    seq_dir = splits_dir / role / "sequences"
    ids: list[str] = []
    paths: list[Path] = []
    with labels.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        col = {n: i for i, n in enumerate(header)}
        if "sample_id" not in col:
            raise ValueError(f"{labels} missing sample_id")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= col["sample_id"]:
                continue
            sid = parts[col["sample_id"]].strip()
            path = seq_dir / f"{sid}.txt"
            if not path.is_file():
                continue
            ids.append(sid)
            paths.append(path)
            if max_n is not None and len(ids) >= max_n:
                break
    return ids, paths


def build_ordered_panel(
    run: CaduceusRun, *, max_per_role: int | None = None
) -> tuple[list[str], list[int], list[Path]]:
    """Return (ids, roles, seq_paths) in train→test→val order."""
    ids: list[str] = []
    roles: list[int] = []
    paths: list[Path] = []
    for role_name in ROLE_NAMES:
        r_ids, r_paths = _load_role_paths(
            run.splits_dir, role_name, max_n=max_per_role
        )
        order = sorted(range(len(r_ids)), key=lambda i: (len(r_ids[i]), r_ids[i]))
        for i in order:
            ids.append(r_ids[i])
            roles.append(role_code(role_name))
            paths.append(r_paths[i])
    return ids, roles, paths


def _read_seqs(paths: list[Path]) -> list[str]:
    return [p.read_text(encoding="utf-8").strip().upper() for p in paths]


def run_out_dir(base: Path, run: CaduceusRun) -> Path:
    if run.fold is None:
        return Path(base) / run.run_name
    return Path(base) / run.run_name / f"fold{run.fold}"


def extract_one(
    run: CaduceusRun,
    out_base: Path,
    *,
    layers: tuple[str, ...],
    batch_size: int,
    device: str,
    skip_existing: bool,
    max_per_role: int | None,
) -> Path:
    out_dir = run_out_dir(out_base, run)
    manifest_path = out_dir / "manifest.json"
    if skip_existing and manifest_path.is_file():
        ok = all((out_dir / f"layer_{k}.npy").is_file() for k in layers)
        if ok:
            print(f"[extract] skip existing {run.key}", flush=True)
            return out_dir

    wait_for_ram_headroom(label=f"cad_extract_wait_{run.key}")
    print(f"[extract] indexing panel {run.key} …", flush=True)
    ids, roles, paths = build_ordered_panel(run, max_per_role=max_per_role)
    n = len(ids)
    if n < 64:
        raise RuntimeError(f"{run.key}: panel too small n={n}")
    n_by_role = {
        "train": sum(1 for r in roles if r == 0),
        "test": sum(1 for r in roles if r == 1),
        "val": sum(1 for r in roles if r == 2),
    }
    memmaps: dict[str, np.memmap] = {}
    total_bytes = sum(n * CADUCEUS_LAYER_DIMS[k] * 4 for k in layers)
    ensure_allocation_fits(total_bytes, label=f"cad_alloc_{run.key}")
    for k in layers:
        memmaps[k] = allocate_layer_memmap(
            out_dir / f"layer_{k}.npy",
            n,
            CADUCEUS_LAYER_DIMS[k],
            label=f"{run.key}_{k}",
        )
    write_ids_roles(out_dir, ids, roles)

    model, tokenizer, _ = load_caduceus_model(run.model_dir, device=device)
    with CaduceusLayerExtractor(
        model,
        tokenizer,
        device=device,
        max_length=run.max_length,
        layers=layers,
        amp=True,
    ) as ex:
        for start in range(0, n, batch_size):
            batch_seqs = _read_seqs(paths[start : start + batch_size])
            feats = ex.extract_batch(batch_seqs)
            for k in layers:
                memmaps[k][start : start + len(batch_seqs)] = feats[k]
            if (start // batch_size) % 20 == 0:
                print(f"[extract] {run.key} {start}/{n}", flush=True)
    for mm in memmaps.values():
        mm.flush()
    del model
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.empty_cache()

    write_caduceus_manifest(
        out_dir,
        run_key=run.key,
        run_name=run.run_name,
        fold=run.fold,
        model_dir=run.model_dir,
        split_csv=run.split_csv,
        splits_dir=run.splits_dir,
        layers={k: CADUCEUS_LAYER_DIMS[k] for k in layers},
        n_by_role=n_by_role,
        extra={
            "batch_size": batch_size,
            "device": device,
            "n_total": n,
            "max_length": run.max_length,
            "max_per_role": max_per_role,
        },
    )
    print(f"[extract] wrote {out_dir} n={n}", flush=True)
    return out_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-root",
        type=Path,
        default=ROOT / "runs_unif" / "caduceus",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "embed_caduceus",
    )
    p.add_argument("--stages", type=str, default="validate,extract")
    p.add_argument(
        "--layers",
        type=str,
        default=",".join(CADUCEUS_DEFAULT_LAYERS),
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--loo-fold", type=int, default=0)
    p.add_argument(
        "--all-loo-folds",
        action="store_true",
        help="Extract every LOO fold (overrides --loo-fold)",
    )
    p.add_argument(
        "--run-keys",
        type=str,
        default=None,
        help="Comma-separated CaduceusRun.key filter",
    )
    p.add_argument(
        "--max-per-role",
        type=int,
        default=None,
        help="Cap sequences per role (debug / faster pairwise smoke)",
    )
    p.add_argument("--max-runs", type=int, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    layers = tuple(s.strip() for s in args.layers.split(",") if s.strip())
    for k in layers:
        if k not in CADUCEUS_LAYER_DIMS:
            raise SystemExit(f"unknown layer {k!r}")

    loo_fold = None if args.all_loo_folds else int(args.loo_fold)
    runs = discover_caduceus_runs(args.runs_root, loo_fold=loo_fold)
    if args.run_keys:
        want = {s.strip() for s in args.run_keys.split(",") if s.strip()}
        runs = [r for r in runs if r.key in want]
        missing = want - {r.key for r in runs}
        if missing:
            raise SystemExit(f"run-keys not found: {sorted(missing)}")
    if args.max_runs is not None:
        runs = runs[: int(args.max_runs)]
    print(
        f"Discovered {len(runs)} Caduceus units under {args.runs_root} "
        f"(loo_fold={loo_fold!r})",
        flush=True,
    )

    val_results: list[Any] = []
    if "validate" in stages:
        val_results = validate_all_caduceus(runs, load_model=False)
        path = write_validation_report(val_results, out / "validation_report.json")
        print(f"[validate] {path}", flush=True)
        print(
            f"[validate] READY={sum(1 for r in val_results if r.status=='READY')} "
            f"FAILED={sum(1 for r in val_results if r.status=='FAILED')}",
            flush=True,
        )
    elif "extract" in stages:
        val_results = validate_all_caduceus(runs, load_model=False)

    if "extract" in stages:
        ready = {r.key for r in val_results if r.status == "READY"}
        targets = [r for r in runs if r.key in ready]
        job = f"embed_caduceus_extract_{int(time.time())}"
        _append_queue(
            name=job,
            status="RUNNING",
            job=f"python -m src.embed.run_caduceus extract n={len(targets)}",
            pid=os.getpid(),
            estimated_time=f"{max(1, len(targets)) * 90}m",
            job_class=CLASS_GPU_TRAIN,
            peak_ram_gib=36.0,
            gpus=(0,) if str(args.device).startswith("cuda") else (),
            log=str(out / "extract.log"),
        )
        try:
            for run in targets:
                try:
                    extract_one(
                        run,
                        out,
                        layers=layers,
                        batch_size=int(args.batch_size),
                        device=str(args.device),
                        skip_existing=bool(args.skip_existing),
                        max_per_role=args.max_per_role,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[extract] FAILED {run.key}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
            _update_queue(job, "COMPLETED")
        except Exception:
            _update_queue(job, "FAILED")
            raise

    (out / "run_complete.json").write_text(
        json.dumps(
            {
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "stages": sorted(stages),
                "n_runs": len(runs),
                "layers": list(layers),
                "loo_fold": loo_fold,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
