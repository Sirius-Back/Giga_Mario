#!/usr/bin/env python3
"""Refresh during-train monitoring figures from live Lightning / jsonl logs.

Uses existing ``src.train_viz`` learning-curve code (cnsplots + Altair).

Typical monitor loop while a job writes ``**/metrics.csv``::

  watch -n 60 'conda run -n caduceus_env python -m src.train_viz.train_monitor \\
    --run-dir run/run0/direct'

Pipeline Hydra calls ``refresh_train_monitor`` after each train stage, and
``refresh_pipeline_monitors`` at the end (direct + adversarial when present).

Also exports Caduceus-shaped ``<run_dir>/tensorboard/`` from synced jsonl.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _merge_lightning_epochs(metrics_csv: Path) -> list[dict[str, Any]]:
    """Collapse sparse Lightning metrics.csv rows into one record per epoch."""
    by_ep: dict[int, dict[str, Any]] = {}
    with metrics_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        for row in reader:
            ep_raw = row.get("epoch")
            if ep_raw in (None, ""):
                continue
            ep = int(float(ep_raw))
            rec = by_ep.setdefault(ep, {"epoch": ep})
            for key in fieldnames:
                if key in {"epoch", "step"}:
                    continue
                if not (
                    key.startswith("train_")
                    or key.startswith("val_")
                    or key.startswith("test_")
                ):
                    continue
                raw = row.get(key)
                if raw not in (None, ""):
                    try:
                        rec[key] = float(raw)
                    except ValueError:
                        pass
            raw_step = row.get("step")
            if raw_step not in (None, ""):
                try:
                    rec["step"] = float(raw_step)
                except ValueError:
                    pass
    return [by_ep[k] for k in sorted(by_ep)]


def _epoch_obj(rec: dict[str, Any]) -> dict[str, Any]:
    ep = int(rec["epoch"])
    obj: dict[str, Any] = {"epoch": ep}
    for prefix, split_name in (
        ("train", "train"),
        ("val", "validation"),
        ("test", "test"),
    ):
        block: dict[str, float] = {}
        loss_key = f"{prefix}_loss"
        if loss_key in rec:
            block["loss"] = float(rec[loss_key])
        for name in (
            "pearson",
            "spearman",
            "mse",
            "rmse",
            "mae",
            "r2",
            "genewise_pearson_median",
            "samplewise_pearson_median",
        ):
            key = f"{prefix}_{name}"
            if key in rec:
                block[name] = float(rec[key])
        if block:
            obj[split_name] = block
    if "step" in rec:
        obj["global_step"] = int(rec["step"])
    return obj


def find_lightning_metrics_csv(run_dir: Path) -> Path | None:
    csvs = sorted(
        p
        for p in Path(run_dir).rglob("metrics.csv")
        if "lightning_logs" in p.parts or p.parent.name in {"lightning_logs", "version_0"}
    )
    if not csvs:
        csvs = sorted(Path(run_dir).rglob("metrics.csv"))
    return csvs[0] if csvs else None


def sync_train_metrics_jsonl(
    run_dir: Path,
    *,
    preserve_zsv: bool = True,
) -> Path | None:
    """Rewrite ``logs/train_metrics.jsonl`` from Lightning CSV when present.

    Preserves non-smoke / ZSV final records already in the jsonl when requested.
    Returns the jsonl path, or None if nothing usable was found.
    """
    run_dir = Path(run_dir)
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jsonl_path = logs / "train_metrics.jsonl"

    kept: list[dict[str, Any]] = []
    if preserve_zsv and jsonl_path.is_file():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict) or rec.get("smoke"):
                continue
            # Keep ZSV / final-only rows (epoch may be "final")
            if "zero-shot-validation" in rec or "zero_shot" in rec:
                kept.append(rec)
            elif rec.get("epoch") == "final":
                kept.append(rec)

    # Prefer attaching ZSV from zero_shot_metrics.json if not already kept
    zsv_path = logs / "zero_shot_metrics.json"
    if preserve_zsv and zsv_path.is_file() and not any(
        "zero-shot-validation" in r or "zero_shot" in r for r in kept
    ):
        try:
            zsv = json.loads(zsv_path.read_text(encoding="utf-8"))
            metrics = zsv.get("metrics") if isinstance(zsv, dict) else None
            if isinstance(metrics, dict):
                kept.append({"epoch": "final", "zero-shot-validation": metrics})
        except json.JSONDecodeError:
            pass

    metrics_csv = find_lightning_metrics_csv(run_dir)
    epoch_rows: list[dict[str, Any]] = []
    if metrics_csv is not None:
        epoch_rows = _merge_lightning_epochs(metrics_csv)

    # Fallback: keep existing non-smoke epoch rows if no CSV
    if not epoch_rows and jsonl_path.is_file():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(rec, dict)
                and not rec.get("smoke")
                and isinstance(rec.get("epoch"), int)
            ):
                epoch_rows.append(rec)

    if not epoch_rows and not kept:
        return None

    lines: list[str] = []
    log_lines: list[str] = []
    for rec in epoch_rows:
        if "train" in rec or "validation" in rec:
            obj = rec
        else:
            obj = _epoch_obj(rec)
        lines.append(json.dumps(obj, sort_keys=True))
        tr = obj.get("train", {})
        va = obj.get("validation", {})
        log_lines.append(
            f"epoch={obj.get('epoch')} train_loss={tr.get('loss', float('nan'))} "
            f"val_loss={va.get('loss', float('nan'))} "
            f"val_pearson={va.get('pearson', float('nan'))}"
        )
        if isinstance(obj.get("epoch"), int):
            ep_dir = logs / f"epoch{int(obj['epoch'])}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            (ep_dir / "metrics.json").write_text(
                json.dumps(obj, indent=2) + "\n", encoding="utf-8"
            )
    for rec in kept:
        lines.append(json.dumps(rec, sort_keys=True))

    # Epoch-only log for learning curves (skip epoch="final" / ZSV rows —
    # those break numeric x-axis in train_viz._series).
    epoch_only = logs / "train_metrics_epochs.jsonl"
    epoch_lines = [
        line
        for line in lines
        if isinstance(json.loads(line).get("epoch"), int)
    ]
    epoch_only.write_text(
        "\n".join(epoch_lines) + ("\n" if epoch_lines else ""), encoding="utf-8"
    )

    jsonl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    (logs / "metrics.log").write_text(
        "\n".join(log_lines) + ("\n" if log_lines else ""), encoding="utf-8"
    )
    meta = {
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "lightning_csv": str(metrics_csv) if metrics_csv else None,
        "n_epochs": len(epoch_rows),
        "n_extra": len(kept),
        "epochs_jsonl": str(epoch_only),
    }
    (logs / "train_monitor_sync.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return jsonl_path


def refresh_train_monitor(
    run_dir: Path,
    *,
    outdir: Path | None = None,
    model: str | None = None,
    title: str | None = None,
    include_split_compare: bool = True,
) -> dict[str, Any]:
    """Sync metrics then render learning curves (+ optional split_compare)."""
    run_dir = Path(run_dir)
    outdir = Path(outdir) if outdir else (run_dir / "figures" / "train_monitor")
    outdir.mkdir(parents=True, exist_ok=True)

    jsonl = sync_train_metrics_jsonl(run_dir)
    manifest: dict[str, Any] = {
        "run_dir": str(run_dir),
        "outdir": str(outdir),
        "jsonl": str(jsonl) if jsonl else None,
        "learning_curves": [],
        "split_compare": None,
        "status": "ok",
    }

    def _finalize(status: str) -> dict[str, Any]:
        manifest["status"] = status
        try:
            from src.train_viz.tensorboard_metrics import write_tensorboard_from_jsonl

            manifest["tensorboard"] = write_tensorboard_from_jsonl(run_dir)
        except Exception as exc:  # noqa: BLE001
            manifest["tensorboard_error"] = f"{type(exc).__name__}: {exc}"
        (outdir / "train_monitor_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return manifest

    if jsonl is None or not jsonl.is_file() or jsonl.stat().st_size == 0:
        smoke_path = run_dir / "logs" / "train_metrics.jsonl"
        if smoke_path.is_file() and smoke_path.stat().st_size > 0:
            manifest["jsonl"] = str(smoke_path)
            return _finalize("smoke_only")
        return _finalize("no_metrics")

    # Skip smoke-only jsonl for curve plotting
    has_epoch = False
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if isinstance(rec, dict) and not rec.get("smoke") and isinstance(rec.get("epoch"), int):
            if "train" in rec or "validation" in rec:
                has_epoch = True
                break
    if not has_epoch:
        return _finalize("smoke_only")

    from src.train_viz.viz import main as train_viz_main

    model_name = model or run_dir.name
    viz_title = title or f"Train monitor — {model_name}"
    # Clear previous figures in outdir to avoid stale Figure_NN indices confusion
    for old in outdir.glob("Figure_*"):
        old.unlink()
    for name in (
        "train_metrics.csv",
        "training_summary.csv",
        "training_summary.md",
        "visualization_config.yaml",
    ):
        p = outdir / name
        if p.is_file():
            p.unlink()
    ms = outdir / "manuscript"
    if ms.is_dir():
        shutil.rmtree(ms)

    epochs_jsonl = run_dir / "logs" / "train_metrics_epochs.jsonl"
    viz_log = epochs_jsonl if epochs_jsonl.is_file() and epochs_jsonl.stat().st_size else jsonl
    rc = train_viz_main(
        [
            "--logs",
            str(viz_log),
            "--outdir",
            str(outdir),
            "--model",
            model_name,
            "--title",
            viz_title,
        ]
    )
    if rc not in (0, None):
        status = f"train_viz_failed:{rc}"
    else:
        status = "ok"
        manifest["learning_curves"] = sorted(str(p) for p in outdir.glob("Figure_*"))

    if include_split_compare:
        try:
            from src.train_viz.split_compare import run_split_compare

            sc_out = run_dir / "figures" / "split_compare"
            sc = run_split_compare(run_dir, sc_out, model=model_name)
            manifest["split_compare"] = sc
        except Exception as exc:  # noqa: BLE001
            manifest["split_compare_error"] = f"{type(exc).__name__}: {exc}"

    return _finalize(status)


def adversarial_train_dir(pipeline_root: Path) -> Path | None:
    """Return ``adversarial/train`` if that stage directory exists."""
    train = Path(pipeline_root) / "adversarial" / "train"
    return train if train.is_dir() else None


def refresh_pipeline_monitors(
    pipeline_root: Path,
    *,
    run_id: str | None = None,
    include_split_compare: bool = True,
) -> dict[str, Any]:
    """Refresh direct + adversarial (if present) train monitors and TensorBoard."""
    root = Path(pipeline_root)
    rid = run_id or root.name
    out: dict[str, Any] = {
        "pipeline_root": str(root),
        "direct": None,
        "adversarial": None,
        "status": "ok",
    }
    direct = root / "direct"
    if direct.is_dir():
        out["direct"] = refresh_train_monitor(
            direct,
            model=f"{rid}_direct",
            title=f"{rid} direct — train monitor",
            include_split_compare=include_split_compare,
        )
    adv = adversarial_train_dir(root)
    if adv is not None:
        out["adversarial"] = refresh_train_monitor(
            adv,
            model=f"{rid}_adversarial",
            title=f"{rid} adversarial — train monitor",
            include_split_compare=include_split_compare,
        )
    elif (root / "adversarial").is_dir():
        out["adversarial"] = {"status": "no_train_outdir", "path": str(root / "adversarial")}

    statuses = []
    for key in ("direct", "adversarial"):
        block = out.get(key)
        if isinstance(block, dict) and "status" in block:
            statuses.append(str(block["status"]))
    if statuses and all(s in {"no_metrics", "smoke_only", "no_train_outdir"} for s in statuses):
        if "ok" not in statuses:
            out["status"] = "no_usable_metrics"
    (root / "pipeline_monitor_manifest.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", type=Path, help="Single train outdir (direct or adv/train)")
    g.add_argument(
        "--pipeline-root",
        type=Path,
        help="Pipeline out_root: refresh direct + adversarial/train if present",
    )
    p.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=None,
        help="Default: <run-dir>/figures/train_monitor (single-run mode only)",
    )
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--run-id", type=str, default=None, help="Label prefix for pipeline mode")
    p.add_argument(
        "--no-split-compare",
        action="store_true",
        help="Skip final train/val/test/ZSV comparison panel",
    )
    args = p.parse_args(argv)
    if args.pipeline_root is not None:
        man = refresh_pipeline_monitors(
            args.pipeline_root,
            run_id=args.run_id,
            include_split_compare=not args.no_split_compare,
        )
    else:
        man = refresh_train_monitor(
            args.run_dir,
            outdir=args.outdir,
            model=args.model,
            title=args.title,
            include_split_compare=not args.no_split_compare,
        )
    print(json.dumps(man, indent=2))
    ok_statuses = {"ok", "smoke_only", "no_metrics", "no_usable_metrics", "no_train_outdir"}
    if args.pipeline_root is not None:
        return 0 if man.get("status") in ok_statuses else 1
    return 0 if man.get("status") in ok_statuses else 1


if __name__ == "__main__":
    raise SystemExit(main())
