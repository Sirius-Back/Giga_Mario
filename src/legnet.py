#!/usr/bin/env python3
"""human_legnet train entry — sole @legnet training script.

Input:
  --data-path  TSV with seq_id, seq, mean_value, fold (1..10), rev
               (from @legnet-adapt → legnet_ready/)

Output (--out, default runs/legnet/<data_stem>/):
  logs/                 run_config.json, train_metrics.jsonl, metrics.log, epoch{N}/
  tensorboard/          train/val scalars (Lightning + jsonl backfill)
  model_{val}_{test}/   upstream Lightning dump + predictions
  checkpoints/          periodic every-N-epoch .ckpt files (default N=10)
  best_model/           best val_pearson .ckpt + best_meta.json
  final_model/          copy of best_model (selected after train)
  metrics_summary.json / .md
  train_time.json

Wraps software/human_legnet/core.py (do not reimplement the CNN train loop).
Normalizes Lightning metrics.csv → Caduceus-shaped jsonl for @train-viz.

Example:
  python -m src.legnet --data-path legnet_ready/GCF_….tsv --demo --epochs 20
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.legnet_demo_metrics import summarize, write_markdown

REQUIRED_COLS = ("seq_id", "seq", "mean_value", "fold", "rev")
DEFAULT_VENDOR = Path("software/human_legnet")
STITCHED_LEN = 230


def _validate_tsv(path: Path, *, max_check: int = 5000) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Data TSV missing: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Data TSV empty: {path}")
    n = 0
    folds: set[int] = set()
    bad_len = 0
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header in {path}")
        missing = [c for c in REQUIRED_COLS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} missing columns {missing}; got {reader.fieldnames}")
        for row in reader:
            n += 1
            f = int(float(row["fold"]))
            folds.add(f)
            if n <= max_check and len(row["seq"]) != STITCHED_LEN:
                bad_len += 1
    if n == 0:
        raise ValueError(f"No data rows in {path}")
    if folds - set(range(1, 11)):
        raise ValueError(
            f"fold values must be in 1..10 (human_legnet CV); got {sorted(folds)}. "
            "Remap with (fold%10)+1 or re-run @legnet-adapt."
        )
    if bad_len:
        raise ValueError(
            f"{bad_len} sequences among first {max_check} rows ≠ {STITCHED_LEN} bp"
        )
    return {"n_rows": n, "folds": sorted(folds), "seq_len": STITCHED_LEN}


_METRIC_KEYS = (
    "loss",
    "pearson",
    "spearman",
    "mse",
    "rmse",
    "mae",
    "r2",
    "genewise_pearson_median",
    "samplewise_pearson_median",
)


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
                if key in {"epoch", "step"} or key not in row:
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


def _split_block(rec: dict[str, Any], prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    loss_key = f"{prefix}_loss"
    if loss_key in rec:
        out["loss"] = float(rec[loss_key])
    for name in _METRIC_KEYS:
        if name == "loss":
            continue
        key = f"{prefix}_{name}"
        if key in rec:
            out[name] = float(rec[key])
    return out


def _write_caduceus_like_logs(out_dir: Path, epoch_rows: list[dict[str, Any]]) -> None:
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jsonl_path = logs / "train_metrics.jsonl"
    log_path = logs / "metrics.log"
    lines_jsonl: list[str] = []
    lines_log: list[str] = []
    for rec in epoch_rows:
        ep = int(rec["epoch"])
        obj: dict[str, Any] = {"epoch": ep}
        train = _split_block(rec, "train")
        val = _split_block(rec, "val")
        test = _split_block(rec, "test")
        if train:
            obj["train"] = train
        if val:
            obj["validation"] = val
        if test:
            obj["test"] = test
        if "step" in rec:
            obj["global_step"] = int(rec["step"])
        lines_jsonl.append(json.dumps(obj, sort_keys=True))
        tr = obj.get("train", {})
        va = obj.get("validation", {})
        lines_log.append(
            f"epoch={ep} train_loss={tr.get('loss', float('nan'))} "
            f"val_loss={va.get('loss', float('nan'))} "
            f"val_pearson={va.get('pearson', float('nan'))} "
            f"val_spearman={va.get('spearman', float('nan'))} "
            f"val_mse={va.get('mse', float('nan'))} "
            f"val_rmse={va.get('rmse', float('nan'))} "
            f"val_mae={va.get('mae', float('nan'))} "
            f"val_r2={va.get('r2', float('nan'))}"
        )
        ep_dir = logs / f"epoch{ep}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / "metrics.json").write_text(
            json.dumps(obj, indent=2) + "\n", encoding="utf-8"
        )
    jsonl_path.write_text("\n".join(lines_jsonl) + ("\n" if lines_jsonl else ""), encoding="utf-8")
    log_path.write_text("\n".join(lines_log) + ("\n" if lines_log else ""), encoding="utf-8")


def _parse_epoch_from_ckpt_name(name: str) -> int | None:
    """Extract epoch index from Lightning filenames like ``pearson-epoch=12-…``."""
    import re

    m = re.search(r"epoch[=_-](\d+)", name)
    if not m:
        return None
    return int(m.group(1))


def _parse_val_pearson_from_ckpt_name(name: str) -> float | None:
    if "val_pearson=" not in name:
        return None
    try:
        return float(name.split("val_pearson=")[1].replace(".ckpt", ""))
    except ValueError:
        return None


def _copy_checkpoints(out_dir: Path) -> dict[str, Any]:
    """Promote best val_pearson ckpt to best_model/ and final_model/.

    Periodic ``epoch-*.ckpt`` files (every N epochs from human_legnet) are
    collected under ``checkpoints/``. Last-epoch weights stay in Lightning dirs
    only — ``final_model/`` is always the selected best checkpoint.
    """
    best_dir = out_dir / "best_model"
    final_dir = out_dir / "final_model"
    ckpt_root = out_dir / "checkpoints"
    best_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    pearson_ckpts = sorted(out_dir.rglob("pearson-*.ckpt"))
    # Lightning stores under …/lightning_logs/…/checkpoints/; only skip our
    # promoted roots (best_model/, final_model/, top-level checkpoints/).
    skip_roots = {best_dir.resolve(), final_dir.resolve(), ckpt_root.resolve()}
    periodic_ckpts = sorted(
        p
        for p in out_dir.rglob("epoch-*.ckpt")
        if p.name.startswith("epoch-")
        and not any(
            skip == p.resolve().parent or skip in p.resolve().parents
            for skip in skip_roots
        )
    )
    last_ckpts = sorted(out_dir.rglob("last_model-*.ckpt"))

    best_dst: str | None = None
    final_dst: str | None = None
    best_meta: dict[str, Any] | None = None

    if pearson_ckpts:

        def _score(p: Path) -> float:
            parsed = _parse_val_pearson_from_ckpt_name(p.name)
            return parsed if parsed is not None else float("-inf")

        src = max(pearson_ckpts, key=_score)
        dst = best_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        best_dst = str(dst)
        # final_model = best (not last)
        final_path = final_dir / src.name
        if src.resolve() != final_path.resolve():
            shutil.copy2(src, final_path)
        final_dst = str(final_path)
        ep = _parse_epoch_from_ckpt_name(src.name)
        score = _parse_val_pearson_from_ckpt_name(src.name)
        best_meta = {
            "epoch": ep,
            "metric": "val_pearson",
            "value": score,
            "val_pearson": score,
            "selection": "max_val_pearson",
            "checkpoint": src.name,
            "promoted_to_final": True,
        }
        (best_dir / "best_meta.json").write_text(
            json.dumps(best_meta, indent=2) + "\n", encoding="utf-8"
        )
        (final_dir / "best_meta.json").write_text(
            json.dumps(best_meta, indent=2) + "\n", encoding="utf-8"
        )
    elif last_ckpts:
        # Fallback when no pearson monitor ckpt exists
        src = max(last_ckpts, key=lambda p: p.stat().st_mtime)
        dst = final_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        final_dst = str(dst)
        best_meta = {
            "epoch": _parse_epoch_from_ckpt_name(src.name),
            "metric": "last",
            "selection": "last_epoch_fallback",
            "checkpoint": src.name,
            "promoted_to_final": True,
        }
        (final_dir / "best_meta.json").write_text(
            json.dumps(best_meta, indent=2) + "\n", encoding="utf-8"
        )

    for src in periodic_ckpts:
        dst = ckpt_root / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

    return {
        "best_model": best_dst,
        "final_model": final_dst,
        "checkpoints": str(ckpt_root) if periodic_ckpts else None,
        "best_meta": best_meta,
    }


def _find_metrics_csv(out_dir: Path) -> Path | None:
    csvs = sorted(out_dir.rglob("metrics.csv"))
    return csvs[0] if csvs else None


def run(
    *,
    data_path: Path,
    out_dir: Path,
    vendor: Path,
    epochs: int,
    device: int,
    n_devices: int,
    seed: int,
    demo: bool,
    use_shift: bool,
    reverse_augment: bool,
    use_reverse_channel: bool,
    train_batch_size: int,
    valid_batch_size: int,
    num_workers: int,
    checkpoint_every_n_epochs: int = 10,
    early_stopping_patience: int = 0,
    min_epochs: int = 0,
) -> int:
    data_path = data_path.resolve()
    out_dir = out_dir.resolve()
    vendor = vendor.resolve()
    core = vendor / "core.py"
    if not core.is_file():
        raise FileNotFoundError(
            f"human_legnet core.py missing at {core}. "
            "Clone https://github.com/autosome-ru/human_legnet into software/human_legnet"
        )

    stats = _validate_tsv(data_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    run_config = {
        "skill": "legnet",
        "producer": "src/legnet.py",
        "data_path": str(data_path),
        "out": str(out_dir),
        "vendor": str(vendor),
        "epochs": epochs,
        "device": device,
        "n_devices": n_devices,
        "seed": seed,
        "demo": demo,
        "use_shift": use_shift,
        "reverse_augment": reverse_augment,
        "use_reverse_channel": use_reverse_channel,
        "train_batch_size": train_batch_size,
        "valid_batch_size": valid_batch_size,
        "num_workers": num_workers,
        "checkpoint_every_n_epochs": checkpoint_every_n_epochs,
        "early_stopping_patience": early_stopping_patience,
        "min_epochs": min_epochs,
        "data_stats": stats,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (logs / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )

    cmd_core = [
        str(core),
        "--model_dir",
        str(out_dir),
        "--data_path",
        str(data_path),
        "--epoch_num",
        str(epochs),
        "--device",
        str(device),
        "--seed",
        str(seed),
        "--train_batch_size",
        str(train_batch_size),
        "--valid_batch_size",
        str(valid_batch_size),
        "--num_workers",
        str(num_workers),
        "--checkpoint_every_n_epochs",
        str(checkpoint_every_n_epochs),
        "--early_stopping_patience",
        str(int(early_stopping_patience)),
        "--min_epochs",
        str(int(min_epochs)),
    ]
    if demo:
        cmd_core.append("--demo")
    if use_shift:
        cmd_core.append("--use_shift")
    if reverse_augment:
        cmd_core.append("--reverse_augment")
    if use_reverse_channel:
        cmd_core.append("--use_reverse_channel")

    env = os.environ.copy()
    env["LEGNET_N_DEVICES"] = str(max(1, n_devices))
    # Single process entry: Lightning Trainer(devices=N, strategy=ddp) spawns
    # workers. Do not wrap with torchrun — that double-launches DDP and hangs
    # or IndexErrors depending on devices=N vs devices=1.
    # Launch via numpy-compat wrapper (LegNet-only; does not affect Caduceus).
    launcher = Path(__file__).resolve().parent / "legnet_core_launcher.py"
    cmd = [sys.executable, str(launcher), *cmd_core]

    t0 = time.time()
    print(
        f"Launching (n_devices={n_devices}): {' '.join(cmd)}",
        flush=True,
    )
    proc = subprocess.run(cmd, cwd=str(vendor), check=False, env=env)
    elapsed = time.time() - t0
    train_time = {
        "elapsed_sec": elapsed,
        "exit_code": proc.returncode,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "train_time.json").write_text(
        json.dumps(train_time, indent=2) + "\n", encoding="utf-8"
    )
    if proc.returncode != 0:
        print(f"human_legnet core.py failed with exit {proc.returncode}", flush=True)
        return proc.returncode

    metrics_csv = _find_metrics_csv(out_dir)
    if metrics_csv is not None:
        epoch_rows = _merge_lightning_epochs(metrics_csv)
        _write_caduceus_like_logs(out_dir, epoch_rows)
        print(f"Wrote {out_dir / 'logs' / 'train_metrics.jsonl'} ({len(epoch_rows)} epochs)")
    else:
        print(f"WARNING: no metrics.csv under {out_dir}", flush=True)

    try:
        from src.train_viz.tensorboard_metrics import write_tensorboard_from_jsonl

        tb_man = write_tensorboard_from_jsonl(out_dir)
        print(
            f"tensorboard status={tb_man.get('status')} "
            f"n_scalars={tb_man.get('n_scalars')} → {tb_man.get('tensorboard')}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: tensorboard export skipped: {type(exc).__name__}: {exc}", flush=True)

    ckpt_info = _copy_checkpoints(out_dir)
    print(f"Checkpoints: {ckpt_info}", flush=True)

    summary = summarize(out_dir)
    (out_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(summary, out_dir / "metrics_summary.md")
    print(f"Wrote {out_dir / 'metrics_summary.md'}", flush=True)
    test = summary.get("test", {})
    if "pearson" in test:
        print(
            f"test pearson={test['pearson']:.4f} spearman={test['spearman']:.4f} "
            f"rmse={test['rmse']:.4f} n={int(test['n'])}",
            flush=True,
        )
    print(f"Done in {elapsed:.1f}s → {out_dir}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-path", type=Path, required=True, help="legnet_ready TSV")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dir (default runs/legnet/<data_stem>)",
    )
    ap.add_argument("--vendor", type=Path, default=DEFAULT_VENDOR)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", type=int, default=0, help="Primary GPU index (single-GPU mode)")
    ap.add_argument(
        "--n-devices",
        type=int,
        default=1,
        help="Number of GPUs (Lightning ddp_spawn when >1; sets LEGNET_N_DEVICES)",
    )
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--demo", action="store_true", help="Single CV split (test=1,val=2)")
    ap.add_argument("--use-shift", action="store_true")
    ap.add_argument("--reverse-augment", action="store_true")
    ap.add_argument("--use-reverse-channel", action="store_true")
    ap.add_argument("--train-batch-size", type=int, default=1024)
    ap.add_argument("--valid-batch-size", type=int, default=1024)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument(
        "--checkpoint-every-n-epochs",
        type=int,
        default=10,
        help="Periodic Lightning checkpoints every N epochs (0 disables). "
        "Best val_pearson is always kept; final_model/ is set to best after train.",
    )
    ap.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after N epochs without val_pearson improvement (0=off).",
    )
    ap.add_argument(
        "--min-epochs",
        type=int,
        default=0,
        help="Minimum epochs before early stopping can end training (0=off).",
    )
    args = ap.parse_args(argv)

    out = args.out
    if out is None:
        out = Path("runs/legnet") / args.data_path.stem

    return run(
        data_path=args.data_path,
        out_dir=out,
        vendor=args.vendor,
        epochs=args.epochs,
        device=args.device,
        n_devices=args.n_devices,
        seed=args.seed,
        demo=args.demo,
        use_shift=args.use_shift,
        reverse_augment=args.reverse_augment,
        use_reverse_channel=args.use_reverse_channel,
        train_batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size,
        num_workers=args.num_workers,
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
        early_stopping_patience=args.early_stopping_patience,
        min_epochs=args.min_epochs,
    )


if __name__ == "__main__":
    raise SystemExit(main())
