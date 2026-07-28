#!/usr/bin/env python3
"""human_legnet train entry — sole @legnet training script.

Input:
  --data-path  TSV with seq_id, seq, mean_value, fold (1..10), rev
               (from @legnet-adapt → legnet_ready/)

Output (--out, default runs/legnet/<data_stem>/):
  logs/                 run_config.json, train_metrics.jsonl, metrics.log, epoch{N}/
  model_{val}_{test}/   upstream Lightning dump + predictions
  best_model/           best val_pearson .ckpt
  final_model/          last epoch .ckpt
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


def _merge_lightning_epochs(metrics_csv: Path) -> list[dict[str, Any]]:
    """Collapse sparse Lightning metrics.csv rows into one record per epoch."""
    by_ep: dict[int, dict[str, Any]] = {}
    with metrics_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ep_raw = row.get("epoch")
            if ep_raw in (None, ""):
                continue
            ep = int(float(ep_raw))
            rec = by_ep.setdefault(ep, {"epoch": ep})
            for key in ("train_loss", "val_loss", "val_pearson", "step"):
                raw = row.get(key)
                if raw not in (None, ""):
                    try:
                        rec[key] = float(raw)
                    except ValueError:
                        pass
    return [by_ep[k] for k in sorted(by_ep)]


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
        if "train_loss" in rec:
            obj["train"] = {"loss": rec["train_loss"]}
        val: dict[str, float] = {}
        if "val_loss" in rec:
            val["loss"] = rec["val_loss"]
        if "val_pearson" in rec:
            val["pearson"] = rec["val_pearson"]
        if val:
            obj["validation"] = val
        if "step" in rec:
            obj["global_step"] = int(rec["step"])
        lines_jsonl.append(json.dumps(obj, sort_keys=True))
        # human log line
        tr = obj.get("train", {})
        va = obj.get("validation", {})
        lines_log.append(
            f"epoch={ep} train_loss={tr.get('loss', float('nan'))} "
            f"val_loss={va.get('loss', float('nan'))} "
            f"val_pearson={va.get('pearson', float('nan'))}"
        )
        ep_dir = logs / f"epoch{ep}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / "metrics.json").write_text(
            json.dumps(obj, indent=2) + "\n", encoding="utf-8"
        )
    jsonl_path.write_text("\n".join(lines_jsonl) + ("\n" if lines_jsonl else ""), encoding="utf-8")
    log_path.write_text("\n".join(lines_log) + ("\n" if lines_log else ""), encoding="utf-8")


def _copy_checkpoints(out_dir: Path) -> dict[str, str | None]:
    best_dir = out_dir / "best_model"
    final_dir = out_dir / "final_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    pearson_ckpts = sorted(out_dir.rglob("pearson-*.ckpt"))
    last_ckpts = sorted(out_dir.rglob("last_model-*.ckpt"))
    best_dst: str | None = None
    last_dst: str | None = None
    if pearson_ckpts:
        def _score(p: Path) -> float:
            name = p.name
            if "val_pearson=" in name:
                try:
                    return float(name.split("val_pearson=")[1].replace(".ckpt", ""))
                except ValueError:
                    return float("-inf")
            return float("-inf")

        src = max(pearson_ckpts, key=_score)
        dst = best_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        best_dst = str(dst)
    if last_ckpts:
        src = max(last_ckpts, key=lambda p: p.stat().st_mtime)
        dst = final_dir / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        last_dst = str(dst)
    return {"best_model": best_dst, "final_model": last_dst}


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
    if n_devices > 1:
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={n_devices}",
            *cmd_core,
        ]
    else:
        cmd = [sys.executable, *cmd_core]

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
    )


if __name__ == "__main__":
    raise SystemExit(main())
