"""Resume/restart Caduceus train for run1 on existing caduceus_input (skip re-split).

Uses AMP + capped val/test eval. After epoch-1 ETR (~3.5 min/epoch), default
epochs=137 targets ~8h wall on 2×V100 (batch 192, max_length 256).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "runs" / "run1" / "direct"
SPLITS = OUT / "caduceus_input"

# Epoch-1 wall ≈ 210 s → floor(8*3600/210) = 137 epochs ≈ 8.0 h
DEFAULT_EPOCHS = 137


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from out/best_model (default: on)",
    )
    args = p.parse_args(argv)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2,3")
    # Avoid hostname→Gloo Invalid-argument socket failures on this host.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "0")
    if not (SPLITS / "train" / "labels.tsv").is_file():
        raise FileNotFoundError(f"Missing caduceus_input under {SPLITS}")
    # num_workers forced to 0 under DDP inside src.caduceus (CUDA+fork hang).
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "-m",
        "src.caduceus",
        "--splits-dir",
        str(SPLITS),
        "--out",
        str(OUT),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        "192",
        "--eval-batch-size",
        "192",
        "--max-length",
        "256",
        "--seed",
        "42",
        "--task",
        "regression",
        "--num-workers",
        "0",
        "--amp",
        "--eval-max-samples",
        "8192",
        "--train-eval-max-samples",
        "4096",
    ]
    if args.resume:
        cmd.append("--resume")
    print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    print("cmd:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        return rc
    # Post-train: mice ZSV + monitor (pipeline contract when zsv=true).
    zsv = ROOT / "runs" / "run1"
    zsv_cmd = [
        sys.executable,
        "-m",
        "src.pipeline.zsv_eval",
        "--model",
        "caduceus",
        "--outdir",
        str(OUT),
        "--split-root",
        str(zsv),
    ]
    print("zsv_cmd:", " ".join(zsv_cmd), flush=True)
    rc = subprocess.call(zsv_cmd)
    if rc != 0:
        return rc
    mon = [
        sys.executable,
        "-m",
        "src.train_viz.train_monitor",
        "--run-dir",
        str(OUT),
    ]
    print("monitor:", " ".join(mon), flush=True)
    return subprocess.call(mon)


if __name__ == "__main__":
    raise SystemExit(main())
