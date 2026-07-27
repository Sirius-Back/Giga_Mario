#!/usr/bin/env python3
"""Evaluate a saved TPM Caduceus checkpoint on OUT_ADAPT_ZS caduceus_ready.

Reuses Dataset / collate / evaluate_split from train_caduceus_tpm.py.
Writes zero_shot_metrics.json + human metrics.log line (metrics.md keys).
Does not invent metrics — numbers come from actual inference only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Import shared train/eval helpers (same project)
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from train_caduceus_tpm import (  # noqa: E402
    TpmWindowDataset,
    collate_pad,
    evaluate_split,
)

_SKILL_SCRIPTS = Path(__file__).resolve().parents[1] / ".cursor" / "skills" / "caduceus" / "scripts"
if _SKILL_SCRIPTS.is_dir():
    sys.path.insert(0, str(_SKILL_SCRIPTS))
from metrics_logging import format_epoch_log  # noqa: E402

REQUIRED_KEYS = (
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="RUN_TPM/final_model (HF save_pretrained dir)",
    )
    p.add_argument(
        "--zero-shot-root",
        type=Path,
        required=True,
        help="OUT_ADAPT_ZS root (contains caduceus_ready/)",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        required=True,
        help="Output zero_shot_metrics.json path",
    )
    p.add_argument(
        "--out-log",
        type=Path,
        default=None,
        help="Optional human metrics.log path (default: sibling of out-json)",
    )
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="torch device (e.g. cuda:0 or cpu)",
    )
    return p.parse_args()


def resolve_zs_split(zs_ready: Path) -> str:
    for cand in ("all", "test", "val", "train"):
        if (zs_ready / cand / "labels.tsv").is_file():
            return cand
    raise FileNotFoundError(
        f"No labels.tsv under {zs_ready}/{{all,test,val,train}}"
    )


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    zs_root = args.zero_shot_root.resolve()
    out_json = args.out_json.resolve()
    out_log = (
        args.out_log.resolve()
        if args.out_log
        else out_json.with_name("zero_shot_metrics.log")
    )

    if not (model_dir / "config.json").is_file():
        raise SystemExit(f"Missing model config: {model_dir}/config.json")
    # Refuse predict-split1 / multi-class checkpoints (TPM is regression)
    cfg = json.loads((model_dir / "config.json").read_text())
    problem = cfg.get("problem_type")
    n_labels = cfg.get("num_labels")
    if n_labels is None:
        # HF may omit num_labels when id2label has a single regression head
        id2label = cfg.get("id2label") or {}
        n_labels = len(id2label) if id2label else None
    if problem != "regression":
        raise SystemExit(
            f"Refusing non-TPM regression checkpoint at {model_dir}: "
            f"problem_type={problem!r} num_labels={n_labels!r}"
        )
    if n_labels is not None and int(n_labels) != 1:
        raise SystemExit(
            f"Refusing multi-label/class checkpoint at {model_dir}: "
            f"problem_type={problem!r} num_labels={n_labels!r}"
        )

    zs_ready = zs_root / "caduceus_ready"
    if not zs_ready.is_dir():
        raise SystemExit(f"Missing caduceus_ready: {zs_ready}")
    split = resolve_zs_split(zs_ready)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 4
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    ds = TpmWindowDataset(zs_ready, split, tokenizer, args.max_length)
    loader = DataLoader(
        ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=lambda b: collate_pad(b, pad_id),
    )
    criterion = torch.nn.MSELoss()

    print(
        json.dumps(
            {
                "model_dir": str(model_dir),
                "zero_shot_root": str(zs_root),
                "zs_split": split,
                "n_samples": len(ds),
                "device": str(device),
                "eval_batch_size": args.eval_batch_size,
                "max_length": args.max_length,
                "seed": args.seed,
            },
            indent=2,
        ),
        flush=True,
    )

    metrics = evaluate_split(model, loader, device, criterion)
    missing = [k for k in REQUIRED_KEYS if k not in metrics]
    if missing:
        raise SystemExit(f"metrics.md keys missing after eval: {missing}")

    generated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "split": "zero-shot-validation",
        "zero-shot-validation": metrics,
        "provenance": {
            "source": "scripts/eval_caduceus_tpm_zs.py",
            "model_dir": str(model_dir),
            "zero_shot_root": str(zs_root),
            "zs_caduceus_ready_split": split,
            "n": metrics.get("n"),
            "device": str(device),
            "eval_batch_size": args.eval_batch_size,
            "max_length": args.max_length,
            "seed": args.seed,
            "generated_at_utc": generated_at,
            "note": (
                "Actual inference on RUN_TPM/final_model vs OUT_ADAPT_ZS; "
                "not fabricated. Dedicated T-9 eval (not promotion of epoch5)."
            ),
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    log_line = format_epoch_log("zero-shot-validation", metrics, epoch=None)
    out_log.write_text(log_line + "\n")
    print(log_line, flush=True)
    print(f"Wrote {out_json}", flush=True)
    print(f"Wrote {out_log}", flush=True)


if __name__ == "__main__":
    main()
