#!/usr/bin/env python3
"""Score a saved final_model on train/val/test and write predictions + metrics.

Reads run_config.json (data_root, max_length, labels) beside final_model.
Writes:
  {run_dir}/final_predictions.csv   columns: split,y_true,y_score,y_pred
  {run_dir}/final_metrics.json      per-split loss/accuracy/auc (+ overall)

Used by @train-viz for final + ROC/AUC plots. Does not invent labels or paths.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class GbTxtDataset(Dataset):
    def __init__(self, root: Path, split: str, label_to_id: dict[str, int], tokenizer, max_length: int):
        self.samples: list[tuple[Path, int]] = []
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split dir missing: {split_dir}")
        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir() or label_dir.name not in label_to_id:
                continue
            lid = label_to_id[label_dir.name]
            for p in sorted(label_dir.glob("*.txt")):
                self.samples.append((p, lid))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        seq = path.read_text().strip().upper()
        enc = self.tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Trapezoidal ROC-AUC for binary labels {0,1}. NaN if undefined."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    classes = np.unique(y_true)
    if classes.size < 2:
        return float("nan")
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)
    # collapse tied scores
    distinct = np.r_[np.where(np.diff(y_score))[0], y_true.size - 1]
    tpr = tps[distinct] / n_pos
    fpr = fps[distinct] / n_neg
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    return float(np.trapz(tpr, fpr))


@torch.no_grad()
def score_split(model, loader, device) -> dict[str, object]:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_score: list[float] = []
    y_pred: list[int] = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        bs = batch["labels"].size(0)
        losses.append(out.loss.item() * bs)
        logits = out.logits
        probs = torch.softmax(logits, dim=-1)
        # Binary / multi-class: score = P(class=1) when binary; else max prob
        if probs.size(-1) == 2:
            score = probs[:, 1]
        else:
            score = probs.max(dim=-1).values
        pred = logits.argmax(dim=-1)
        y_true.extend(batch["labels"].cpu().tolist())
        y_score.extend(score.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
    n = len(y_true)
    if n == 0:
        return {
            "loss": float("nan"),
            "accuracy": float("nan"),
            "auc": float("nan"),
            "n": 0,
            "y_true": [],
            "y_score": [],
            "y_pred": [],
        }
    yt = np.asarray(y_true, dtype=int)
    ys = np.asarray(y_score, dtype=float)
    yp = np.asarray(y_pred, dtype=int)
    acc = float((yp == yt).mean())
    auc = roc_auc_binary(yt, ys) if len(np.unique(yt)) == 2 else float("nan")
    return {
        "loss": float(sum(losses) / n),
        "accuracy": acc,
        "auc": auc,
        "n": n,
        "y_true": y_true,
        "y_score": y_score,
        "y_pred": y_pred,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory with final_model/ and run_config.json",
    )
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument(
        "--device",
        default=None,
        help="cuda / cuda:0 / cpu (default: cuda if available)",
    )
    args = ap.parse_args(argv)

    run_dir = args.run_dir.resolve()
    model_dir = run_dir / "final_model"
    cfg_path = run_dir / "run_config.json"
    if not model_dir.is_dir():
        print(f"ERROR: final_model missing: {model_dir}", file=sys.stderr)
        return 2
    if not cfg_path.is_file():
        print(f"ERROR: run_config.json missing: {cfg_path}", file=sys.stderr)
        return 2

    cfg = json.loads(cfg_path.read_text())
    data_root = Path(cfg["data_root"])
    if not data_root.is_dir():
        print(f"ERROR: data_root not found: {data_root}", file=sys.stderr)
        return 2
    max_length = int(cfg.get("max_length", 512))
    labels_map = cfg.get("labels")
    if isinstance(labels_map, dict):
        # {"0": "high", "1": "low"} → name → id
        label_to_id = {str(v): int(k) for k, v in labels_map.items()}
    else:
        train_dir = data_root / "train"
        names = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
        label_to_id = {n: i for i, n in enumerate(names)}
    if not label_to_id:
        print("ERROR: could not resolve label_to_id from config/data", file=sys.stderr)
        return 2

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, trust_remote_code=True
    )
    model.to(device)

    # Caduceus GB trees use "val" on disk; logs use "validation"
    split_map = [("train", "train"), ("val", "validation"), ("test", "test")]
    pred_rows: list[dict[str, object]] = []
    metrics: dict[str, object] = {"run_dir": str(run_dir), "data_root": str(data_root)}

    for disk_split, report_split in split_map:
        split_path = data_root / disk_split
        if not split_path.is_dir():
            print(f"WARNING: skip missing split {split_path}", file=sys.stderr)
            continue
        ds = GbTxtDataset(data_root, disk_split, label_to_id, tokenizer, max_length)
        if len(ds) == 0:
            print(f"WARNING: empty split {disk_split}", file=sys.stderr)
            continue
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        out = score_split(model, loader, device)
        metrics[report_split] = {
            "loss": out["loss"],
            "accuracy": out["accuracy"],
            "auc": out["auc"],
            "n": out["n"],
        }
        for yt, ys, yp in zip(out["y_true"], out["y_score"], out["y_pred"]):
            pred_rows.append(
                {
                    "split": report_split,
                    "y_true": int(yt),
                    "y_score": float(ys),
                    "y_pred": int(yp),
                }
            )
        print(
            f"{report_split}: n={out['n']} loss={out['loss']:.4f} "
            f"acc={out['accuracy']:.4f} auc={out['auc']}"
        )

    if not pred_rows:
        print("ERROR: no predictions written (all splits empty/missing)", file=sys.stderr)
        return 2

    pred_path = run_dir / "final_predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["split", "y_true", "y_score", "y_pred"])
        w.writeheader()
        w.writerows(pred_rows)

    metrics_path = run_dir / "final_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {pred_path}")
    print(f"Wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
