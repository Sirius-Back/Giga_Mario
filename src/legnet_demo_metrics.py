#!/usr/bin/env python3
"""Summarize human_legnet demo metrics from Lightning logs + prediction TSV.

Computes metrics.md-style regression suite on test predictions
(pearson, spearman, mse, rmse, mae, r2) vs mean_value.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    return _pearson(ranks(xs), ranks(ys))


def _r2(ys: list[float], preds: list[float]) -> float:
    n = len(ys)
    if n < 2:
        return float("nan")
    my = sum(ys) / n
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, preds))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def regression_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    n = len(y_true)
    if n == 0 or n != len(y_pred):
        raise ValueError(f"length mismatch or empty: {n} vs {len(y_pred)}")
    errs = [p - y for y, p in zip(y_true, y_pred)]
    mse = sum(e * e for e in errs) / n
    mae = sum(abs(e) for e in errs) / n
    return {
        "n": float(n),
        "pearson": _pearson(y_true, y_pred),
        "spearman": _spearman(y_true, y_pred),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mae,
        "r2": _r2(y_true, y_pred),
    }


def find_predictions(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.rglob("predictions*.tsv"))
    # Prefer predictions_new_format if present
    for p in candidates:
        if "new_format" in p.name:
            return p
    return candidates[0] if candidates else None


def load_test_metrics(pred_path: Path) -> dict[str, Any]:
    y_true: list[float] = []
    y_pred: list[float] = []
    with pred_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"empty predictions: {pred_path}")
        for row in reader:
            yt = float(row["mean_value"])
            if "forw_pred" in row and "rev_pred" in row and row["forw_pred"] and row["rev_pred"]:
                yp = 0.5 * (float(row["forw_pred"]) + float(row["rev_pred"]))
            elif "forw_pred" in row and row["forw_pred"]:
                yp = float(row["forw_pred"])
            elif "pred" in row and row["pred"]:
                yp = float(row["pred"])
            else:
                raise KeyError(f"no prediction columns in {pred_path}: {reader.fieldnames}")
            y_true.append(yt)
            y_pred.append(yp)
    out = regression_metrics(y_true, y_pred)
    out["predictions_path"] = str(pred_path)
    out["pred_aggregation"] = "mean(forw_pred,rev_pred)" if "forw_pred" in (reader.fieldnames or []) else "single"
    return out


def parse_lightning_metrics(run_dir: Path) -> dict[str, Any]:
    """Best-effort parse of Lightning metrics.csv under model_*/lightning_logs."""
    csvs = sorted(run_dir.rglob("metrics.csv"))
    best_val_pearson: float | None = None
    last_row: dict[str, Any] = {}
    epochs: list[dict[str, Any]] = []
    for path in csvs:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                # Lightning may write sparse rows; keep rows with epoch
                ep = row.get("epoch")
                if ep is None or ep == "":
                    continue
                rec: dict[str, Any] = {"epoch": int(float(ep)), "source": str(path)}
                for key in ("val_pearson", "val_loss", "train_loss"):
                    raw = row.get(key)
                    if raw not in (None, ""):
                        rec[key] = float(raw)
                if "val_pearson" in rec:
                    vp = rec["val_pearson"]
                    if best_val_pearson is None or vp > best_val_pearson:
                        best_val_pearson = vp
                epochs.append(rec)
                last_row = rec
    return {
        "metrics_csv_files": [str(p) for p in csvs],
        "best_val_pearson": best_val_pearson,
        "last_logged": last_row,
        "n_epoch_rows": len(epochs),
        "epoch_rows": epochs[-40:],  # tail for summary size
    }


def find_checkpoints(run_dir: Path) -> dict[str, list[str]]:
    ckpts = sorted(run_dir.rglob("*.ckpt"))
    best = [str(p) for p in ckpts if "pearson" in p.name]
    last = [str(p) for p in ckpts if "last_model" in p.name]
    return {"best_val_pearson_ckpts": best, "last_ckpts": last, "all_ckpts": [str(p) for p in ckpts]}


def summarize(run_dir: Path) -> dict[str, Any]:
    pred = find_predictions(run_dir)
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "lightning": parse_lightning_metrics(run_dir),
        "checkpoints": find_checkpoints(run_dir),
    }
    if pred is None:
        summary["test"] = {"error": "no predictions*.tsv found"}
    else:
        summary["test"] = load_test_metrics(pred)
    return summary


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    test = summary.get("test", {})
    light = summary.get("lightning", {})
    lines = [
        "# LegNet demo metrics summary",
        "",
        f"- **run_dir:** `{summary.get('run_dir')}`",
        f"- **best val_pearson (Lightning):** {light.get('best_val_pearson')}",
        f"- **last logged:** `{light.get('last_logged')}`",
        "",
        "## Test set (vs mean_value)",
        "",
    ]
    if "error" in test:
        lines.append(f"Error: {test['error']}")
    else:
        lines.extend(
            [
                f"- predictions: `{test.get('predictions_path')}`",
                f"- aggregation: {test.get('pred_aggregation')}",
                f"- n = {int(test.get('n', 0))}",
                f"- pearson = {test.get('pearson')}",
                f"- spearman = {test.get('spearman')}",
                f"- mse = {test.get('mse')}",
                f"- rmse = {test.get('rmse')}",
                f"- mae = {test.get('mae')}",
                f"- r2 = {test.get('r2')}",
                "",
                "Caveat: labels are RNA-seq TPM, not lentiMPRA activity.",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/legnet_demo_GRCh38"),
        help="human_legnet --model_dir output",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="defaults to <run-dir>/metrics_summary.json",
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="defaults to <run-dir>/metrics_summary.md",
    )
    args = ap.parse_args(argv)
    run_dir = args.run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"run dir missing: {run_dir}")
    summary = summarize(run_dir)
    out_json = args.out_json or (run_dir / "metrics_summary.json")
    out_md = args.out_md or (run_dir / "metrics_summary.md")
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, out_md)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    test = summary.get("test", {})
    if "pearson" in test:
        print(
            f"test pearson={test['pearson']:.4f} spearman={test['spearman']:.4f} "
            f"rmse={test['rmse']:.4f} n={int(test['n'])}"
        )
    print(f"best_val_pearson={summary.get('lightning', {}).get('best_val_pearson')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
