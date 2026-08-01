#!/usr/bin/env python3
"""Best-checkpoint train/val/test(/zsv) regression metrics for unified runs.

Why this exists
---------------
- **Caduceus:** epoch jsonl already has Spearman; reports often used the *last*
  epoch. This module reads ``best_model/best_meta.json`` and pulls that epoch.
- **LegNet:** Lightning ``train_spearman`` / ``val_spearman`` are always NaN
  (TorchMetrics SpearmanCorrCoef under 16-mixed); only test predictions exist.
  Re-run the **best** ``.ckpt`` on train/val/test folds and compute Spearman
  offline (same formulas as ``src.legnet_demo_metrics``).

Writes ``<train_dir>/best_split_metrics.json`` (and updates report consumers).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.legnet_demo_metrics import regression_metrics

SPLIT_KEYS = ("train", "val", "test", "zsv")


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def best_meta(train_dir: Path) -> dict[str, Any] | None:
    return _load_json(train_dir / "best_model" / "best_meta.json")


def detect_model_family(train_dir: Path) -> str:
    """Return ``caduceus`` | ``legnet`` | ``unknown`` from path / artifacts."""
    parts = {p.lower() for p in train_dir.parts}
    if "caduceus" in parts:
        return "caduceus"
    if "legnet" in parts:
        return "legnet"
    if (train_dir / "caduceus_input").is_dir():
        return "caduceus"
    if list((train_dir / "best_model").glob("pearson-*.ckpt")):
        return "legnet"
    rc = _load_json(train_dir / "logs" / "run_config.json") or {}
    skill = str(rc.get("skill") or rc.get("model") or "").lower()
    if "caduceus" in skill:
        return "caduceus"
    if "legnet" in skill:
        return "legnet"
    return "unknown"


def _epoch_block_spearman(block: Any) -> float | None:
    if not isinstance(block, dict):
        return None
    for key, val in block.items():
        if "spearman" in str(key).lower():
            return _finite(val)
    return None


def _finite_metric_block(block: Any) -> dict[str, float]:
    """Keep finite numeric metrics from an epoch split block."""
    out: dict[str, float] = {}
    if not isinstance(block, dict):
        return out
    for key, val in block.items():
        if str(key).lower() in {"n", "count"}:
            fv = _finite(val)
            if fv is not None:
                out["n"] = fv
            continue
        fv = _finite(val)
        if fv is not None:
            out[str(key)] = fv
    return out


def _zsv_metric_block(train_dir: Path) -> dict[str, float]:
    zsv_path = train_dir / "logs" / "zero_shot_metrics.json"
    if not zsv_path.is_file():
        zsv_path = train_dir / "zero_shot_metrics.json"
    zsv_doc = _load_json(zsv_path)
    if not isinstance(zsv_doc, dict) or zsv_doc.get("skipped"):
        return {}
    metrics = zsv_doc.get("metrics") if isinstance(zsv_doc.get("metrics"), dict) else zsv_doc
    return _finite_metric_block(metrics)


def caduceus_best_from_jsonl(train_dir: Path) -> dict[str, Any]:
    """Pull train/val/test metrics from the best-checkpoint epoch in jsonl."""
    meta = best_meta(train_dir) or {}
    best_ep = meta.get("epoch")
    if best_ep is None:
        raise FileNotFoundError(f"missing best epoch in {train_dir / 'best_model' / 'best_meta.json'}")
    best_ep = int(best_ep)

    jsonl = train_dir / "logs" / "train_metrics.jsonl"
    if not jsonl.is_file():
        jsonl = train_dir / "train_metrics.jsonl"
    if not jsonl.is_file():
        raise FileNotFoundError(f"missing train_metrics.jsonl under {train_dir}")

    hit: dict[str, Any] | None = None
    for line in jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("epoch") == best_ep and any(
            isinstance(rec.get(k), dict) for k in ("train", "validation", "test")
        ):
            hit = rec
    if hit is None:
        raise ValueError(f"no epoch={best_ep} metrics in {jsonl}")

    by_split = {
        "train": _finite_metric_block(hit.get("train")),
        "val": _finite_metric_block(hit.get("validation") or hit.get("val")),
        "test": _finite_metric_block(hit.get("test")),
        "zsv": _zsv_metric_block(train_dir),
    }
    splits = {k: v.get("spearman") for k, v in by_split.items()}

    return {
        "model": "caduceus",
        "source": "best_epoch_jsonl",
        "best_epoch": best_ep,
        "best_meta": meta,
        "checkpoint": str(train_dir / "best_model"),
        "spearman": splits,
        "metrics_by_split": by_split,
        "epoch_record": {
            k: hit.get(k) for k in ("train", "validation", "test") if k in hit
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _import_legnet_vendor(vendor: Path) -> None:
    from src.legnet_core_launcher import _patch_numpy_legacy_aliases

    _patch_numpy_legacy_aliases()
    vendor = vendor.resolve()
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))


def _predict_split_legnet(
    *,
    model: Any,
    trainer: Any,
    df: Any,
    cfg: Any,
    batch_size: int,
    num_workers: int,
) -> tuple[list[float], list[float]]:
    """Forward-pass (no reverse) → (y_true, y_pred)."""
    import torch
    from dataset import TestSeqDatasetProb
    from torch.utils.data import DataLoader

    if df is None or len(df) == 0:
        return [], []
    ds = TestSeqDatasetProb(
        df,
        use_reverse_channel=cfg.use_reverse_channel,
        shift=0,
        reverse=False,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )
    preds = trainer.predict(model, dataloaders=dl)
    y_hat = torch.concat(preds).detach().float().cpu().view(-1).tolist()
    y_true = [float(v) for v in df["mean_value"].tolist()]
    if len(y_true) != len(y_hat):
        raise ValueError(f"pred/true length mismatch: {len(y_hat)} vs {len(y_true)}")
    return y_true, y_hat


def legnet_eval_best(
    train_dir: Path,
    *,
    device: int | None = None,
    batch_size: int | None = None,
    num_workers: int = 4,
    vendor: Path | None = None,
) -> dict[str, Any]:
    """Load best val_pearson ckpt; score train/val/test folds."""
    import lightning.pytorch as pl
    import pandas as pd

    train_dir = train_dir.resolve()
    meta = best_meta(train_dir) or {}
    ckpt_name = meta.get("checkpoint")
    ckpt_path = train_dir / "best_model" / str(ckpt_name) if ckpt_name else None
    if ckpt_path is None or not ckpt_path.is_file():
        hits = sorted((train_dir / "best_model").glob("pearson-*.ckpt"))
        if not hits:
            raise FileNotFoundError(f"no best pearson ckpt under {train_dir / 'best_model'}")
        ckpt_path = hits[0]

    cfg_path = train_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing {cfg_path}")
    run_cfg = _load_json(train_dir / "logs" / "run_config.json") or {}
    vendor = Path(vendor or run_cfg.get("vendor") or "software/human_legnet").resolve()
    _import_legnet_vendor(vendor)

    from trainer import LitModel
    from training_config import TrainingConfig

    # training=False avoids mkdir/dump side effects on existing model_dir
    tr_cfg = TrainingConfig.from_json(cfg_path, training=False)
    data_path = Path(tr_cfg.data_path)
    if not data_path.is_file():
        # adversarial: data lives next to train/
        alt = train_dir.parent / "legnet_input" / "all.tsv"
        if alt.is_file():
            tr_cfg.data_path = str(alt)
            data_path = alt
        else:
            raise FileNotFoundError(f"LegNet TSV missing: {data_path}")

    df = pd.read_csv(data_path, sep="\t")
    df.columns = ["seq_id", "seq", "mean_value", "fold_num", "rev"][0 : len(df.columns)]
    if "rev" in df.columns:
        df = df[df.rev == 0].copy()

    # Demo convention used by unif LegNet: val=2, test=1
    val_fold = int(run_cfg.get("val_fold") or 2)
    test_fold = int(run_cfg.get("test_fold") or 1)
    train_df = df[~df.fold_num.isin([val_fold, test_fold])]
    val_df = df[df.fold_num == val_fold]
    test_df = df[df.fold_num == test_fold]

    if device is None:
        device = int(run_cfg.get("device") or tr_cfg.device or 0)
    bs = int(batch_size or min(int(tr_cfg.valid_batch_size), 4096))

    model = LitModel.load_from_checkpoint(str(ckpt_path), tr_cfg=tr_cfg)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[int(device)],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        precision="16-mixed",
    )

    split_frames = {"train": train_df, "val": val_df, "test": test_df}
    metrics: dict[str, Any] = {}
    spearman: dict[str, float | None] = {}
    for name, frame in split_frames.items():
        y_true, y_pred = _predict_split_legnet(
            model=model,
            trainer=trainer,
            df=frame,
            cfg=tr_cfg,
            batch_size=bs,
            num_workers=num_workers,
        )
        if not y_true:
            metrics[name] = {"n": 0, "error": "empty_split"}
            spearman[name] = None
            continue
        m = regression_metrics(y_true, y_pred)
        metrics[name] = m
        spearman[name] = _finite(m.get("spearman"))

    # Keep existing ZSV metrics if present (mice TPM; may be task-mismatched for adv)
    zsv_block = _zsv_metric_block(train_dir)
    spearman["zsv"] = zsv_block.get("spearman")
    metrics_by_split = {
        "train": {k: v for k, v in (metrics.get("train") or {}).items() if _finite(v) is not None},
        "val": {k: v for k, v in (metrics.get("val") or {}).items() if _finite(v) is not None},
        "test": {k: v for k, v in (metrics.get("test") or {}).items() if _finite(v) is not None},
        "zsv": zsv_block,
    }

    return {
        "model": "legnet",
        "source": "best_ckpt_repredict",
        "best_epoch": meta.get("epoch"),
        "best_meta": meta,
        "checkpoint": str(ckpt_path),
        "val_fold": val_fold,
        "test_fold": test_fold,
        "device": device,
        "batch_size": bs,
        "spearman": spearman,
        "metrics": metrics,
        "metrics_by_split": metrics_by_split,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def write_best_split_metrics(train_dir: Path, payload: dict[str, Any]) -> Path:
    out = train_dir / "best_split_metrics.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def compute_for_train_dir(
    train_dir: Path,
    *,
    device: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    train_dir = train_dir.resolve()
    out_path = train_dir / "best_split_metrics.json"
    if out_path.is_file() and not force:
        existing = _load_json(out_path)
        if existing and isinstance(existing.get("spearman"), dict):
            return existing

    family = detect_model_family(train_dir)
    if family == "caduceus":
        payload = caduceus_best_from_jsonl(train_dir)
    elif family == "legnet":
        payload = legnet_eval_best(train_dir, device=device)
    else:
        raise ValueError(f"cannot detect model family for {train_dir}")
    write_best_split_metrics(train_dir, payload)
    return payload


def iter_unif_train_dirs(root: Path = Path("runs_unif")) -> list[Path]:
    dirs: list[Path] = []
    if not root.is_dir():
        return dirs
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        for run in sorted(model_dir.iterdir()):
            if not run.is_dir() or any(
                x in run.name for x in ("ARCHIVED", "FAILED", "BAD_", "adversarial_FAILED")
            ):
                continue
            for tdir in (run / "direct", run / "adversarial" / "train"):
                if (tdir / "best_model" / "best_meta.json").is_file():
                    dirs.append(tdir)
    return dirs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "train_dirs",
        nargs="*",
        type=Path,
        help="Train dirs (…/direct or …/adversarial/train). Default: all unif with best_model.",
    )
    ap.add_argument("--runs-root", type=Path, default=Path("runs_unif"))
    ap.add_argument("--device", type=int, default=None, help="GPU index for LegNet repredict")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--caduceus-only",
        action="store_true",
        help="Skip LegNet GPU repredict (jsonl best-epoch only)",
    )
    ap.add_argument(
        "--legnet-only",
        action="store_true",
        help="Only LegNet dirs",
    )
    args = ap.parse_args(argv)

    targets = list(args.train_dirs) if args.train_dirs else iter_unif_train_dirs(args.runs_root)
    if not targets:
        print("No train dirs found", file=sys.stderr)
        return 1

    n_ok = n_fail = 0
    for tdir in targets:
        family = detect_model_family(tdir)
        if args.caduceus_only and family != "caduceus":
            continue
        if args.legnet_only and family != "legnet":
            continue
        try:
            payload = compute_for_train_dir(tdir, device=args.device, force=args.force)
            sp = payload.get("spearman") or {}
            print(
                f"OK {tdir} [{payload.get('source')}] "
                f"train={sp.get('train')} val={sp.get('val')} "
                f"test={sp.get('test')} zsv={sp.get('zsv')}",
                flush=True,
            )
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {tdir}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            n_fail += 1
    print(f"done ok={n_ok} fail={n_fail}", flush=True)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
