"""Final-model zero-shot-validation evaluation on pipeline ZSV trees."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import ensure_dir, read_csv, sanitize_filename


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return float("nan")
    return num / (den_x * den_y)


def _spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    return _pearson(ranks(xs), ranks(ys))


def _mse(xs: list[float], ys: list[float]) -> float:
    if not xs:
        return float("nan")
    return sum((p - t) ** 2 for p, t in zip(xs, ys)) / len(xs)


def load_zsv_pairs(
    *,
    parsed_root: Path,
    predict_root: Path,
) -> list[tuple[str, str, float]]:
    """Return ``(id, sequence, target)`` for ZSV holdouts."""
    parsed_root = Path(parsed_root)
    predict_root = Path(predict_root)
    zsv_parsed = parsed_root / "zero-shot-validation"
    zsv_predict = predict_root / "zero-shot-validation"
    if not zsv_parsed.is_dir():
        # Allow panel-level PREDICT/PARSED/zero-shot-validation before SPLIT
        zsv_parsed = parsed_root if parsed_root.name == "zero-shot-validation" else zsv_parsed
    if not zsv_predict.is_dir():
        zsv_predict = predict_root if predict_root.name == "zero-shot-validation" else zsv_predict
    if not zsv_parsed.is_dir() or not zsv_predict.is_dir():
        raise FileNotFoundError(
            f"ZSV trees missing: need {zsv_parsed} and {zsv_predict}"
        )

    predict_csv = zsv_predict / "predict.csv"
    if predict_csv.is_file():
        rows = read_csv(predict_csv)
        by_id = {r["id"].strip(): float(r["predict_var1"]) for r in rows}
    else:
        by_id = {}
        for ext in zsv_predict.glob("*.ext"):
            by_id[ext.stem] = float(ext.read_text(encoding="utf-8").strip())

    pairs: list[tuple[str, str, float]] = []
    for seq_path in sorted(zsv_parsed.glob("*.ext")):
        rid = seq_path.stem
        if rid not in by_id:
            pred_path = zsv_predict / f"{sanitize_filename(rid)}.ext"
            if not pred_path.is_file():
                raise FileNotFoundError(f"ZSV predict missing for {rid}")
            by_id[rid] = float(pred_path.read_text(encoding="utf-8").strip())
        seq = seq_path.read_text(encoding="utf-8").strip().upper()
        pairs.append((rid, seq, float(by_id[rid])))
    if not pairs:
        raise ValueError(f"No ZSV sequences under {zsv_parsed}")
    return pairs


def metrics_from_preds(preds: list[float], targets: list[float]) -> dict[str, Any]:
    mse = _mse(preds, targets)
    return {
        "n": len(preds),
        "mse": mse,
        "rmse": math.sqrt(mse) if mse == mse else float("nan"),
        "mae": (
            sum(abs(p - t) for p, t in zip(preds, targets)) / len(preds)
            if preds
            else float("nan")
        ),
        "pearson": _pearson(preds, targets),
        "spearman": _spearman(preds, targets),
    }


def write_zsv_artifacts(
    *,
    out_json: Path,
    model: str,
    checkpoint: Path | str,
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Universal ZSV payload → ``zero_shot_metrics.json`` + jsonl/log sidecars."""
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "checkpoint": str(checkpoint),
        "split": "zero-shot-validation",
        "metrics": metrics,
    }
    if extra:
        payload.update(extra)
    ensure_dir(out_json.parent)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log_path = out_json.parent / "metrics.log"
    line = (
        f"zero-shot-validation n={metrics.get('n')} pearson={metrics.get('pearson')} "
        f"spearman={metrics.get('spearman')} mse={metrics.get('mse')} "
        f"rmse={metrics.get('rmse')}"
    )
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    jsonl = out_json.parent / "train_metrics.jsonl"
    zsv_rec = {"epoch": "final", "zero-shot-validation": metrics}
    with jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(zsv_rec, sort_keys=True) + "\n")
    return payload


def eval_legnet_zsv(
    *,
    model_dir: Path,
    parsed_root: Path,
    predict_root: Path,
    out_json: Path,
    batch_size: int = 256,
    device: int = 0,
) -> dict[str, Any]:
    """Run human_legnet checkpoint on ZSV sequences; write metrics JSON."""
    import sys
    import tempfile

    import lightning.pytorch as pl
    import torch
    from torch.utils.data import DataLoader

    pairs = load_zsv_pairs(parsed_root=parsed_root, predict_root=predict_root)
    model_dir = Path(model_dir)
    ckpt = model_dir / "final_model"
    ckpt_files = sorted(ckpt.glob("*.ckpt")) if ckpt.is_dir() else []
    if not ckpt_files:
        ckpt_files = sorted(model_dir.rglob("last_model-*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No LegNet checkpoint under {model_dir}")
    ckpt_path = max(ckpt_files, key=lambda p: p.stat().st_mtime)

    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"LegNet config.json missing: {config_path}")

    vendor = Path("software/human_legnet").resolve()
    if not vendor.is_dir():
        raise FileNotFoundError(f"human_legnet vendor missing: {vendor}")
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from fasta import FastaDataset  # type: ignore  # noqa: WPS433
    from trainer import LitModel  # type: ignore  # noqa: WPS433
    from training_config import TrainingConfig  # type: ignore  # noqa: WPS433

    cfg = TrainingConfig.from_json(config_path, training=False)
    map_loc = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    model = LitModel.load_from_checkpoint(str(ckpt_path), tr_cfg=cfg, map_location=map_loc)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=[device] if accelerator == "gpu" else 1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    with tempfile.TemporaryDirectory(prefix="zsv_legnet_") as tmp:
        fasta_path = Path(tmp) / "zsv.fa"
        with fasta_path.open("w", encoding="utf-8") as fh:
            for rid, seq, _ in pairs:
                fh.write(f">{rid}\n{seq}\n")
        dataset = FastaDataset(str(fasta_path), reverse=False)
        loader = DataLoader(dataset, batch_size=batch_size)
        y_preds = trainer.predict(model, dataloaders=loader)
        if not y_preds:
            raise RuntimeError("LegNet ZSV predict returned no batches")
        preds = torch.concat(y_preds).detach().float().cpu().reshape(-1).tolist()

    targets = [t for _, _, t in pairs]
    if len(preds) != len(targets):
        raise RuntimeError(
            f"ZSV pred/target length mismatch: {len(preds)} vs {len(targets)}"
        )

    metrics = metrics_from_preds([float(p) for p in preds], targets)
    return write_zsv_artifacts(
        out_json=out_json,
        model="legnet",
        checkpoint=ckpt_path,
        metrics=metrics,
    )


def _resolve_zsv_trees(split_root: Path) -> tuple[Path, Path] | None:
    """Find PARSED|FASTA + PREDICT roots that contain zero-shot-validation/."""
    split_root = Path(split_root)
    candidates = [
        (split_root / "FASTA", split_root / "PREDICT"),
        (split_root / "PARSED", split_root / "PREDICT"),
        (split_root.parent / "PARSED", split_root.parent / "PREDICT"),
        (split_root.parent / "FASTA", split_root.parent / "PREDICT"),
    ]
    for p_root, y_root in candidates:
        if (p_root / "zero-shot-validation").is_dir() and (
            y_root / "zero-shot-validation"
        ).is_dir():
            return p_root, y_root
    return None


def _read_run_config(outdir: Path) -> dict[str, Any]:
    for cand in (outdir / "logs" / "run_config.json", outdir / "run_config.json"):
        if cand.is_file():
            try:
                return json.loads(cand.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
    return {}


def eval_caduceus_zsv(
    *,
    model_dir: Path,
    parsed_root: Path,
    predict_root: Path,
    out_json: Path,
    batch_size: int = 192,
    max_length: int = 256,
    device: int = 0,
    amp: bool = True,
) -> dict[str, Any]:
    """Caduceus HF checkpoint on universal ZSV trees (same artifacts as LegNet)."""
    import importlib

    from src import caduceus

    # Long pipeline processes may have imported src.caduceus before evaluate_zsv_root
    # existed; reload once if the attribute is missing.
    if not hasattr(caduceus, "evaluate_zsv_root"):
        caduceus = importlib.reload(caduceus)

    return caduceus.evaluate_zsv_root(
        model_dir=model_dir,
        parsed_root=parsed_root,
        predict_root=predict_root,
        out_json=out_json,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        amp=amp,
    )


def eval_zsv_from_train_outdir(
    *,
    model: str,
    outdir: Path,
    split_root: Path,
    device: int = 0,
) -> dict[str, Any] | None:
    """Dispatch ZSV eval when ``SPLIT`` (or panel) has zero-shot-validation trees.

    Universal contract (LegNet + Caduceus):
      ``{PARSED|FASTA}/zero-shot-validation/*.ext`` + matching PREDICT
      → ``outdir/logs/zero_shot_metrics.json`` (+ jsonl/log sidecars).
    """
    split_root = Path(split_root)
    outdir = Path(outdir)
    trees = _resolve_zsv_trees(split_root)
    if trees is None:
        return None
    parsed, predict = trees

    out_json = outdir / "logs" / "zero_shot_metrics.json"
    model_l = model.lower()
    if model_l in {"legnet", "human_legnet"}:
        return eval_legnet_zsv(
            model_dir=outdir,
            parsed_root=parsed,
            predict_root=predict,
            out_json=out_json,
            device=device,
        )
    if model_l == "caduceus":
        model_dir = outdir / "final_model"
        if not (model_dir / "config.json").is_file():
            model_dir = outdir / "best_model"
        cfg = _read_run_config(outdir)
        max_length = int(cfg.get("max_length") or 256)
        batch_size = int(cfg.get("eval_batch_size") or cfg.get("batch_size") or 192)
        amp = bool(cfg.get("amp", True))
        return eval_caduceus_zsv(
            model_dir=model_dir,
            parsed_root=parsed,
            predict_root=predict,
            out_json=out_json,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            amp=amp,
        )
    raise ValueError(f"Unknown model for ZSV eval: {model}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate final model on ZSV trees")
    p.add_argument("--model", required=True, choices=["legnet", "human_legnet", "caduceus"])
    p.add_argument("--outdir", required=True, type=Path, help="Train run directory")
    p.add_argument("--split-root", required=True, type=Path, help="SPLIT or panel root")
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args(argv)
    result = eval_zsv_from_train_outdir(
        model=args.model,
        outdir=args.outdir,
        split_root=args.split_root,
        device=args.device,
    )
    if result is None:
        print("ERROR: zero-shot-validation trees not found", flush=True)
        return 2
    print(json.dumps(result["metrics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
