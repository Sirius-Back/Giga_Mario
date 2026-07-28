"""Dispatch pipeline SPLIT artifacts to the real Caduceus or LegNet trainers."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

from .common import (
    assert_matching_artifact_ids,
    ensure_dir,
    index_unique_predict_rows,
    read_csv,
)


FOLD_MAP = {"TRAIN": "train", "VAL": "val", "TEST": "test"}


def _link_or_copy(source: Path, destination: Path) -> None:
    """Hardlink real artifacts, falling back only when filesystems differ."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _read_predict(path: Path) -> dict[str, dict[str, str]]:
    """Load predict.csv with a unique ``id`` column (required)."""
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty prediction table: {path}")
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"{path} contains no usable prediction IDs")
    if "id" not in rows[0]:
        raise ValueError(f"{path} must contain an 'id' column")
    if "predict_var1" not in rows[0]:
        raise ValueError(f"{path} must contain a 'predict_var1' column")
    return index_unique_predict_rows(rows, label=str(path))


def _validate_legacy_caduceus_split(folder: Path) -> None:
    for fold in ("train", "val", "test"):
        labels = folder / fold / "labels.tsv"
        sequences = folder / fold / "sequences"
        if not labels.is_file() or not sequences.is_dir():
            raise FileNotFoundError(
                f"Invalid Caduceus layout: need {labels} and {sequences}/"
            )


def adapt_split_for_caduceus(
    folders: Path, *, outdir: Path, task_type: str
) -> tuple[Path, dict[str, int]]:
    """Convert a universal ``SPLIT`` tree into Caduceus's labels/sequences layout.

    Requires unique matching IDs across ``FASTA/*.ext``, ``PREDICT/*.ext``, and
    ``predict.csv`` ``id`` (merged region IDs or mapped composite IDs).
    """
    folders = Path(folders)
    if (folders / "train" / "labels.tsv").is_file():
        _validate_legacy_caduceus_split(folders)
        return folders, {
            fold: sum(1 for _ in (folders / fold / "labels.tsv").open(encoding="utf-8")) - 1
            for fold in ("train", "val", "test")
        }

    if not (folders / "FASTA").is_dir() or not (folders / "PREDICT").is_dir():
        raise FileNotFoundError(
            f"Expected SPLIT/FASTA and SPLIT/PREDICT trees under {folders}"
        )
    if task_type not in {"regression", "classification"}:
        raise ValueError("type must be regression or classification for Caduceus")

    adapted = outdir / "caduceus_input"
    if adapted.exists():
        shutil.rmtree(adapted)
    counts: dict[str, int] = {}
    for source_fold, fold in FOLD_MAP.items():
        fasta_dir = folders / "FASTA" / source_fold
        predict_dir = folders / "PREDICT" / source_fold
        predict_path = predict_dir / "predict.csv"
        if not fasta_dir.exists() and not predict_path.exists():
            predict_rows: dict[str, dict[str, str]] = {}
            sequence_files: list[Path] = []
        elif not fasta_dir.is_dir() or not predict_path.is_file():
            raise FileNotFoundError(
                f"Incomplete {source_fold} bucket: need {fasta_dir}/ and {predict_path}"
            )
        else:
            predict_rows = _read_predict(predict_path)
            assert_matching_artifact_ids(
                fasta_dir=fasta_dir,
                predict_dir=predict_dir,
                predict_by_id=predict_rows,
                bucket=source_fold,
            )
            sequence_files = sorted(fasta_dir.glob("*.ext"))

        labels: list[tuple[str, str]] = []
        for source in sequence_files:
            uid = source.stem
            row = predict_rows[uid]
            value = row["predict_var1"].strip()
            try:
                float(value)
            except ValueError as exc:
                raise ValueError(
                    f"{source_fold} prediction for {uid} is not numeric: {value!r}"
                ) from exc
            if task_type == "classification" and float(value) != int(float(value)):
                raise ValueError(
                    "Classification requires integer predict_var1 labels; "
                    f"{uid} has {value!r}"
                )
            _link_or_copy(source, adapted / fold / "sequences" / f"{uid}.txt")
            labels.append((uid, value))

        labels_path = adapted / fold / "labels.tsv"
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        label_name = "TPM" if task_type == "regression" else "label"
        with labels_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("sample_id", label_name))
            writer.writerows(labels)
        counts[fold] = len(labels)
    return adapted, counts


def materialize_tiny_split(
    source_split: Path, *, outdir: Path, counts: dict[str, int]
) -> Path:
    """Hardlink a deterministic real subset of a universal ``SPLIT`` tree."""
    source_split = Path(source_split)
    if not (source_split / "FASTA").is_dir() or not (source_split / "PREDICT").is_dir():
        raise FileNotFoundError(f"Expected SPLIT tree at {source_split}")
    destination = Path(outdir) / "SPLIT"
    if destination.exists():
        shutil.rmtree(destination)
    for source_fold, fold in FOLD_MAP.items():
        requested = counts[fold]
        if requested <= 0:
            raise ValueError(f"tiny subset count for {fold} must be positive")
        source_fasta = source_split / "FASTA" / source_fold
        source_predict_dir = source_split / "PREDICT" / source_fold
        source_predict = source_predict_dir / "predict.csv"
        predictions = _read_predict(source_predict)
        assert_matching_artifact_ids(
            fasta_dir=source_fasta,
            predict_dir=source_predict_dir,
            predict_by_id=predictions,
            bucket=source_fold,
        )
        files = sorted(source_fasta.glob("*.ext"))
        if len(files) < requested:
            raise ValueError(
                f"{source_fold} only has {len(files)} real sequences; requested {requested}"
            )
        selected = files[:requested]
        rows: list[dict[str, str]] = []
        fieldnames = list(next(iter(predictions.values())).keys())
        for file in selected:
            row = predictions[file.stem]
            _link_or_copy(file, destination / "FASTA" / source_fold / file.name)
            source_prediction = source_predict_dir / file.name
            if not source_prediction.is_file():
                raise FileNotFoundError(source_prediction)
            _link_or_copy(
                source_prediction, destination / "PREDICT" / source_fold / file.name
            )
            rows.append(row)
        index_unique_predict_rows(rows, label=f"tiny {source_fold} predict.csv")
        with (destination / "PREDICT" / source_fold / "predict.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, delimiter="|", lineterminator="\n",
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
    return destination


def _validate_legnet_tsv(path: Path) -> None:
    text = path.read_text(encoding="utf-8").splitlines()
    if not text:
        raise ValueError(f"Empty LegNet TSV: {path}")
    header = text[0].split("\t")
    need = {"seq_id", "seq", "mean_value", "fold", "rev"}
    if not need.issubset(set(header)):
        raise ValueError(f"LegNet TSV missing columns; have {header}")


def run_train(
    *,
    model: str,
    type: str,
    folders: Path,
    outdir: Path,
    strategy: str = "random",
    smoke: bool = False,
    epochs: int = 1,
    max_samples: int | None = None,
    batch_size: int = 1,
    max_length: int = 512,
    seed: int = 42,
    n_devices: int = 1,
    num_workers: int = 8,
    legnet_demo: bool = False,
    zsv_root: Path | None = None,
    eval_zsv: bool = False,
) -> Path:
    """
    Wrapper around Caduceus / LegNet trainers.

    ``smoke=True`` validates a real materialized split and emits only structural
    counts. It never records synthetic losses or quality metrics.

    When ``eval_zsv=True``, evaluates the **final** checkpoint on
    ``{zsv_root}/PARSED|PREDICT/zero-shot-validation`` (required if True).
    """
    model = model.lower()
    type = type.lower()
    folders = Path(folders)
    outdir = ensure_dir(Path(outdir))

    if eval_zsv:
        if zsv_root is None:
            raise ValueError("eval_zsv=True requires zsv_root (panel outdir with ZSV trees)")
        zsv_root = Path(zsv_root)
        parsed_z = zsv_root / "PARSED" / "zero-shot-validation"
        pred_z = zsv_root / "PREDICT" / "zero-shot-validation"
        if not parsed_z.is_dir() or not pred_z.is_dir():
            raise FileNotFoundError(
                "ZSV requested but trees missing: "
                f"need {parsed_z} and {pred_z}"
            )

    if model == "caduceus":
        caduceus_input, counts = adapt_split_for_caduceus(
            folders, outdir=outdir, task_type=type
        )
    elif model in {"legnet", "human_legnet"}:
        tsv = folders if folders.is_file() else folders / "all.tsv"
        if not tsv.is_file():
            # allow building from SPLIT: require at least one parsed ext
            if not list(folders.rglob("*.ext")):
                raise FileNotFoundError(f"LegNet input not found under {folders}")
        else:
            _validate_legnet_tsv(tsv)
    else:
        raise ValueError(f"Unknown model {model!r}")

    if not smoke:
        if model == "caduceus":
            from src import caduceus

            argv = [
                "--splits-dir", str(caduceus_input),
                "--out", str(outdir),
                "--epochs", str(epochs),
                "--batch-size", str(batch_size),
                "--eval-batch-size", str(batch_size),
                "--max-length", str(max_length),
                "--seed", str(seed),
                "--task", type,
            ]
            if max_samples is not None:
                argv += ["--max-samples", str(max_samples)]
            if caduceus.main(argv) != 0:
                raise RuntimeError("src.caduceus returned a non-zero status")
            if eval_zsv:
                from .zsv_eval import eval_zsv_from_train_outdir

                result = eval_zsv_from_train_outdir(
                    model=model, outdir=outdir, split_root=zsv_root  # type: ignore[arg-type]
                )
                if result is None:
                    raise RuntimeError("ZSV eval requested but produced no metrics")
            return outdir

        from src import legnet

        tsv = folders if folders.is_file() else folders / "all.tsv"
        argv = [
            "--data-path", str(tsv), "--out", str(outdir),
            "--epochs", str(epochs), "--seed", str(seed),
            "--n-devices", str(n_devices),
            "--train-batch-size", str(batch_size),
            "--valid-batch-size", str(batch_size),
            "--num-workers", str(num_workers),
        ]
        if max_samples is not None:
            raise ValueError("LegNet does not support pipeline --max-samples")
        if legnet_demo:
            argv.append("--demo")
        if legnet.main(argv) != 0:
            raise RuntimeError("src.legnet returned a non-zero status")
        if eval_zsv:
            from .zsv_eval import eval_zsv_from_train_outdir

            result = eval_zsv_from_train_outdir(
                model=model, outdir=outdir, split_root=zsv_root  # type: ignore[arg-type]
            )
            if result is None:
                raise RuntimeError("ZSV eval requested but produced no metrics")
        return outdir

    logs = ensure_dir(outdir / "logs")
    metrics = logs / "train_metrics.jsonl"
    rows: list[dict[str, Any]] = [{"epoch": 0, "smoke": True, "model": model,
                                    "type": type, "strategy": strategy,
                                    "splits": counts if model == "caduceus" else {}}]
    with metrics.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (outdir / "run_config.json").write_text(
        json.dumps({
            "model": model, "type": type, "folders": str(folders),
            "smoke": True, "strategy": strategy,
            "metrics_interpretation": "structural only; not model-quality metrics",
            "caduceus_input": str(caduceus_input) if model == "caduceus" else None,
            "eval_zsv": eval_zsv,
            "zsv_root": str(zsv_root) if zsv_root else None,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return outdir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Dispatch pipeline inputs to real trainers")
    p.add_argument("--model", required=True, choices=["caduceus", "legnet", "human_legnet"])
    p.add_argument("--type", default="regression")
    p.add_argument("--folders", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--strategy", default="random")
    p.add_argument("--smoke", action="store_true", help="Validate real input; write structural log only")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-devices", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument(
        "--legnet-demo",
        action="store_true",
        help="Use human_legnet's single fold-1 test / fold-2 validation run.",
    )
    p.add_argument(
        "--eval-zsv",
        action="store_true",
        help="After train, evaluate final model on panel ZSV trees",
    )
    p.add_argument(
        "--zsv-root",
        type=Path,
        default=None,
        help="Panel outdir containing PARSED/PREDICT/zero-shot-validation",
    )
    p.add_argument(
        "--tiny-outdir", type=Path, default=None,
        help="Materialize a deterministic 50/10/10 real SPLIT subset and exit",
    )
    p.add_argument("--tiny-train", type=int, default=50)
    p.add_argument("--tiny-val", type=int, default=10)
    p.add_argument("--tiny-test", type=int, default=10)
    args = p.parse_args(argv)
    if args.tiny_outdir is not None:
        if args.model != "caduceus":
            p.error("--tiny-outdir currently supports Caduceus SPLIT trees only")
        print(materialize_tiny_split(
            args.folders, outdir=args.tiny_outdir,
            counts={"train": args.tiny_train, "val": args.tiny_val, "test": args.tiny_test},
        ))
        return 0
    print(run_train(
        model=args.model, type=args.type, folders=args.folders, outdir=args.outdir,
        strategy=args.strategy, smoke=args.smoke, epochs=args.epochs,
        max_samples=args.max_samples, batch_size=args.batch_size,
        max_length=args.max_length, seed=args.seed, n_devices=args.n_devices,
        num_workers=args.num_workers, legnet_demo=args.legnet_demo,
        zsv_root=args.zsv_root, eval_zsv=args.eval_zsv,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
