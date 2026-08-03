"""Completeness checks for Caduceus run units."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.embed.discover_caduceus import CaduceusRun
from src.pipeline.common import SPLIT_CSV_COLUMNS


@dataclass
class CaduceusValidationResult:
    key: str
    status: str  # READY | SKIPPED | FAILED
    reasons: list[str] = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0
    n_val: int = 0
    max_length: int | None = None
    model_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _count_labels(path: Path) -> int:
    with path.open(encoding="utf-8") as fh:
        n = sum(1 for _ in fh) - 1
    return max(0, n)


def validate_caduceus_run(
    run: CaduceusRun, *, load_model: bool = False
) -> CaduceusValidationResult:
    reasons: list[str] = []
    n_train = _count_labels(run.splits_dir / "train" / "labels.tsv")
    n_test = _count_labels(run.splits_dir / "test" / "labels.tsv")
    n_val = _count_labels(run.splits_dir / "val" / "labels.tsv")
    if n_train < 100:
        reasons.append(f"train too small: {n_train}")
    if n_test < 32:
        reasons.append(f"test too small: {n_test}")
    if n_val < 32:
        reasons.append(f"val too small: {n_val}")

    try:
        with run.split_csv.open(encoding="utf-8") as fh:
            header = fh.readline().rstrip("\n").split("|")
        missing = [c for c in SPLIT_CSV_COLUMNS if c not in header]
        if missing:
            reasons.append(f"split.csv missing columns {missing}")
    except OSError as exc:
        reasons.append(f"split.csv: {exc}")

    cfg_path = run.model_dir / "config.json"
    if not cfg_path.is_file():
        reasons.append("model config.json missing")
    else:
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if int(cfg.get("d_model", 0)) not in (256, 0) and "d_model" in cfg:
                pass  # allow other d_model
        except json.JSONDecodeError as exc:
            reasons.append(f"config.json: {exc}")

    if load_model:
        try:
            from src.embed.caduceus_extract import load_caduceus_model

            model, _tok, _ = load_caduceus_model(run.model_dir, device="cpu")
            del model
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"model load: {type(exc).__name__}: {exc}")

    status = "READY" if not reasons else "FAILED"
    return CaduceusValidationResult(
        key=run.key,
        status=status,
        reasons=reasons,
        n_train=n_train,
        n_test=n_test,
        n_val=n_val,
        max_length=run.max_length,
        model_dir=str(run.model_dir),
    )


def validate_all_caduceus(
    runs: list[CaduceusRun], *, load_model: bool = False
) -> list[CaduceusValidationResult]:
    return [validate_caduceus_run(r, load_model=load_model) for r in runs]


def write_validation_report(
    results: list[CaduceusValidationResult], path: Path
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n": len(results),
        "READY": sum(1 for r in results if r.status == "READY"),
        "FAILED": sum(1 for r in results if r.status == "FAILED"),
        "results": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
