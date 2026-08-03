"""Discover Caduceus unif runs with usable HF checkpoints."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.embed.discover import filter_loo_runs

_BAD_RE = re.compile(r"_BAD_|_ARCHIVED_|_FAILED_", re.IGNORECASE)
_FOLD_RE = re.compile(r"^fold(\d+)$")


@dataclass(frozen=True)
class CaduceusRun:
    """One Caduceus train unit (flat run or LOO fold)."""

    run_name: str
    fold: int | None
    root: Path
    train_dir: Path  # …/direct
    model_dir: Path  # …/direct/best_model (HF)
    split_csv: Path
    splits_dir: Path  # …/direct/caduceus_input
    run_config: Path
    max_length: int

    @property
    def key(self) -> str:
        if self.fold is None:
            return self.run_name
        return f"{self.run_name}/fold{self.fold}"

    # Compatibility aliases used by shared store helpers
    @property
    def ckpt_path(self) -> Path:
        return self.model_dir

    @property
    def config_json(self) -> Path:
        return self.model_dir / "config.json"

    @property
    def legnet_tsv(self) -> Path:
        """Unused for Caduceus; kept for store typing symmetry."""
        return self.splits_dir


def _resolve_model_dir(train_dir: Path) -> Path | None:
    for sub in ("best_model", "final_model"):
        d = train_dir / sub
        if (d / "model.safetensors").is_file() or (d / "pytorch_model.bin").is_file():
            if (d / "config.json").is_file():
                return d.resolve()
    return None


def _max_length_from_config(run_config: Path, model_dir: Path) -> int:
    if run_config.is_file():
        try:
            cfg = json.loads(run_config.read_text(encoding="utf-8"))
            if "max_length" in cfg:
                return int(cfg["max_length"])
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return 256


def _try_unit(
    *, run_name: str, fold: int | None, unit_root: Path
) -> CaduceusRun | None:
    train_dir = unit_root / "direct"
    split_csv = unit_root / "split.csv"
    splits_dir = train_dir / "caduceus_input"
    run_config = train_dir / "run_config.json"
    if not train_dir.is_dir():
        return None
    model_dir = _resolve_model_dir(train_dir)
    if model_dir is None:
        return None
    if not split_csv.is_file():
        return None
    if not splits_dir.is_dir():
        return None
    for role in ("train", "test", "val"):
        if not (splits_dir / role / "labels.tsv").is_file():
            return None
        if not (splits_dir / role / "sequences").is_dir():
            return None
    return CaduceusRun(
        run_name=run_name,
        fold=fold,
        root=unit_root.resolve(),
        train_dir=train_dir.resolve(),
        model_dir=model_dir,
        split_csv=split_csv.resolve(),
        splits_dir=splits_dir.resolve(),
        run_config=run_config.resolve(),
        max_length=_max_length_from_config(run_config, model_dir),
    )


def discover_caduceus_runs(
    runs_root: Path, *, loo_fold: int | None = None
) -> list[CaduceusRun]:
    """Scan ``runs_unif/caduceus``; skip BAD/ARCHIVED/FAILED; optional LOO filter."""
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        raise FileNotFoundError(f"Caduceus runs root missing: {runs_root}")

    found: list[CaduceusRun] = []
    for child in sorted(runs_root.iterdir()):
        if not child.is_dir():
            continue
        if _BAD_RE.search(child.name):
            continue
        run_name = child.name
        fold_dirs = sorted(
            p for p in child.iterdir() if p.is_dir() and _FOLD_RE.match(p.name)
        )
        if fold_dirs:
            for fd in fold_dirs:
                m = _FOLD_RE.match(fd.name)
                assert m is not None
                unit = _try_unit(
                    run_name=run_name, fold=int(m.group(1)), unit_root=fd
                )
                if unit is not None:
                    found.append(unit)
            continue
        unit = _try_unit(run_name=run_name, fold=None, unit_root=child)
        if unit is not None:
            found.append(unit)

    return filter_loo_runs(found, loo_fold)
