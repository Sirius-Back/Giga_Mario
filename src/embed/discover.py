"""Discover LegNet unif runs with usable checkpoints."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_BAD_RE = re.compile(r"_BAD_", re.IGNORECASE)
_FOLD_RE = re.compile(r"^fold(\d+)$")
_KEY_FOLD_RE = re.compile(r"/fold(\d+)$")


@dataclass(frozen=True)
class LegNetRun:
    """One train unit (flat run or LOO fold)."""

    run_name: str
    fold: int | None  # None = non-LOO; else fold index
    root: Path  # run root (or fold root)
    train_dir: Path  # …/direct
    split_csv: Path
    legnet_tsv: Path
    config_json: Path
    ckpt_path: Path
    best_meta: Path

    @property
    def key(self) -> str:
        if self.fold is None:
            return self.run_name
        return f"{self.run_name}/fold{self.fold}"


def _resolve_ckpt(train_dir: Path) -> Path | None:
    for sub in ("final_model", "best_model"):
        d = train_dir / sub
        if not d.is_dir():
            continue
        ckpts = sorted(d.glob("*.ckpt"))
        if ckpts:
            # Prefer pearson-* when present
            pearson = [p for p in ckpts if p.name.startswith("pearson-")]
            return max(pearson or ckpts, key=lambda p: p.stat().st_mtime)
    return None


def _try_unit(
    *,
    run_name: str,
    fold: int | None,
    unit_root: Path,
) -> LegNetRun | None:
    train_dir = unit_root / "direct"
    split_csv = unit_root / "split.csv"
    legnet_tsv = unit_root / "legnet_input" / "all.tsv"
    config_json = train_dir / "config.json"
    best_meta = train_dir / "best_model" / "best_meta.json"
    if not train_dir.is_dir():
        return None
    ckpt = _resolve_ckpt(train_dir)
    if ckpt is None:
        return None
    if not config_json.is_file():
        return None
    if not split_csv.is_file() or not legnet_tsv.is_file():
        return None
    return LegNetRun(
        run_name=run_name,
        fold=fold,
        root=unit_root.resolve(),
        train_dir=train_dir.resolve(),
        split_csv=split_csv.resolve(),
        legnet_tsv=legnet_tsv.resolve(),
        config_json=config_json.resolve(),
        ckpt_path=ckpt.resolve(),
        best_meta=best_meta.resolve() if best_meta.is_file() else best_meta,
    )


def filter_loo_runs(
    runs: list[LegNetRun], loo_fold: int | None
) -> list[LegNetRun]:
    """Keep only ``fold==loo_fold`` for LOO units; pass through non-LOO.

    ``loo_fold=None`` keeps every fold (debug / full LOO tables).
    Default publication figures use ``loo_fold=0`` (first CV fold).
    """
    if loo_fold is None:
        return list(runs)
    out: list[LegNetRun] = []
    for r in runs:
        if r.fold is None or r.fold == loo_fold:
            out.append(r)
    return out


def discover_legnet_runs(
    runs_root: Path, *, loo_fold: int | None = None
) -> list[LegNetRun]:
    """Scan ``runs_unif/legnet`` for extractable units; skip ``*_BAD_*``.

    Symlinks are followed (e.g. ``run5_legnet_hashfrag`` → ``runs/run5``).
    When ``loo_fold`` is set, only that LOO fold index is kept.
    """
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        raise FileNotFoundError(f"LegNet runs root missing: {runs_root}")

    found: list[LegNetRun] = []
    for child in sorted(runs_root.iterdir()):
        # Allow symlink dirs (legacy hashFrag run5 → runs_unif/legnet)
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
