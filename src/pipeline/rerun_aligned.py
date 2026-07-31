"""Rerun-with-different-strategy helpers for Hydra ``/pipeline``.

Aligned reproducibility suite under ``runs_aligned/``:

* locate a prior ``split.csv`` (+ optional intermediates) from a user hint
* reuse those folds **or** build a fresh train/test/val at ~3:1:1 only
* stage artifacts into a **new** out root without mutating the source
* refuse to overwrite existing run artifacts
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .common import SPLIT_CSV_COLUMNS, ensure_dir, read_csv, write_csv


# Relative paths commonly produced beside split.csv (copied/linked when present).
INTERMEDIATE_RELPATHS: tuple[str, ...] = (
    "split.csv",
    "SPLIT",
    "split_predict_summary.json",
    "split_summary.json",
    "hashfrag",
    "kmer_features",
    "gc_features",
    "features",
    "MARKED_pangenome",
    "MARKED_blastp",
    "MARKED_parsed",
    "figures/sbs",
    "figures/split",
)

# Destinations that must not already exist under a rerun out_root.
PROTECTED_OUT_RELPATHS: tuple[str, ...] = (
    "split.csv",
    "SPLIT",
    "direct",
    "adversarial",
    "legnet_input",
    "hydra_resolved_config.yaml",
    "rerun_manifest.json",
)

# train:test:val ≈ 3:1:1 → fractions 0.6 / 0.2 / 0.2
ALIGNED_RATIOS: tuple[float, float, float] = (3.0, 1.0, 1.0)
_RATIO_FRAC_TOL = 0.05

# Default train schedule for aligned / unif reruns.
ALIGNED_EPOCHS = 30
ALIGNED_MIN_EPOCHS = 15
ALIGNED_EARLY_STOPPING_PATIENCE = 10
ALIGNED_EVAL_MAX_SAMPLES = 8192
ALIGNED_ADV_EPOCHS = 10
ALIGNED_ADV_MIN_EPOCHS = 0
ALIGNED_ADV_EARLY_STOPPING_PATIENCE = 5


def unif_run_name(
    run_i: int | str,
    model: str,
    split: str,
    split_params: str | None = None,
) -> str:
    """``run{i}_{model}_{split}[_{params}]`` (no empty param suffix)."""
    base = f"run{run_i}_{model}_{split}"
    if split_params:
        return f"{base}_{split_params}"
    return base


def unif_out_root(
    run_i: int | str,
    model: str,
    split: str,
    *,
    project_root: Path,
    split_params: str | None = None,
) -> Path:
    """``runs_unif/{model}/run{i}_{model}_{split}[_params]``."""
    name = unif_run_name(run_i, model, split, split_params)
    return Path(project_root) / "runs_unif" / str(model) / name


def unif_src_dir(
    run_i: int | str,
    model: str,
    split: str,
    *,
    project_root: Path,
    split_params: str | None = None,
) -> Path:
    """``src/runs_unif/run{i}_{model}_{split}[_params]``."""
    name = unif_run_name(run_i, model, split, split_params)
    return Path(project_root) / "src" / "runs_unif" / name


def legacy_run_roots(run_i: int | str, *, project_root: Path) -> tuple[Path, Path]:
    """Return ``(src/runs/run{i}, runs/run{i})``."""
    root = Path(project_root)
    return root / "src" / "runs" / f"run{run_i}", root / "runs" / f"run{run_i}"


def count_train_test_val(rows: Sequence[dict[str, str]]) -> dict[str, int]:
    counts = {"train": 0, "test": 0, "val": 0, "zsv": 0, "other": 0}
    for row in rows:
        label = str(row.get("train_test", "")).strip().lower()
        if label in counts:
            counts[label] += 1
        elif label in {"zeroshotvalidation", "zero-shot", "zeroshot"}:
            counts["zsv"] += 1
        else:
            counts["other"] += 1
    return counts


def rewrite_split_table_aligned(
    source_split_csv: Path,
    dest_split_csv: Path,
    *,
    seed: int = 42,
    prefer_label_swap: bool = True,
) -> dict[str, Any]:
    """Rebuild train/test/val on the present ID table at ≈3:1:1; keep ZSV rows.

    When the present table is already ≈1:1:3 (legacy inverted ``ratios=[1,1,3]``),
    swap **train ↔ val** labels (deterministic; preserves ID membership of the
    large pool → train). Otherwise reassign assignable IDs with
    ``ratios=(3,1,1)``. Never mutates ``source_split_csv``.
    """
    from src.pipeline.generate_fold import is_zsv_fold
    from src.splits.common import assign_folds_random

    source_split_csv = Path(source_split_csv)
    dest_split_csv = Path(dest_split_csv)
    if dest_split_csv.resolve() == source_split_csv.resolve():
        raise FileExistsError(
            "rewrite_split_table_aligned refuses to overwrite the source split.csv"
        )
    if dest_split_csv.exists():
        raise FileExistsError(f"Refusing overwrite: {dest_split_csv}")

    rows = read_csv(source_split_csv)
    if not rows:
        raise ValueError(f"split.csv is empty: {source_split_csv}")
    required = {"ID", "train_test", "fold"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            f"split.csv missing columns {sorted(missing)}: {source_split_csv}"
        )

    before = count_train_test_val(rows)
    assignable_idx = [
        i
        for i, row in enumerate(rows)
        if not is_zsv_fold(row["train_test"]) and not is_zsv_fold(row.get("fold", ""))
    ]
    method: str
    # Detect legacy inverted 1:1:3 (train:test:val) → fix via train↔val swap.
    ttv = (before["train"], before["test"], before["val"])
    if prefer_label_swap and sum(ttv) > 0 and is_aligned_ratios(
        (ttv[2], ttv[1], ttv[0])  # val, test, train ≈ 3:1:1 when current is 1:1:3
    ):
        method = "swap_train_val"
        out_rows: list[dict[str, str]] = []
        for row in rows:
            label = str(row["train_test"]).strip().lower()
            new_label = label
            if label == "train":
                new_label = "val"
            elif label == "val":
                new_label = "train"
            out_rows.append(
                {
                    "ID": row["ID"],
                    "train_test": new_label if label in {"train", "val", "test"} else row["train_test"],
                    "fold": row["fold"],
                }
            )
    else:
        method = "reassign_random_3_1_1"
        import random as _random

        rng = _random.Random(int(seed))
        order = list(assignable_idx)
        rng.shuffle(order)
        folds = assign_folds_random(len(order), ratios=ALIGNED_RATIOS)
        assigned = {idx: lab for idx, lab in zip(order, folds)}
        out_rows = []
        for i, row in enumerate(rows):
            if i in assigned:
                out_rows.append(
                    {
                        "ID": row["ID"],
                        "train_test": assigned[i],
                        "fold": row["fold"],
                    }
                )
            else:
                out_rows.append(
                    {
                        "ID": row["ID"],
                        "train_test": "zsv",
                        "fold": "zsv",
                    }
                )

    dest_split_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(dest_split_csv, out_rows, SPLIT_CSV_COLUMNS)
    after = count_train_test_val(out_rows)
    if not is_aligned_ratios((after["train"], after["test"], after["val"])):
        raise RuntimeError(
            f"aligned rewrite failed to reach ≈3:1:1: before={before} after={after} "
            f"method={method}"
        )
    return {
        "method": method,
        "source_split_csv": str(source_split_csv),
        "dest_split_csv": str(dest_split_csv),
        "counts_before": before,
        "counts_after": after,
        "seed": seed,
        "ratios": list(ALIGNED_RATIOS),
    }


@dataclass(frozen=True)
class SourceArtifacts:
    """Resolved prior-run artifacts for a rerun."""

    root: Path
    split_csv: Path
    intermediates: dict[str, Path] = field(default_factory=dict)


def parse_override_keys(overrides: Sequence[str] | None) -> set[str]:
    """Return top-level Hydra override keys (``epochs`` from ``epochs=30``)."""
    keys: set[str] = set()
    for tok in overrides or ():
        token = str(tok).lstrip("+~")
        if "=" not in token:
            continue
        key = token.split("=", 1)[0]
        # Nested keys: train.foo → train (top-level group); keep full for exact match.
        keys.add(key.split(".", 1)[0])
        keys.add(key)
    return keys


def is_aligned_ratios(
    ratios: tuple[float, float, float] | Sequence[float],
    *,
    tol: float = _RATIO_FRAC_TOL,
) -> bool:
    """True when train:test:val fractions match ~3:1:1 within ``tol``."""
    vals = tuple(float(x) for x in ratios)
    if len(vals) != 3 or any(v <= 0 for v in vals):
        return False
    total = sum(vals)
    fracs = tuple(v / total for v in vals)
    target = tuple(v / sum(ALIGNED_RATIOS) for v in ALIGNED_RATIOS)
    return all(abs(a - b) <= tol for a, b in zip(fracs, target, strict=True))


def require_aligned_ratios(
    ratios: tuple[float, float, float] | Sequence[float] | None,
) -> tuple[float, float, float]:
    """Return validated ~3:1:1 ratios (default ALIGNED_RATIOS)."""
    if ratios is None:
        return ALIGNED_RATIOS
    vals = tuple(float(x) for x in ratios)
    if not is_aligned_ratios(vals):
        raise ValueError(
            "rerun without source_split requires train:test:val ≈ 3:1:1 "
            f"(got {vals!r}; allowed e.g. [3,1,1] or [0.6,0.2,0.2])"
        )
    return vals  # type: ignore[return-value]


def resolve_source_artifacts(hint: str | Path, *, project_root: Path | None = None) -> SourceArtifacts:
    """Locate ``split.csv`` and optional intermediates from a user hint.

    ``hint`` may be:

    * path to ``split.csv``
    * prior run root containing ``split.csv``
    * relative path resolved against ``project_root`` (default: cwd)
    """
    root = Path(project_root) if project_root is not None else Path.cwd()
    raw = Path(hint)
    path = raw if raw.is_absolute() else (root / raw)
    path = path.resolve()

    if path.is_file() and path.name == "split.csv":
        split_csv = path
        run_root = path.parent
    elif path.is_dir():
        split_csv = path / "split.csv"
        run_root = path
        if not split_csv.is_file():
            # Common nesting: runs/<id>/… or …/direct not used for split.csv
            alt = path / "SPLIT"
            raise FileNotFoundError(
                f"No split.csv under {path} "
                f"(expected {split_csv}"
                + (f"; SPLIT exists but split.csv missing" if alt.is_dir() else "")
                + "). Pass source_split=…/split.csv or the run root that contains it."
            )
    else:
        raise FileNotFoundError(
            f"source_split hint not found as file or directory: {path}"
        )

    if not split_csv.is_file():
        raise FileNotFoundError(f"Missing split.csv: {split_csv}")

    rows = read_csv(split_csv)
    if not rows:
        raise ValueError(f"split.csv is empty: {split_csv}")
    required = {"ID", "train_test", "fold"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"split.csv missing columns {sorted(missing)}: {split_csv}")

    intermediates: dict[str, Path] = {"split.csv": split_csv}
    for rel in INTERMEDIATE_RELPATHS:
        if rel == "split.csv":
            continue
        candidate = run_root / rel
        if candidate.exists():
            intermediates[rel] = candidate

    return SourceArtifacts(root=run_root, split_csv=split_csv, intermediates=intermediates)


def assert_fresh_out_root(out_root: Path) -> None:
    """Fail if ``out_root`` already has protected pipeline artifacts."""
    out_root = Path(out_root)
    conflicts = [
        rel for rel in PROTECTED_OUT_RELPATHS if (out_root / rel).exists()
    ]
    if conflicts:
        raise FileExistsError(
            "rerun refuses to overwrite existing artifacts under "
            f"{out_root}: {', '.join(conflicts)}. Choose a new run_id / out_root."
        )


def _link_or_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing overwrite: {destination}")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _link_or_copy_tree(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing overwrite: {destination}")
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(source)
        target = destination / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            _link_or_copy_file(path, target)


def stage_source_into_out_root(
    source: SourceArtifacts,
    out_root: Path,
    *,
    copy_intermediates: bool = True,
    include_split_tree: bool = False,
) -> dict[str, Any]:
    """Copy/link prior split (+ intermediates) into a fresh ``out_root``.

    Always **copies** ``split.csv`` (never hardlinks) so later writes cannot
    mutate the source inode. Large trees are hardlinked when possible.

    ``SPLIT/`` is skipped by default — callers rematerialize into the new
    out_root so ZSV trees stay consistent with the panel.
    """
    out_root = ensure_dir(Path(out_root))
    assert_fresh_out_root(out_root)

    dest_split = out_root / "split.csv"
    # Byte-copy so source and dest never share an inode.
    shutil.copy2(source.split_csv, dest_split)

    staged: dict[str, str] = {"split.csv": str(dest_split)}
    if copy_intermediates:
        for rel, src in source.intermediates.items():
            if rel == "split.csv":
                continue
            if rel == "SPLIT" and not include_split_tree:
                continue
            dest = out_root / rel
            if dest.exists():
                raise FileExistsError(f"Refusing overwrite: {dest}")
            if src.is_dir():
                _link_or_copy_tree(src, dest)
            elif src.is_file():
                _link_or_copy_file(src, dest)
            staged[rel] = str(dest)

    return {
        "source_root": str(source.root),
        "source_split_csv": str(source.split_csv),
        "staged": staged,
        "reuse_folds": True,
        "include_split_tree": include_split_tree,
    }


def write_rerun_manifest(out_root: Path, payload: dict[str, Any]) -> Path:
    """Write ``rerun_manifest.json`` (fails if already present)."""
    out_root = Path(out_root)
    path = out_root / "rerun_manifest.json"
    if path.exists():
        raise FileExistsError(f"Refusing overwrite: {path}")
    out_root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def apply_rerun_schedule(
    *,
    epochs: int,
    min_epochs: int,
    early_stopping_patience: int,
    overridden: Iterable[str],
) -> tuple[int, int, int]:
    """Apply aligned epoch defaults unless the key was CLI-overridden."""
    keys = set(overridden)
    if "epochs" not in keys:
        epochs = ALIGNED_EPOCHS
    if "min_epochs" not in keys:
        min_epochs = ALIGNED_MIN_EPOCHS
    if "early_stopping_patience" not in keys:
        early_stopping_patience = ALIGNED_EARLY_STOPPING_PATIENCE
    if int(epochs) < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if int(min_epochs) < 0:
        raise ValueError(f"min_epochs must be >= 0, got {min_epochs}")
    if int(early_stopping_patience) < 0:
        raise ValueError(
            f"early_stopping_patience must be >= 0, got {early_stopping_patience}"
        )
    if int(min_epochs) > int(epochs):
        raise ValueError(
            f"min_epochs ({min_epochs}) cannot exceed epochs ({epochs})"
        )
    return int(epochs), int(min_epochs), int(early_stopping_patience)


def default_aligned_out_root(run_id: str, *, project_root: Path) -> Path:
    """Legacy default under ``runs_aligned/`` (prefer ``unif_out_root`` for new runs)."""
    return Path(project_root) / "runs_aligned" / str(run_id)
