#!/usr/bin/env python3
"""Shared helpers for src/splits/* strategies."""
from __future__ import annotations

import csv
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

SEP = "|"
TEST_FRACTION = 0.10
VAL_FRACTION_OF_TRAINPOOL = 0.10  # Caduceus-style 90/10 on train pool


def sanitize_id(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "sample"


def resolve_ready_dir(root: Path, ready: Path | None = None) -> Path:
    """Prefer explicit --ready; else ./ready then ./data_ready."""
    if ready is not None:
        p = ready if ready.is_absolute() else root / ready
        if not p.is_dir():
            raise FileNotFoundError(f"ready dir missing: {p}")
        return p.resolve()
    for name in ("ready", "data_ready"):
        p = (root / name).resolve()
        if p.is_dir():
            return p
    raise FileNotFoundError("Neither ready/ nor data_ready/ found under project root")


def resolve_raw_dir(root: Path, raw: Path | None = None) -> Path:
    p = raw if raw is not None else Path("raw")
    p = p if p.is_absolute() else root / p
    if not p.is_dir():
        raise FileNotFoundError(f"raw dir missing: {p}")
    return p.resolve()


def sample_id_from_ready_row(row: dict[str, str]) -> str:
    return sanitize_id(
        f"{row['Genome']}_{row['GeneOrID']}_{row['Chr']}_"
        f"{row['Position_start']}_{row['Position_end']}"
    )


def load_ready_table(ready_dir: Path) -> list[dict[str, Any]]:
    """Load ready.csv (pipe-delimited) + optional caduceus_ready labels paths."""
    csv_path = ready_dir / "ready.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing {csv_path}")
    rows: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8", newline="") as fh:
        # detect delimiter
        first = fh.readline()
        fh.seek(0)
        delim = "|" if SEP in first.split("\n")[0][:80] else ","
        reader = csv.DictReader(fh, delimiter=delim)
        required = {
            "Genome",
            "GeneOrID",
            "Chr",
            "Position_start",
            "Position_end",
            "TPM",
        }
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"ready.csv missing columns {required - set(reader.fieldnames or [])}"
            )
        seq_root = ready_dir / "caduceus_ready" / "all" / "sequences"
        for r in reader:
            sid = sample_id_from_ready_row(r)
            seq_path = seq_root / f"{sid}.txt"
            try:
                tpm = float(r["TPM"])
            except ValueError:
                tpm = 0.0
            rows.append(
                {
                    "sample_id": sid,
                    "Genome": r["Genome"],
                    "GeneOrID": r["GeneOrID"],
                    "Chr": r["Chr"],
                    "Position_start": int(r["Position_start"]),
                    "Position_end": int(r["Position_end"]),
                    "TPM": tpm,
                    "sequence_path": seq_path if seq_path.is_file() else None,
                }
            )
    if not rows:
        raise ValueError(f"empty ready table: {csv_path}")
    return rows


def assign_folds_random(
    n: int, *, ratios: tuple[float, float, float] | None = None
) -> list[str]:
    """Assign train/val/test labels using default or explicit train:test:val ratios.

    ``ratios=None`` preserves the Caduceus-aligned behavior: 10% test, then
    10% of the remaining samples for validation.  Explicit ratios are ordered
    ``(train, test, val)`` and use largest-remainder allocation while ensuring
    each split receives at least one sample.
    """
    if n < 3:
        raise ValueError(f"need >=3 samples for train/val/test; got {n}")
    if ratios is None:
        n_test = max(1, int(round(n * TEST_FRACTION)))
        n_remain = n - n_test
        n_val = max(1, int(round(n_remain * VAL_FRACTION_OF_TRAINPOOL)))
        n_train = n_remain - n_val
    else:
        if len(ratios) != 3 or any(value <= 0 for value in ratios):
            raise ValueError("ratios must contain three positive train:test:val values")
        total = sum(ratios)
        targets = [n * value / total for value in ratios]
        counts = [int(target) for target in targets]
        remaining = n - sum(counts)
        residuals = [target - int(target) for target in targets]
        for index in sorted(range(3), key=lambda i: (-residuals[i], i))[:remaining]:
            counts[index] += 1
        # All three roles must be represented. Move an item from the largest
        # donor split when an extremely small positive ratio rounded to zero.
        for index, count in enumerate(counts):
            if count == 0:
                donor = max(range(3), key=lambda i: (counts[i], -i))
                if counts[donor] <= 1:
                    raise ValueError(f"cannot allocate non-empty folds for n={n}")
                counts[donor] -= 1
                counts[index] = 1
        n_train, n_test, n_val = counts
    if n_train < 1:
        raise ValueError(f"train empty after ratios for n={n}")
    labels = (["train"] * n_train) + (["val"] * n_val) + (["test"] * n_test)
    if len(labels) != n:
        raise RuntimeError("fold label length mismatch")
    return labels


def assign_folds_stratified(
    items: list[Any],
    strata: list[str],
    rng,
    *,
    ratios: tuple[float, float, float] | None = None,
) -> list[str]:
    """Assign train/val/test preserving stratum proportions (by M1 fold)."""
    if len(items) != len(strata):
        raise ValueError("items/strata length mismatch")
    by_s: dict[str, list[int]] = {}
    for i, s in enumerate(strata):
        by_s.setdefault(s, []).append(i)
    out = [""] * len(items)
    for s in sorted(by_s):
        idxs = by_s[s]
        rng.shuffle(idxs)
        folds = assign_folds_random(len(idxs), ratios=ratios)
        for i, fold in zip(idxs, folds):
            out[i] = fold
    if any(not f for f in out):
        raise RuntimeError("stratified assignment left empty folds")
    return out


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_pipe_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(SEP.join(fields) + "\n")
        for r in rows:
            fh.write(SEP.join(str(r.get(c, "")) for c in fields) + "\n")


def materialize_fold(
    fold_dir: Path,
    samples: list[dict[str, Any]],
    *,
    label_field: str,
    label_fields: list[str],
) -> None:
    """Write sequences/ + labels.tsv + ready.csv for one fold (no reconversion)."""
    if fold_dir.exists():
        shutil.rmtree(fold_dir)
    seq_dir = fold_dir / "sequences"
    seq_dir.mkdir(parents=True)
    label_rows: list[dict[str, Any]] = []
    ready_rows: list[dict[str, Any]] = []
    missing_seq = 0
    for s in samples:
        sid = s["sample_id"]
        src = s.get("sequence_path")
        rel = f"sequences/{sid}.txt"
        if src is not None and Path(src).is_file():
            link_or_copy(Path(src), seq_dir / f"{sid}.txt")
        else:
            missing_seq += 1
        lab = {"sample_id": sid, "path": rel}
        for f in label_fields:
            lab[f] = s.get(f, "")
        label_rows.append(lab)
        ready_rows.append(
            {
                "Genome": s["Genome"],
                "GeneOrID": s["GeneOrID"],
                "Chr": s["Chr"],
                "Position_start": s["Position_start"],
                "Position_end": s["Position_end"],
                "TPM": s.get("TPM", 0.0),
                "sample_id": sid,
                label_field: s.get(label_field, ""),
            }
        )
    write_tsv(fold_dir / "labels.tsv", label_rows, ["sample_id", "path", *label_fields])
    write_pipe_csv(
        fold_dir / "ready.csv",
        ready_rows,
        [
            "Genome",
            "GeneOrID",
            "Chr",
            "Position_start",
            "Position_end",
            "TPM",
            "sample_id",
            label_field,
        ],
    )
    (fold_dir / "README.md").write_text(
        f"# Fold `{fold_dir.name}`\n\n"
        f"Samples: {len(samples)}\n"
        f"Primary label field: `{label_field}`\n"
        f"Missing sequence files: {missing_seq}\n"
        f"Sequences are hardlinked/symlinked from ready/ (no reconversion).\n",
        encoding="utf-8",
    )
