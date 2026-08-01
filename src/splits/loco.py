"""Leave-one-chromosome-out (LOCO) / chromosome-grain split.

Caption: ``splits/LOCO.md``.
Wired into ``split-predict`` as ``type=loco``.

Fold grain is ``(genome, chr)``: every region on the same contig of the same
organism shares one ``train_test`` label. Fold → train/val/test is stratified
by a normalized **chromosome number token** so homologous ranks (e.g. chr1
across species) are balanced across roles.
"""
from __future__ import annotations

import json
import random
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.pipeline.common import SPLIT_CSV_COLUMNS, ensure_dir, read_csv, write_csv
from src.pipeline.generate_fold import is_zsv_fold, normalize_fold_label
from src.splits.common import assign_folds_random

__all__ = (
    "SPLIT_ID",
    "chrom_number_token_explicit",
    "assign_chrom_number_tokens",
    "fold_id_for",
    "run_loco_split_assign",
)

SPLIT_ID = "loco"

_RE_AUTOSOME = re.compile(
    r"(?i)^(?:chr|chromosome)?[_ ]?(\d+)$"
)
_RE_SEX = re.compile(r"(?i)^(?:chr|chromosome)?[_ ]?([XY])$")
_RE_MT = re.compile(r"(?i)^(?:chr|chromosome)?[_ ]?(M|MT|MITO)$")
_RE_UNPLACED = re.compile(
    r"(?i)^(NW_|NT_|unplaced)|^.*(?:unplaced|_random_|_alt).*"
)


def fold_id_for(genome: str, chrom: str) -> str:
    """Stable fold id: ``genome|chr`` (raw contig id)."""
    return f"{genome}|{chrom}"


def chrom_number_token_explicit(raw_chr: str) -> str | None:
    """Return a chrom-number token from an explicit name, else ``None``.

    Handles ``chr1`` / ``1`` / ``X`` / ``Y`` / ``MT`` and RefSeq-style
    unplaced scaffolds (``NW_`` / ``NT_`` / random / alt). Primary assembled
    accessions (e.g. ``NC_*``) return ``None`` so the caller can assign
    per-genome ordinals.
    """
    s = (raw_chr or "").strip()
    if not s:
        return "unplaced"
    m = _RE_AUTOSOME.fullmatch(s)
    if m:
        return str(int(m.group(1)))  # normalize leading zeros
    m = _RE_SEX.fullmatch(s)
    if m:
        return m.group(1).upper()
    m = _RE_MT.fullmatch(s)
    if m:
        return "MT"
    if _RE_UNPLACED.match(s):
        return "unplaced"
    return None


def assign_chrom_number_tokens(
    genome_chr_pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], str]:
    """Map each ``(genome, chr)`` to a stratification token.

    Explicit name tokens are used when parseable. Remaining contigs (typically
    primary ``NC_*`` RefSeq chromosomes) receive per-genome lexicographic
    ordinals ``\"1\"``, ``\"2\"``, … so the first primary chromosome of each
    species shares stratum ``\"1\"``.
    """
    unique = sorted(set(genome_chr_pairs))
    out: dict[tuple[str, str], str] = {}
    pending: dict[str, list[str]] = defaultdict(list)
    for genome, chrom in unique:
        explicit = chrom_number_token_explicit(chrom)
        if explicit is not None:
            out[(genome, chrom)] = explicit
        else:
            pending[genome].append(chrom)
    for genome, chroms in pending.items():
        for i, chrom in enumerate(sorted(set(chroms)), start=1):
            out[(genome, chrom)] = str(i)
    return out


def _load_id_rows(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"id_csv is empty: {path}")
    required = {"genome", "chr", "ID"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(
            f"id_csv missing columns {sorted(missing)}; have {list(rows[0])}"
        )
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for row_number, row in enumerate(rows, start=2):
        rid = row["ID"].strip()
        if not rid:
            raise ValueError(f"id_csv has blank ID at row {row_number}")
        if rid in seen:
            raise ValueError(f"id_csv has duplicate ID {rid!r}")
        seen.add(rid)
        genome = row["genome"].strip()
        chrom = row["chr"].strip()
        if not genome or not chrom:
            raise ValueError(
                f"id_csv row {row_number} has blank genome/chr "
                f"(ID={rid!r})"
            )
        out.append(
            {
                "ID": rid,
                "genome": genome,
                "chr": chrom,
            }
        )
    return out


def _load_fold_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    rows = read_csv(Path(path))
    if not rows:
        raise ValueError(f"fold.csv is empty: {path}")
    if "ID" not in rows[0] or "fold" not in rows[0]:
        raise ValueError(f"fold.csv missing ID/fold; have {list(rows[0])}")
    out: dict[str, str] = {}
    for row in rows:
        rid = row["ID"].strip()
        if not rid:
            continue
        out[rid] = normalize_fold_label(row["fold"])
    return out


def _assign_folds_by_chrom_number(
    fold_ids: list[str],
    *,
    fold_strata: dict[str, str],
    seed: int,
    ratios: tuple[float, float, float] | None,
) -> dict[str, str]:
    """Map chromosome folds → train/val/test, stratified by chrom-number token.

    ``assign_folds_random`` requires n≥3; strata (or leftover pools) smaller
    than 3 use a deterministic cyclic fallback so tiny panels still run.
    """
    if not fold_ids:
        return {}
    if len(fold_ids) < 3:
        labels = ["train", "val", "test"]
        return {fid: labels[i % 3] for i, fid in enumerate(sorted(fold_ids))}

    by_stratum: dict[str, list[str]] = defaultdict(list)
    for fid in fold_ids:
        by_stratum[fold_strata[fid]].append(fid)

    rng = random.Random(seed)
    out: dict[str, str] = {}
    leftovers: list[str] = []
    for stratum in sorted(by_stratum):
        members = list(by_stratum[stratum])
        if len(members) < 3:
            leftovers.extend(members)
            continue
        rng.shuffle(members)
        labels = assign_folds_random(len(members), ratios=ratios)
        out.update(zip(members, labels))

    if leftovers:
        if len(leftovers) >= 3:
            order = list(leftovers)
            rng.shuffle(order)
            labels = assign_folds_random(len(order), ratios=ratios)
            out.update(zip(order, labels))
        else:
            for i, fid in enumerate(sorted(leftovers)):
                out[fid] = ["train", "val", "test"][i % 3]
    return out


def run_loco_split_assign(
    *,
    outdir: Path | str,
    id_csv: Path | str,
    fold_csv: Path | str | None = None,
    seed: int = 42,
    ratios: tuple[float, float, float] | None = None,
    max_ids: int | None = None,
    plot: bool = False,
) -> dict[str, Any]:
    """Assign train/val/test at ``(genome, chr)`` grain; write ``split.csv``.

    Returns a summary dict including ``split_csv`` and chrom-token metadata.
    """
    _ = plot  # caption allows --plot; no SBS PCA for metadata-only LOCO
    outdir = ensure_dir(Path(outdir))
    id_csv_p = Path(id_csv)
    if not id_csv_p.is_file():
        raise FileNotFoundError(f"id_csv missing: {id_csv_p}")

    if fold_csv is None:
        warnings.warn("Warning: folds are not included", UserWarning, stacklevel=2)

    id_rows = _load_id_rows(id_csv_p)
    if max_ids is not None and max_ids > 0:
        id_rows = id_rows[: int(max_ids)]

    fold_map = _load_fold_map(Path(fold_csv) if fold_csv else None)
    id_set = {r["ID"] for r in id_rows}
    unknown = set(fold_map) - id_set
    if unknown:
        raise ValueError(
            f"fold.csv contains ID absent from id_csv: {sorted(unknown)[0]!r}"
        )

    zsv_ids: list[str] = []
    assignable: list[dict[str, str]] = []
    for row in id_rows:
        rid = row["ID"]
        raw = fold_map.get(rid, "0")
        if is_zsv_fold(raw):
            zsv_ids.append(rid)
        else:
            assignable.append(row)

    if not assignable:
        raise ValueError("no assignable IDs after ZSV holdout")

    pairs = [(r["genome"], r["chr"]) for r in assignable]
    token_map = assign_chrom_number_tokens(pairs)

    fold_members: dict[str, list[str]] = defaultdict(list)
    fold_strata: dict[str, str] = {}
    for row in assignable:
        fid = fold_id_for(row["genome"], row["chr"])
        fold_members[fid].append(row["ID"])
        fold_strata[fid] = token_map[(row["genome"], row["chr"])]

    fold_ids = sorted(fold_members)
    fold_to_tt = _assign_folds_by_chrom_number(
        fold_ids,
        fold_strata=fold_strata,
        seed=int(seed),
        ratios=ratios,
    )

    id_to_tt: dict[str, str] = {}
    id_to_fold: dict[str, str] = {}
    for fid, members in fold_members.items():
        label = fold_to_tt[fid]
        for rid in members:
            id_to_tt[rid] = label
            id_to_fold[rid] = fid

    rows: list[dict[str, Any]] = []
    zsv_set = set(zsv_ids)
    for row in id_rows:
        rid = row["ID"]
        if rid in zsv_set:
            rows.append({"ID": rid, "train_test": "zsv", "fold": "zsv"})
        else:
            rows.append(
                {
                    "ID": rid,
                    "train_test": id_to_tt[rid],
                    "fold": id_to_fold[rid],
                }
            )

    # Contract: never split one (genome, chr) across train/val/test
    by_fold_labels: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r["train_test"] == "zsv":
            continue
        by_fold_labels[r["fold"]].add(r["train_test"])
    leaks = {f: labs for f, labs in by_fold_labels.items() if len(labs) > 1}
    if leaks:
        example = sorted(leaks)[0]
        raise RuntimeError(
            f"chromosome fold split across roles: {example!r} → {leaks[example]}"
        )

    split_csv = outdir / "split.csv"
    write_csv(split_csv, rows, SPLIT_CSV_COLUMNS)

    meta = {
        "strategy": SPLIT_ID,
        "seed": int(seed),
        "ratios": list(ratios) if ratios is not None else None,
        "n_folds": len(fold_ids),
        "n_assignable": len(assignable),
        "n_zsv": len(zsv_ids),
        "chrom_token_counts": {
            tok: sum(1 for v in fold_strata.values() if v == tok)
            for tok in sorted(set(fold_strata.values()))
        },
        "fold_to_train_test": fold_to_tt,
        "fold_strata": fold_strata,
    }
    meta_path = outdir / "loco_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    counts = {
        "train": sum(1 for r in rows if r["train_test"] == "train"),
        "val": sum(1 for r in rows if r["train_test"] == "val"),
        "test": sum(1 for r in rows if r["train_test"] == "test"),
        "zsv": sum(1 for r in rows if r["train_test"] == "zsv"),
    }
    return {
        "split_csv": str(split_csv),
        "meta_json": str(meta_path),
        "counts": counts,
        "n_folds": len(fold_ids),
        "fold_strata": fold_strata,
    }
