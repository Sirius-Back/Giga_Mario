"""Tests for LOCO chromosome-grain split."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from src.pipeline.common import read_csv, write_csv
from src.pipeline.split_predict import run_split_predict
from src.splits.loco import (
    assign_chrom_number_tokens,
    chrom_number_token_explicit,
    fold_id_for,
    run_loco_split_assign,
)

ID_COLS = [
    "genome",
    "chr",
    "pos1",
    "pos2",
    "gene_nameORnon_coding_ID",
    "raw_target_ID",
    "ID",
]


def _write_panel(
    tmp: Path,
    rows: list[tuple[str, str, str]],
    *,
    zsv: set[str] | None = None,
) -> tuple[Path, Path]:
    """rows: (ID, genome, chr)."""
    zsv = zsv or set()
    id_rows = []
    fold_rows = []
    for i, (rid, genome, chrom) in enumerate(rows, start=1):
        id_rows.append(
            {
                "genome": genome,
                "chr": chrom,
                "pos1": str(i),
                "pos2": str(i + 1),
                "gene_nameORnon_coding_ID": rid,
                "raw_target_ID": rid,
                "ID": rid,
            }
        )
        fold_rows.append({"ID": rid, "fold": "zsv" if rid in zsv else "0"})
    id_csv = tmp / "ID.csv"
    fold_csv = tmp / "fold.csv"
    write_csv(id_csv, id_rows, ID_COLS)
    write_csv(fold_csv, fold_rows, ["ID", "fold"])
    return id_csv, fold_csv


def test_chrom_number_token_explicit() -> None:
    assert chrom_number_token_explicit("chr1") == "1"
    assert chrom_number_token_explicit("1") == "1"
    assert chrom_number_token_explicit("X") == "X"
    assert chrom_number_token_explicit("chrY") == "Y"
    assert chrom_number_token_explicit("MT") == "MT"
    assert chrom_number_token_explicit("NW_023337852.1") == "unplaced"
    assert chrom_number_token_explicit("NC_000001.11") is None


def test_assign_chrom_number_tokens_per_genome_ordinal() -> None:
    pairs = [
        ("hg", "NC_000001.11"),
        ("hg", "NC_000002.12"),
        ("hg", "NW_1.1"),
        ("mm", "NC_000067.7"),
        ("mm", "NC_000068.8"),
        ("mm", "chrX"),
    ]
    tokens = assign_chrom_number_tokens(pairs)
    assert tokens[("hg", "NC_000001.11")] == "1"
    assert tokens[("hg", "NC_000002.12")] == "2"
    assert tokens[("hg", "NW_1.1")] == "unplaced"
    assert tokens[("mm", "NC_000067.7")] == "1"
    assert tokens[("mm", "NC_000068.8")] == "2"
    assert tokens[("mm", "chrX")] == "X"
    # Same rank across genomes share the stratification key
    assert tokens[("hg", "NC_000001.11")] == tokens[("mm", "NC_000067.7")]


def test_loco_same_chrom_same_split_and_zsv(tmp_path: Path) -> None:
    # 3 genomes × 3 primary chroms (+ unplaced) → enough folds for stratify
    rows: list[tuple[str, str, str]] = []
    for g in ("g1", "g2", "g3"):
        for c in ("NC_A.1", "NC_B.1", "NC_C.1"):
            for k in range(2):
                rows.append((f"{g}_{c}_{k}", g, c))
        rows.append((f"{g}_unp", g, "NW_u.1"))
    rows.append(("ZSV1", "g1", "NC_A.1"))
    id_csv, fold_csv = _write_panel(tmp_path, rows, zsv={"ZSV1"})

    summary = run_loco_split_assign(
        outdir=tmp_path / "out",
        id_csv=id_csv,
        fold_csv=fold_csv,
        seed=42,
        ratios=(0.6, 0.2, 0.2),
    )
    split_rows = read_csv(Path(summary["split_csv"]))
    by = {r["ID"]: r for r in split_rows}

    assert by["ZSV1"]["train_test"] == "zsv"
    assert by["ZSV1"]["fold"] == "zsv"

    # All non-ZSV genes on g1|NC_A.1 share one train_test
    chrom_ids = [rid for rid, g, c in rows if g == "g1" and c == "NC_A.1" and rid != "ZSV1"]
    labels = {by[i]["train_test"] for i in chrom_ids}
    assert len(labels) == 1
    assert by[chrom_ids[0]]["fold"] == fold_id_for("g1", "NC_A.1")

    # No chromosome fold leaks across roles
    by_fold: dict[str, set[str]] = defaultdict(set)
    for r in split_rows:
        if r["train_test"] == "zsv":
            continue
        by_fold[r["fold"]].add(r["train_test"])
    assert all(len(v) == 1 for v in by_fold.values())

    assert summary["counts"]["zsv"] == 1
    assert summary["n_folds"] == 3 * 3 + 3  # 3 genomes × 3 NC + 3 unplaced


def test_split_predict_type_loco(tmp_path: Path) -> None:
    rows: list[tuple[str, str, str]] = []
    for g in ("g1", "g2", "g3"):
        for c in ("1", "2", "3"):
            rows.append((f"{g}_c{c}", g, c))
    id_csv, fold_csv = _write_panel(tmp_path, rows)
    out = run_split_predict(
        outdir=tmp_path / "sp",
        type="loco",
        id_csv=id_csv,
        fold_csv=fold_csv,
        seed=7,
        ratios=(0.6, 0.2, 0.2),
    )
    assert out.is_file()
    split_rows = read_csv(out)
    assert {r["train_test"] for r in split_rows} <= {"train", "val", "test"}
    # Explicit chr names → fold genome|chr; stratum by number
    assert any(r["fold"] == "g1|1" for r in split_rows)
