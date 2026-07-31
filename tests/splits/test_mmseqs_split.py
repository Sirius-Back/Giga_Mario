"""MMseqs2 cluster-first split: unit helpers + optional CLI smoke."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline.common import write_csv
from src.pipeline.split_predict import run_split_predict
from src.splits.mmseqs import DEFAULT_RATIOS, run_mmseqs_split_assign
from src.splits.sbs.backends.mmseqs import (
    cluster_map_to_dense_ids,
    find_mmseqs,
    parse_cluster_tsv,
)


def _mmseqs_available() -> bool:
    try:
        find_mmseqs()
        return True
    except FileNotFoundError:
        return False


def _mock_panel(tmp_path: Path, n: int = 24) -> tuple[Path, Path, Path]:
    marked = tmp_path / "MARKED"
    marked.mkdir()
    id_rows = []
    fold_rows = []
    for i in range(1, n + 1):
        rid = str(i)
        if i <= n // 3:
            seq = ("ATGC" * 20) + ("A" * (i % 3))
            genome = "GCF_A"
        elif i <= 2 * n // 3:
            seq = ("GGCC" * 20) + ("T" * (i % 3))
            genome = "GCF_B"
        else:
            seq = ("TTTTAAAA" * 10) + ("C" * (i % 3))
            genome = "GCF_C"
        (marked / f"{rid}.fa").write_text(
            f">{genome}|chr1|{i}|{i + 80}|g{i}|t{i}|{rid}\n{seq}\n",
            encoding="utf-8",
        )
        id_rows.append(
            {
                "genome": genome,
                "chr": "chr1",
                "pos1": str(i),
                "pos2": str(i + 80),
                "gene_nameORnon_coding_ID": f"g{i}",
                "raw_target_ID": f"t{i}",
                "ID": rid,
            }
        )
        fold_rows.append({"ID": rid, "fold": "zsv" if i == 1 else "0"})
    id_csv = tmp_path / "ID.csv"
    fold_csv = tmp_path / "fold.csv"
    write_csv(
        id_csv,
        id_rows,
        [
            "genome",
            "chr",
            "pos1",
            "pos2",
            "gene_nameORnon_coding_ID",
            "raw_target_ID",
            "ID",
        ],
    )
    write_csv(fold_csv, fold_rows, ["ID", "fold"])
    return marked, id_csv, fold_csv


def test_default_ratios_are_60_20_20() -> None:
    assert DEFAULT_RATIOS == (0.6, 0.2, 0.2)


def test_parse_cluster_tsv_and_dense_ids(tmp_path: Path) -> None:
    tsv = tmp_path / "clu_cluster.tsv"
    tsv.write_text(
        "repA\trepA\n"
        "repA\tm1\n"
        "repA\tm2\n"
        "repB\trepB\n"
        "repB\tm3\n",
        encoding="utf-8",
    )
    member_to_rep = parse_cluster_tsv(tsv)
    assert member_to_rep["m1"] == "repA"
    assert member_to_rep["m3"] == "repB"
    dense = cluster_map_to_dense_ids(
        member_to_rep, ids=["repA", "m1", "m2", "repB", "m3", "orphan"]
    )
    assert dense["repA"] == dense["m1"] == dense["m2"]
    assert dense["repB"] == dense["m3"]
    assert dense["orphan"] not in {dense["repA"], dense["repB"]}


def test_parse_cluster_tsv_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_cluster_tsv(tmp_path / "missing.tsv")


@pytest.mark.skipif(not _mmseqs_available(), reason="mmseqs binary not available")
def test_mmseqs_split_smoke(tmp_path: Path) -> None:
    marked, id_csv, fold_csv = _mock_panel(tmp_path)
    outdir = tmp_path / "out"
    summary = run_mmseqs_split_assign(
        outdir=outdir,
        fna=marked,
        id_csv=id_csv,
        fold_csv=fold_csv,
        seed=42,
        ratios=DEFAULT_RATIOS,
        threads=2,
        min_seq_id=0.5,
        plot=False,
    )
    split_csv = Path(summary["split_csv"])
    assert split_csv.is_file()
    text = split_csv.read_text(encoding="utf-8")
    assert "train" in text and "test" in text and "val" in text
    assert "zsv" in text
    labels = {}
    for line in text.splitlines()[1:]:
        if not line.strip():
            continue
        rid, tt, _fold = line.split("|")
        labels[rid] = tt
    assert labels["1"] == "zsv"
    assert set(labels.values()) >= {"train", "test", "val", "zsv"}


@pytest.mark.skipif(not _mmseqs_available(), reason="mmseqs binary not available")
def test_mmseqs_via_split_predict(tmp_path: Path) -> None:
    marked, id_csv, fold_csv = _mock_panel(tmp_path, n=18)
    out = run_split_predict(
        outdir=tmp_path / "sp",
        type="mmseqs",
        seed=7,
        id_csv=id_csv,
        fold_csv=fold_csv,
        marked_fasta=marked,
        ratios=None,  # strategy default 60:20:20
        threads=2,
        min_seq_id=0.5,
        force=True,
    )
    assert out.is_file()
    assert out.name == "split.csv"
