"""K-mer SBS strategy tests (DSK backend + ready_legnet subset at k=2)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.pipeline.common import read_csv, write_csv
from src.pipeline.split_predict import run_split_predict
from src.splits.kmer import run_kmer_split_assign
from src.splits.sbs.backends.kmer import (
    DSK_MIN_K,
    KmerFeatureBackend,
    count_kmers_local,
    find_dsk,
    parse_dsk_ascii,
)
from src.splits.sbs.features import compute_feature_table


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Project LegNet-ready panel (MARKED layout). Caption/user name: legnet_ready.
READY_LEGNET = PROJECT_ROOT / "ready_legnet"
LEGNET_MARKED = READY_LEGNET / "MARKED"
LEGNET_ID_CSV = READY_LEGNET / "ID.csv"
HAS_DSK = shutil.which("dsk") is not None and shutil.which("dsk2ascii") is not None


def _mock_panel(tmp_path: Path, n: int = 24) -> tuple[Path, Path, Path]:
    marked = tmp_path / "MARKED"
    marked.mkdir()
    id_rows = []
    fold_rows = []
    for i in range(1, n + 1):
        rid = str(i)
        if i <= n // 2:
            seq = ("AA" * 40) + ("TT" * 20)  # AT-rich dinucleotides
            genome = "GCF_A"
        else:
            seq = ("GC" * 40) + ("CG" * 20)  # GC-rich dinucleotides
            genome = "GCF_B"
        (marked / f"{rid}.fa").write_text(
            f">{genome}|chr1|{i}|{i+10}|g{i}|t{i}|{rid}\n{seq}\n",
            encoding="utf-8",
        )
        id_rows.append(
            {
                "genome": genome,
                "chr": "chr1",
                "pos1": str(i),
                "pos2": str(i + 10),
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


def test_parse_dsk_ascii_sums_duplicates() -> None:
    text = "AAA 2\nAAC 1\nAAA 3\n"
    assert parse_dsk_ascii(text) == {"AAA": 5, "AAC": 1}


def test_count_kmers_local_k2() -> None:
    counts = count_kmers_local("ACGTAC", k=2)
    assert counts["AC"] == 2
    assert counts["CG"] == 1
    assert counts["GT"] == 1
    assert counts["TA"] == 1


def test_kmer_feature_backend_k2_local(tmp_path: Path) -> None:
    marked, _, _ = _mock_panel(tmp_path, n=6)
    with pytest.warns(UserWarning, match="DSK does not support"):
        ft = compute_feature_table(marked, KmerFeatureBackend(k=2))
    assert ft.n == 6
    assert all(name.startswith("kmer_") for name in ft.feature_names)
    assert ft.matrix.shape[1] == len(ft.feature_names)
    # Relative abundance rows sum ~1 when any kmers observed
    row_sums = ft.matrix.sum(axis=1)
    assert (row_sums > 0.99).all()


def test_kmer_split_on_mock_k2(tmp_path: Path) -> None:
    marked, id_csv, fold_csv = _mock_panel(tmp_path)
    with pytest.warns(UserWarning, match="DSK does not support"):
        summary = run_kmer_split_assign(
            outdir=tmp_path / "kmer_out",
            fna=marked,
            id_csv=id_csv,
            fold_csv=fold_csv,
            seed=42,
            kmer_length=2,
            n_clusters=2,
            cluster_method="kmeans",
            plot=False,
        )
    split_rows = read_csv(Path(summary["split_csv"]))
    by_id = {r["ID"]: r for r in split_rows}
    assert by_id["1"]["train_test"] == "zsv"
    assert summary["k"] == [2]
    assert Path(summary["feature_table"]).is_file()


def test_split_predict_type_kmer(tmp_path: Path) -> None:
    marked, id_csv, fold_csv = _mock_panel(tmp_path)
    with pytest.warns(UserWarning, match="DSK does not support"):
        split_csv = run_split_predict(
            outdir=tmp_path / "pipeline_kmer",
            type="kmer",
            seed=7,
            id_csv=id_csv,
            fold_csv=fold_csv,
            marked_fasta=marked,
            n_clusters=2,
            cluster_method="kmeans",
            plot=False,
            kmer_size=2,
        )
    rows = read_csv(split_csv)
    assert len(rows) == 24
    assert any(r["train_test"] == "zsv" for r in rows)


@pytest.mark.skipif(not HAS_DSK, reason="dsk/dsk2ascii not on PATH")
def test_kmer_dsk_k3_smoke(tmp_path: Path) -> None:
    find_dsk()  # raises if missing
    marked, id_csv, fold_csv = _mock_panel(tmp_path, n=8)
    summary = run_kmer_split_assign(
        outdir=tmp_path / "kmer_dsk",
        fna=marked,
        id_csv=id_csv,
        fold_csv=fold_csv,
        seed=42,
        k=3,
        n_clusters=2,
        cluster_method="kmeans",
        plot=False,
    )
    assert summary["k"] == [3]
    assert Path(summary["split_csv"]).is_file()
    assert len(summary["feature_names"]) >= 1


@pytest.mark.skipif(
    not LEGNET_MARKED.is_dir() or not LEGNET_ID_CSV.is_file(),
    reason="ready_legnet/MARKED or ID.csv missing",
)
def test_kmer_on_ready_legnet_subset_k2(tmp_path: Path) -> None:
    """User-requested smoke: subset of LegNet-ready panel, kmer_length=2."""
    id_rows_all = read_csv(LEGNET_ID_CSV)
    # Prefer human assembly rows when present; take a small contiguous subset.
    human = [
        r
        for r in id_rows_all
        if str(r.get("genome", "")).startswith("GCF_000001405")
    ]
    pool = human if len(human) >= 40 else id_rows_all
    subset = pool[:40]
    use_ids = [r["ID"].strip() for r in subset]
    existing = [rid for rid in use_ids if (LEGNET_MARKED / f"{rid}.fa").is_file()]
    assert len(existing) >= 12

    # Hold out first existing id as zsv for contract coverage.
    zsv_id = existing[0]
    id_rows = [r for r in subset if r["ID"].strip() in set(existing)]
    fold_rows = [
        {"ID": r["ID"].strip(), "fold": "zsv" if r["ID"].strip() == zsv_id else "0"}
        for r in id_rows
    ]
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

    with pytest.warns(UserWarning, match="DSK does not support"):
        summary = run_kmer_split_assign(
            outdir=tmp_path / "legnet_kmer_k2",
            fna=LEGNET_MARKED,
            id_csv=id_csv,
            fold_csv=fold_csv,
            seed=42,
            ids=existing,
            kmer_length=2,
            cluster_method="kmeans",
            n_clusters=2,
            plot=False,
        )
    split_rows = read_csv(Path(summary["split_csv"]))
    assert len(split_rows) == len(existing)
    assert summary["k"] == [2]
    held = [r for r in split_rows if r["ID"] == zsv_id]
    assert held and held[0]["train_test"] == "zsv"
    assert Path(summary["feature_table"]).is_file()
    # Dinucleotide vocabulary is bounded (observed ⊆ 16)
    assert 1 <= len(summary["feature_names"]) <= 16
    assert DSK_MIN_K == 3
