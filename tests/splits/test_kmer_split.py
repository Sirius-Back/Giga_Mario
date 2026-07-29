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


def test_count_kmers_local_is_abundance_not_presence() -> None:
    """Overlapping counter must tally multiplicity, not binary presence."""
    # AAAAA → AA at starts 0,1,2,3 → count 4 (presence would be 1)
    counts = count_kmers_local("AAAAA", k=2)
    assert counts == {"AA": 4}
    # AAAACAAA (L=8, k=3) → AAA,AAA,AAC,ACA,CAA,AAA → AAA×3
    counts3 = count_kmers_local("AAAACAAA", k=3)
    assert counts3["AAA"] == 3
    assert counts3["AAC"] == 1
    assert counts3["ACA"] == 1
    assert counts3["CAA"] == 1
    assert sum(counts3.values()) == 6  # L-k+1 overlapping windows
    # Distinct from presence/absence: unique keys would all be 1
    assert max(counts3.values()) > 1


def test_feature_table_preserves_abundance_ratios(tmp_path: Path) -> None:
    """Feature matrix uses count-derived values (normalize=none → raw counts)."""
    marked = tmp_path / "MARKED"
    marked.mkdir()
    # Region 1: AA×4 from AAAAA; region 2: AA×1 from AATT (dinucs AA,AT,TT)
    (marked / "1.fa").write_text(">g|c|1|2|a|t|1\nAAAAA\n", encoding="utf-8")
    (marked / "2.fa").write_text(">g|c|1|2|b|t|2\nAATT\n", encoding="utf-8")
    (marked / "3.fa").write_text(">g|c|1|2|c|t|3\nTTTT\n", encoding="utf-8")
    ft = compute_feature_table(
        marked, KmerFeatureBackend(k=2, normalize="none", engine="auto")
    )
    names = list(ft.feature_names)
    i = {rid: idx for idx, rid in enumerate(ft.ids)}
    aa_col = names.index("kmer_AA")
    assert ft.matrix[i["1"], aa_col] == pytest.approx(4.0)  # not 1.0
    assert ft.matrix[i["2"], aa_col] == pytest.approx(1.0)
    # Relative mode: multiplicity still reflected as fraction, not {0,1}
    ft_rel = compute_feature_table(
        marked, KmerFeatureBackend(k=2, normalize="relative", engine="auto")
    )
    aa_col_r = list(ft_rel.feature_names).index("kmer_AA")
    # seq1: only AA → relative 1.0; seq2: AA/AT/TT → AA relative 1/3
    assert ft_rel.matrix[i["1"], aa_col_r] == pytest.approx(1.0)
    assert ft_rel.matrix[i["2"], aa_col_r] == pytest.approx(1.0 / 3.0)


def test_multi_k_list_concatenates_all_lengths(tmp_path: Path) -> None:
    """k=[1,2] counts both lengths and prefixes columns k1_ / k2_ (local path)."""
    marked = tmp_path / "MARKED"
    marked.mkdir()
    for rid, seq in (("1", "ACGTACGT"), ("2", "GGGGCCCC"), ("3", "ATATATAT")):
        (marked / f"{rid}.fa").write_text(
            f">g|c|1|2|g{rid}|t{rid}|{rid}\n{seq}\n", encoding="utf-8"
        )
    ft = compute_feature_table(
        marked, KmerFeatureBackend(k=[1, 2], normalize="none", engine="python")
    )
    assert ft.extras is not None
    assert ft.extras["k"] == [1, 2]
    assert any(n.startswith("k1_") for n in ft.feature_names)
    assert any(n.startswith("k2_") for n in ft.feature_names)
    assert not any(n.startswith("kmer_") for n in ft.feature_names)
    names = list(ft.feature_names)
    i1 = list(ft.ids).index("1")
    # ACGTACGT: A×2,C×2,G×2,T×2; AC×2,CG×2,GT×2,TA×1
    assert ft.matrix[i1, names.index("k1_A")] == pytest.approx(2.0)
    assert ft.matrix[i1, names.index("k2_AC")] == pytest.approx(2.0)
    assert ft.matrix[i1, names.index("k2_TA")] == pytest.approx(1.0)


def test_split_predict_accepts_kmer_size_list(tmp_path: Path) -> None:
    marked, id_csv, fold_csv = _mock_panel(tmp_path, n=12)
    split_csv = run_split_predict(
        outdir=tmp_path / "pipeline_multi_k",
        type="kmer",
        seed=3,
        id_csv=id_csv,
        fold_csv=fold_csv,
        marked_fasta=marked,
        n_clusters=2,
        cluster_method="kmeans",
        plot=False,
        kmer_size=(1, 2),
    )
    assert Path(split_csv).is_file()
    import json

    summary = json.loads(
        (tmp_path / "pipeline_multi_k" / "kmer_split_meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["k"] == [1, 2]
    assert any(n.startswith("k1_") for n in summary["feature_names"])
    assert any(n.startswith("k2_") for n in summary["feature_names"])


def test_kmer_feature_backend_k2_local(tmp_path: Path) -> None:
    marked, _, _ = _mock_panel(tmp_path, n=6)
    ft = compute_feature_table(marked, KmerFeatureBackend(k=2, engine="auto"))
    assert ft.n == 6
    assert all(name.startswith("kmer_") for name in ft.feature_names)
    assert ft.matrix.shape[1] == len(ft.feature_names)
    # Relative abundance rows sum ~1 when any kmers observed
    row_sums = ft.matrix.sum(axis=1)
    assert (row_sums > 0.99).all()


def test_kmer_split_on_mock_k2(tmp_path: Path) -> None:
    marked, id_csv, fold_csv = _mock_panel(tmp_path)
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
    assert summary.get("engine") in {"native", "python", "auto"}


def test_split_predict_type_kmer(tmp_path: Path) -> None:
    marked, id_csv, fold_csv = _mock_panel(tmp_path)
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
        engine="dsk",
    )
    assert summary["k"] == [3]
    assert summary.get("engine") == "dsk"
    assert Path(summary["split_csv"]).is_file()
    assert len(summary["feature_names"]) >= 1


def test_native_counter_matches_python_for_k_from_2() -> None:
    from src.splits.sbs.backends.kmer import count_kmers
    from src.splits.sbs.backends.native import try_get_native_counter

    native = try_get_native_counter()
    assert native is not None, "build native lib: python -m src.splits.sbs.backends.native.build"
    for k in (2, 3, 4, 5):
        for seq in ("AAAAA", "ACGTACGTNACGT", "ggccaat t", "ATGC" * 20):
            assert count_kmers(seq, k, engine="python") == count_kmers(
                seq, k, engine="native"
            )


def test_production_k2_engine_is_inprocess_not_dsk(tmp_path: Path) -> None:
    """Real-run default must support k=2 without DSK."""
    marked, id_csv, fold_csv = _mock_panel(tmp_path, n=12)
    summary = run_kmer_split_assign(
        outdir=tmp_path / "prod_k2",
        fna=marked,
        id_csv=id_csv,
        fold_csv=fold_csv,
        seed=1,
        kmer_length=2,
        cluster_method="kmeans",
        n_clusters=2,
        plot=False,
        engine="auto",
    )
    assert summary["engine"] in {"native", "python"}
    assert summary["engine"] != "dsk"


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
        engine="auto",
    )
    split_rows = read_csv(Path(summary["split_csv"]))
    assert len(split_rows) == len(existing)
    assert summary["k"] == [2]
    assert summary.get("engine") in {"native", "python"}
    held = [r for r in split_rows if r["ID"] == zsv_id]
    assert held and held[0]["train_test"] == "zsv"
    assert Path(summary["feature_table"]).is_file()
    # Dinucleotide vocabulary is bounded (observed ⊆ 16)
    assert 1 <= len(summary["feature_names"]) <= 16
    assert DSK_MIN_K == 3
