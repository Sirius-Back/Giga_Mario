"""Pangenome contingency split: mock + ready_legnet 3-genome smoke."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline.common import read_csv, write_csv
from src.pipeline.split_predict import run_split_predict
from src.splits.pangenome import (
    PangenomeAdaptRequiredError,
    build_contingency_clusters,
    ensure_marked_pangenome,
    intersect_pangenome,
    run_pangenome_split_assign,
)
from src.splits.pangenome_native import ensure_built


PROJECT_ROOT = Path(__file__).resolve().parents[2]
READY_LEGNET = PROJECT_ROOT / "ready_legnet"
RAW_FNA = PROJECT_ROOT / "raw" / "fna"

# Smallest three assemblies present in both raw/fna and ready_legnet.
THREE_GENOMES = (
    "GCF_001704415.2",  # goat
    "GCF_000003025.6",  # pig
    "GCF_002863925.1",  # horse
)


def _mock_panel(tmp_path: Path, n: int = 24) -> tuple[Path, Path, Path, Path, Path]:
    marked = tmp_path / "MARKED"
    parsed = tmp_path / "PARSED"
    marked.mkdir()
    parsed.mkdir()
    id_rows = []
    fold_rows = []
    # Shared motif so first half collapses into one contingency component.
    shared = "ACGTACGTACGTACGTACGT"
    unique_gc = "GCGCGCGCGCGCGCGCGCGC"
    unique_at = "ATATATATATATATATATAT"
    for i in range(1, n + 1):
        rid = str(i)
        if i <= n // 2:
            seq = shared + ("A" * 20)
            genome = "GCF_A"
        elif i <= (3 * n) // 4:
            seq = unique_gc + ("G" * 20)
            genome = "GCF_B"
        else:
            seq = unique_at + ("T" * 20)
            genome = "GCF_C"
        (marked / f"{rid}.fa").write_text(
            f">{genome}|chr1|{i}|{i+10}|g{i}|t{i}|{rid}\n{seq}\n",
            encoding="utf-8",
        )
        # Drop one ID from PARSED to exercise the filter.
        if i != n:
            (parsed / f"{rid}.ext").write_text(seq + "\n", encoding="utf-8")
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
    return marked, parsed, id_csv, fold_csv, tmp_path


def test_native_library_builds() -> None:
    path = ensure_built(force=False)
    assert path.is_file()


def test_intersect_drops_missing_parsed(tmp_path: Path) -> None:
    marked, parsed, id_csv, _, _ = _mock_panel(tmp_path, n=10)
    kept = intersect_pangenome(marked, parsed)
    assert "10" not in kept
    assert "1" in kept
    assert len(kept) == 9


def test_contingency_shared_kmers_same_cluster() -> None:
    ensure_built()
    shared = "ACGTACGTACGTACGTACGTACGT"
    a = shared + "AAAAAAAAAA"
    b = shared + "TTTTTTTTTT"
    c = "GGGGGGGGGGGGGGGGGGGGGGGG"
    res = build_contingency_clusters([a, b, c], k=8, collect_edges=True)
    assert res.cluster_ids[0] == res.cluster_ids[1]
    assert res.cluster_ids[0] != res.cluster_ids[2]
    assert res.n_clusters >= 2
    assert len(res.edge_u) >= 1


def test_pangenome_split_on_mock(tmp_path: Path) -> None:
    marked, parsed, id_csv, fold_csv, _ = _mock_panel(tmp_path)
    summary = run_pangenome_split_assign(
        outdir=tmp_path / "pg_out",
        marked_pangenome=marked,
        parsed=parsed,
        id_csv=id_csv,
        fold_csv=fold_csv,
        seed=42,
        k=8,
        plot=True,
    )
    split_rows = read_csv(Path(summary["split_csv"]))
    by_id = {r["ID"]: r for r in split_rows}
    assert "24" not in by_id  # filtered (no PARSED)
    assert by_id["1"]["train_test"] == "zsv"
    assert Path(summary["marked_parsed"]).is_dir()
    assert summary["marked_source"]["source"] == "marked_pangenome"
    figs = tmp_path / "pg_out" / "figures"
    assert (figs / "contingency_graph.json").is_file()
    assert (figs / "contingency_graph.dot").is_file()
    assert (figs / "contingency_graph.pdf").is_file() or (
        figs / "contingency_graph.png"
    ).is_file()
    assert (figs / "Figure_pangenome_contingency_fold_train_test.pdf").is_file() or (
        figs / "Figure_pangenome_contingency_fold_train_test.png"
    ).is_file()
    graph_dir = tmp_path / "pg_out" / "graph"
    assert (graph_dir / "contingency_graph.npz").is_file()
    assert (graph_dir / "ids.txt").is_file()
    assert (graph_dir / "nodes.tsv").is_file()
    assert (graph_dir / "edges.tsv").is_file()
    assert (graph_dir / "contingency_graph_meta.json").is_file()
    from src.splits.pangenome import load_contingency_graph

    loaded = load_contingency_graph(graph_dir)
    assert len(loaded["ids"]) == len(split_rows)
    assert len(loaded["cluster_ids"]) == len(split_rows)

    # Shared-motif non-ZSV IDs should share a fold when not held out.
    shared_folds = {by_id[str(i)]["fold"] for i in range(2, 13)}
    assert len(shared_folds) == 1


def test_a2a_requires_adapt_when_panel_marked_not_reused(tmp_path: Path) -> None:
    marked, parsed, id_csv, fold_csv, _ = _mock_panel(tmp_path)
    with pytest.raises(PangenomeAdaptRequiredError, match="A2A"):
        run_pangenome_split_assign(
            outdir=tmp_path / "pg_a2a",
            panel_marked=marked,
            parsed=parsed,
            id_csv=id_csv,
            fold_csv=fold_csv,
            reuse_panel_marked=False,
            seed=42,
            k=8,
            plot=False,
        )


def test_reuse_panel_marked_opt_in(tmp_path: Path) -> None:
    marked, parsed, id_csv, fold_csv, _ = _mock_panel(tmp_path)
    with pytest.warns(UserWarning, match="reuse_panel_marked"):
        summary = run_pangenome_split_assign(
            outdir=tmp_path / "pg_reuse",
            panel_marked=marked,
            parsed=parsed,
            id_csv=id_csv,
            fold_csv=fold_csv,
            reuse_panel_marked=True,
            seed=42,
            k=8,
            plot=False,
        )
    assert summary["marked_source"]["source"] == "reuse_panel_marked"
    assert Path(summary["marked_parsed"]).is_dir()


def test_ensure_marked_pangenome_prefers_explicit(tmp_path: Path) -> None:
    marked, _, _, _, _ = _mock_panel(tmp_path, n=6)
    mp, meta = ensure_marked_pangenome(
        outdir=tmp_path / "out",
        marked_pangenome=marked,
    )
    assert mp == marked
    assert meta["source"] == "marked_pangenome"


def test_split_predict_type_pangenome_mock(tmp_path: Path) -> None:
    marked, parsed, id_csv, fold_csv, _ = _mock_panel(tmp_path)
    out = run_split_predict(
        outdir=tmp_path / "sp_out",
        type="pangenome",
        id_csv=id_csv,
        fold_csv=fold_csv,
        marked_pangenome=marked,
        parsed=parsed,
        seed=42,
        kmer_size=8,
        plot=True,
    )
    assert out.is_file()
    rows = read_csv(out)
    assert any(r["train_test"] == "zsv" for r in rows)


@pytest.mark.skipif(
    not (READY_LEGNET / "MARKED").is_dir() or not (READY_LEGNET / "PARSED").is_dir(),
    reason="ready_legnet MARKED/PARSED missing",
)
@pytest.mark.skipif(
    not RAW_FNA.is_dir(),
    reason="raw/fna missing",
)
def test_pangenome_three_genomes_ready_legnet(tmp_path: Path) -> None:
    """Smoke: 3 genomes from raw ∩ ready_legnet (capped IDs for pytest runtime)."""
    raw_gcfs = set()
    for p in RAW_FNA.glob("*.fna"):
        if p.name.endswith(".fai"):
            continue
        parts = p.name.split("_")
        if len(parts) >= 2 and parts[0] in {"GCF", "GCA"}:
            raw_gcfs.add(f"{parts[0]}_{parts[1]}")
    for g in THREE_GENOMES:
        assert g in raw_gcfs, f"{g} not found under raw/fna"

    id_csv = READY_LEGNET / "ID.csv"
    fold_csv = READY_LEGNET / "fold.csv"
    summary = run_pangenome_split_assign(
        outdir=tmp_path / "pg_3g",
        # Smoke only: panel LegNet window == intentional reuse (not production default).
        panel_marked=READY_LEGNET / "MARKED",
        reuse_panel_marked=True,
        parsed=READY_LEGNET / "PARSED",
        id_csv=id_csv,
        fold_csv=fold_csv if fold_csv.is_file() else None,
        seed=42,
        genomes=THREE_GENOMES,
        max_ids=300,
        k=15,
        plot=True,
    )
    assert summary["n_ids"] <= 300
    assert summary["n_ids"] >= 3
    split_rows = read_csv(Path(summary["split_csv"]))
    assert len(split_rows) == summary["n_ids"]
    labels = {r["train_test"] for r in split_rows}
    assert labels & {"train", "val", "test", "zsv"}
    assert Path(summary["split_csv"]).is_file()
    figs = tmp_path / "pg_3g" / "figures"
    assert (figs / "contingency_graph.json").is_file()


def test_refine_large_components_by_modularity_splits_large_cc() -> None:
    from src.splits.pangenome import refine_large_components_by_modularity

    # Two motif families inside one oversized fake CC → Louvain should split.
    shared_a = ("ACGT" * 30)
    shared_b = ("TGCA" * 30)
    ids = [str(i) for i in range(12)]
    sequences = [shared_a + ("A" * i) for i in range(6)] + [
        shared_b + ("T" * i) for i in range(6)
    ]
    cluster_ids = [0] * 12
    out, meta = refine_large_components_by_modularity(
        ids,
        sequences,
        cluster_ids,
        k=4,
        max_fold_size=5,
        max_edges=50_000,
        seed=42,
    )
    assert meta["n_large_components"] == 1
    assert len(set(out)) >= 2
    assert len(out) == 12
