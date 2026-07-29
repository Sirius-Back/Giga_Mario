"""GC strategy (SBS + GC%/AAA% features) mock + optional MARKED smoke."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline.common import read_csv, write_csv
from src.pipeline.split_predict import run_split_predict
from src.splits.gc import run_gc_split_assign


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MARKED = PROJECT_ROOT / "ready_caduceus" / "MARKED"
ID_CSV = PROJECT_ROOT / "ready_caduceus" / "ID.csv"
FOLD_CSV = PROJECT_ROOT / "ready_caduceus" / "fold.csv"


def _mock_panel(tmp_path: Path, n: int = 24) -> tuple[Path, Path, Path, Path]:
    marked = tmp_path / "MARKED"
    marked.mkdir()
    id_rows = []
    fold_rows = []
    strat_rows = []
    for i in range(1, n + 1):
        rid = str(i)
        # Two composition groups: AT-rich vs GC-rich
        if i <= n // 2:
            seq = ("AAA" * 10) + ("T" * 10)  # high AAA, low GC
            genome = "GCF_A"
            strat = "low"
        else:
            seq = ("GCGC" * 10)  # high GC, low AAA
            genome = "GCF_B"
            strat = "high"
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
        strat_rows.append({"ID": rid, "strat1": strat, "w": "1"})
    id_csv = tmp_path / "ID.csv"
    fold_csv = tmp_path / "fold.csv"
    strat_csv = tmp_path / "stratification.csv"
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
    write_csv(strat_csv, strat_rows, ["ID", "strat1", "w"])
    return marked, id_csv, fold_csv, strat_csv


def test_gc_split_on_mock(tmp_path: Path) -> None:
    marked, id_csv, fold_csv, strat_csv = _mock_panel(tmp_path)
    summary = run_gc_split_assign(
        outdir=tmp_path / "gc_out",
        fna=marked,
        id_csv=id_csv,
        fold_csv=fold_csv,
        stratification_csv=strat_csv,
        seed=42,
        n_clusters=2,
        cluster_method="kmeans",
        plot=True,
        custom_label_csv=strat_csv,
        custom_label_column="strat1",
    )
    split_rows = read_csv(Path(summary["split_csv"]))
    by_id = {r["ID"]: r for r in split_rows}
    assert by_id["1"]["train_test"] == "zsv"
    assert Path(summary["feature_table"]).is_file()
    figs = tmp_path / "gc_out" / "figures"
    assert (figs / "pca_by_cluster.pdf").is_file() or (figs / "pca_by_cluster.png").is_file()
    assert (figs / "pca_by_train_test.pdf").is_file() or (
        figs / "pca_by_train_test.png"
    ).is_file()
    assert (figs / "pca_by_genome.pdf").is_file() or (figs / "pca_by_genome.png").is_file()
    assert list(figs.glob("pca_by_custom_strat1.*"))


def test_gc_split_dbscan_default(tmp_path: Path) -> None:
    marked, id_csv, fold_csv, strat_csv = _mock_panel(tmp_path, n=30)
    summary = run_gc_split_assign(
        outdir=tmp_path / "gc_dbscan",
        fna=marked,
        id_csv=id_csv,
        fold_csv=fold_csv,
        stratification_csv=strat_csv,
        seed=42,
        cluster_method="dbscan",
        dbscan_min_samples=3,
        plot=False,
    )
    assert summary["cluster_method"] == "dbscan"
    assert Path(summary["split_csv"]).is_file()


def test_split_predict_type_gc(tmp_path: Path) -> None:
    marked, id_csv, fold_csv, strat_csv = _mock_panel(tmp_path)
    split_csv = run_split_predict(
        outdir=tmp_path / "pipeline_gc",
        type="gc",
        seed=7,
        id_csv=id_csv,
        fold_csv=fold_csv,
        stratification_csv=strat_csv,
        marked_fasta=marked,
        n_clusters=2,
        cluster_method="kmeans",
        plot=False,
    )
    rows = read_csv(split_csv)
    assert len(rows) == 24
    assert any(r["train_test"] == "zsv" for r in rows)


@pytest.mark.skipif(not MARKED.is_dir(), reason="ready_caduceus/MARKED missing")
def test_gc_on_marked_subset(tmp_path: Path) -> None:
    """Real MARKED smoke on feature clustering + PCA (not O(n²) distances)."""
    ids: list[str] = []
    if FOLD_CSV.is_file() and ID_CSV.is_file():
        fold_rows = read_csv(FOLD_CSV)
        for row in fold_rows:
            if row.get("fold", "").strip().lower() in {"zsv", "zeroshotvalidation"}:
                continue
            if row.get("genome", "").startswith("GCF_000001405"):
                ids.append(row["ID"].strip())
            if len(ids) >= 200:
                break
    if len(ids) < 12:
        ids = [p.stem for p in sorted(MARKED.glob("*.fa"))[:200]]

    zsv_ids: list[str] = []
    if FOLD_CSV.is_file():
        for row in read_csv(FOLD_CSV):
            if row.get("fold", "").strip().lower() in {"zsv", "zeroshotvalidation"}:
                zsv_ids.append(row["ID"].strip())
                if len(zsv_ids) >= 5:
                    break

    use_ids = zsv_ids + ids
    id_rows = []
    fold_rows = []
    if ID_CSV.is_file():
        wanted = set(use_ids)
        for row in read_csv(ID_CSV):
            if row["ID"] in wanted:
                id_rows.append(row)
    zsv_set = set(zsv_ids)
    for row in id_rows:
        fold_rows.append(
            {"ID": row["ID"], "fold": "zsv" if row["ID"] in zsv_set else "0"}
        )

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

    existing = [rid for rid in use_ids if (MARKED / f"{rid}.fa").is_file()]
    assert len(existing) >= 12

    summary = run_gc_split_assign(
        outdir=tmp_path / "marked_gc",
        fna=MARKED,
        id_csv=id_csv,
        fold_csv=fold_csv,
        seed=42,
        ids=existing,
        cluster_method="dbscan",
        dbscan_min_samples=5,
        plot=True,
    )
    split_rows = read_csv(Path(summary["split_csv"]))
    assert len(split_rows) == len(existing)
    if zsv_ids:
        held = [r for r in split_rows if r["ID"] in zsv_set]
        assert held and all(r["train_test"] == "zsv" for r in held)
    assert summary["plot"] is not None
    assert Path(summary["feature_table"]).is_file()
