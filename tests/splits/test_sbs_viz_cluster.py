"""SBS viz + clustering-method switching tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.splits.sbs.assign import (
    CLUSTER_METHODS,
    cluster_feature_table,
    normalize_cluster_method,
)
from src.splits.sbs.features import FeatureTable
from src.splits.sbs.visualize import (
    DEFAULT_PLOT_N,
    _labels_are_numeric,
    plot_sbs_pca_diagnostics,
    prepare_plot_indices,
)


def test_normalize_cluster_method_aliases() -> None:
    assert normalize_cluster_method("K-Means") == "kmeans"
    assert normalize_cluster_method("elbow") == "kmeans_elbow"
    assert normalize_cluster_method("pca+kmeans") == "pca_kmeans"
    assert normalize_cluster_method("hclust") == "hierarchical"
    with pytest.raises(ValueError, match="Unknown cluster_method"):
        normalize_cluster_method("magic")


def test_cluster_method_switching_changes_labels() -> None:
    rng = np.random.default_rng(0)
    n = 80
    # Two blobs → kmeans(k=2) should differ from k=5
    mat = np.vstack(
        [
            rng.normal([0, 0], 0.2, size=(n // 2, 2)),
            rng.normal([5, 5], 0.2, size=(n // 2, 2)),
        ]
    )
    ids = tuple(str(i) for i in range(n))
    ft = FeatureTable(
        ids=ids, matrix=mat, feature_names=("GC_pct", "AAA_pct"), backend="test"
    )
    m2, meta2 = cluster_feature_table(ft, method="kmeans", n_clusters=2, seed=42)
    m5, meta5 = cluster_feature_table(ft, method="kmeans", n_clusters=5, seed=42)
    assert meta2["method_used"] == "kmeans"
    assert meta5["method_used"] == "kmeans"
    assert meta2["n_clusters"] == 2
    assert meta5["n_clusters"] == 5
    assert set(m2.values()) != set(m5.values()) or list(m2.values()) != list(m5.values())

    me, metae = cluster_feature_table(ft, method="kmeans_elbow", seed=42)
    assert metae["method_used"] == "kmeans_elbow"
    assert len(me) == n

    mp, metap = cluster_feature_table(ft, method="pca_kmeans", n_clusters=2, seed=42)
    assert metap["method_used"] == "pca_kmeans"
    assert len(mp) == n

    md, metad = cluster_feature_table(ft, method="dbscan", seed=42)
    assert metad["method_used"] == "dbscan"
    assert len(md) == n


def test_prepare_plot_indices_subsample_and_pc1_pc2_sort() -> None:
    rng = np.random.default_rng(1)
    coords = rng.normal(size=(50_000, 2))
    idx = prepare_plot_indices(coords, max_points=DEFAULT_PLOT_N, seed=42)
    assert len(idx) == DEFAULT_PLOT_N
    prod = coords[idx, 0] * coords[idx, 1]
    assert np.all(prod[:-1] <= prod[1:] + 1e-15)


def test_labels_are_numeric_fold() -> None:
    assert _labels_are_numeric(["0", "1", "2", "10"])
    assert _labels_are_numeric(["0", "1", "zsv"])  # ZSV holdouts allowed
    assert not _labels_are_numeric(["train", "val", "test"])
    assert not _labels_are_numeric(["0", "train"])


def test_plot_sbs_numeric_fold_gradient(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    n = 120
    rng = np.random.default_rng(2)
    mat = rng.normal(size=(n, 2))
    ids = tuple(str(i) for i in range(n))
    ft = FeatureTable(
        ids=ids, matrix=mat, feature_names=("GC_pct", "AAA_pct"), backend="test"
    )
    rows = [
        {
            "region": rid,
            "cluster": str(i % 5),
            "train_test": ["train", "val", "test"][i % 3],
            "fold": str(i % 5),
            "additional": "{}",
        }
        for i, rid in enumerate(ids)
    ]
    meta = plot_sbs_pca_diagnostics(
        ft, rows, outdir=tmp_path / "fig", seed=42, max_points=50
    )
    assert meta["max_points"] == 50
    assert meta["sort"] == "PC1*PC2"
    assert meta["panels"][0]["numeric_color"] is True
    assert (tmp_path / "fig" / "pca_by_cluster.pdf").is_file()


def test_supported_methods_exported() -> None:
    assert "dbscan" in CLUSTER_METHODS
    assert "kmeans_elbow" in CLUSTER_METHODS
    assert DEFAULT_PLOT_N == 10_000
