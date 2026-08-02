"""Unit tests for pairwise geometry compare helpers."""

from __future__ import annotations

import numpy as np

from src.embed.distances import fit_train_stats
from src.embed.pairwise import (
    linear_cka,
    orthogonal_procrustes,
    pairwise_distance_matrix,
    rsa_spearman,
    transform_for_rdm,
)


def test_cka_identical_is_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 8))
    assert abs(linear_cka(x, x) - 1.0) < 1e-6


def test_cka_orthogonal_rotation_invariant():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(80, 6))
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    y = x @ q
    assert abs(linear_cka(x, y) - 1.0) < 1e-5


def test_procrustes_recovers_rotation():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 4))
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    y = x @ q
    y_al, disp = orthogonal_procrustes(x, y)
    assert disp < 1e-6
    assert np.allclose(x - x.mean(0), y_al, atol=1e-5)


def test_rsa_self():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(40, 5))
    stats = fit_train_stats(x)
    t = transform_for_rdm(x, stats, "centered_cosine")
    d = pairwise_distance_matrix(t, metric="centered_cosine")
    assert rsa_spearman(d, d) == 1.0 or abs(rsa_spearman(d, d) - 1.0) < 1e-9


def test_filter_loo_store_keys():
    from src.embed.pairwise import filter_loo_store_keys

    keys = [
        "run2_legnet_random",
        "run5_legnet_hashfrag",
        "run31_legnet_pangenome_k7_w0_100_loo5/fold0",
        "run31_legnet_pangenome_k7_w0_100_loo5/fold1",
        "run31_legnet_pangenome_k7_w0_100_loo5/fold4",
    ]
    assert filter_loo_store_keys(keys, None) == keys
    kept = filter_loo_store_keys(keys, 0)
    assert kept == [
        "run2_legnet_random",
        "run5_legnet_hashfrag",
        "run31_legnet_pangenome_k7_w0_100_loo5/fold0",
    ]


def test_short_run_label():
    from src.embed.pairwise import short_run_label

    assert short_run_label("run5_legnet_hashfrag") == "HASHFRAG"
    assert short_run_label("run11_legnet_kmer_k4") == "KMER K4"
    assert (
        short_run_label("run31_legnet_pangenome_k7_w0_100_loo5/fold0")
        == "PG K7 W0 100 LOO5"
    )
    assert short_run_label("r2_random") == "RANDOM"


def test_plot_lower_triangle_hypotenuse(tmp_path):
    from src.embed.pairwise import plot_lower_triangle_hypotenuse

    mat = np.array(
        [
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.7],
            [0.8, 0.7, 1.0],
        ]
    )
    pdf = tmp_path / "t.pdf"
    svg = tmp_path / "t.svg"
    plot_lower_triangle_hypotenuse(
        mat,
        ["a", "b", "c"],
        title="test",
        out_pdf=pdf,
        out_svg=svg,
        cmap="viridis",
        label_fontsize=10,
    )
    assert pdf.is_file() and pdf.stat().st_size > 500
    assert svg.is_file() and svg.stat().st_size > 500
