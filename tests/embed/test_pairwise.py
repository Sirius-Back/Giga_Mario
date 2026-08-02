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
