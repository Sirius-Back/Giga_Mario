"""Distance invariants and chunked max-sim vs reference."""

from __future__ import annotations

import numpy as np
import pytest

from src.embed.distances import (
    fit_train_stats,
    pairwise_max_similarity,
    prepare_metric_matrix,
    transform_centered_l2,
)
from src.embed.leakage import auc_l_tau, l_tau, summarize_scores


def test_centered_cosine_self_sim():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 16))
    stats = fit_train_stats(x)
    y = transform_centered_l2(x, stats)
    # diagonal of y@y.T ≈ 1
    sims = (y * y).sum(axis=1)
    assert np.allclose(sims, 1.0, atol=1e-6)


def test_pairwise_max_matches_naive():
    rng = np.random.default_rng(1)
    train = rng.normal(size=(40, 8))
    test = rng.normal(size=(12, 8))
    stats = fit_train_stats(train)
    g = prepare_metric_matrix(train, stats, "centered_cosine")
    q = prepare_metric_matrix(test, stats, "centered_cosine")
    got = pairwise_max_similarity(q, g, metric="centered_cosine", chunk=5)
    naive = (q @ g.T).max(axis=1)
    assert np.allclose(got, naive, atol=1e-8)


def test_whitening_reduces_channel_corr():
    rng = np.random.default_rng(2)
    # Moderately correlated channels 0–1 (ridge-ZCA is exact for Σ+εI, not Σ)
    a = rng.normal(size=(800, 1))
    b = 0.7 * a + 0.3 * rng.normal(size=(800, 1))
    x = np.concatenate([a, b, rng.normal(size=(800, 2))], axis=1)
    stats = fit_train_stats(x, ridge=1e-6)
    assert stats.whiten is not None
    assert stats.cond is not None and stats.cond > 1.5
    raw01 = abs(np.corrcoef(x.T)[0, 1])
    yw = (x - stats.mean) @ stats.whiten
    w01 = abs(np.corrcoef(yw.T)[0, 1])
    assert raw01 > 0.6
    assert w01 < raw01 * 0.5


def test_l_tau_and_auc():
    scores = np.array([0.2, 0.5, 0.9, 1.0])
    tau, lt = l_tau(scores, tau_grid=np.array([0.0, 0.5, 0.9, 1.0]))
    assert lt[0] == pytest.approx(1.0)
    assert lt[1] == pytest.approx(0.75)
    assert lt[2] == pytest.approx(0.5)
    assert auc_l_tau(tau, lt) > 0
    summ = summarize_scores(scores, tau0=0.9)
    assert summ["P_ge_0.9"] == pytest.approx(0.5)
