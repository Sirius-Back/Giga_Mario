"""Caduceus embed dims / RCPS helpers / discover."""

from __future__ import annotations

from pathlib import Path

import torch

from src.embed.caduceus_extract import (
    CADUCEUS_LAYER_DIMS,
    _mean_max_pool,
    rcps_to_strand_avg,
)
from src.embed.discover_caduceus import discover_caduceus_runs
from src.embed.pairwise import short_run_label


def test_rcps_to_strand_avg_shape():
    b, l, d = 2, 8, 256
    fwd = torch.randn(b, l, d)
    rc = torch.randn(b, l, d)
    # packed as Caduceus RCPS: [fwd | flip(rc)]
    packed = torch.cat([fwd, torch.flip(rc, dims=[1, 2])], dim=-1)
    avg = rcps_to_strand_avg(packed, d)
    assert avg.shape == (b, l, d)
    # recover: avg should equal 0.5*(fwd + rc) after the model flip convention
    expect = 0.5 * (fwd + rc)
    assert torch.allclose(avg, expect, atol=1e-5)


def test_mean_max_pool_dim():
    x = torch.randn(3, 10, 256)
    y = _mean_max_pool(x)
    assert y.shape == (3, 512)


def test_short_run_label_caduceus():
    assert short_run_label("run16_caduceus_hashfrag") == "HASHFRAG"
    assert short_run_label("run1_caduceus_random") == "RANDOM"
    assert (
        short_run_label("run32_caduceus_pangenome_k7_w0_100_loo5/fold0")
        == "PG K7 W0 100 LOO5"
    )


def test_discover_caduceus_loo_fold0():
    root = Path("runs_unif/caduceus")
    if not root.is_dir():
        return
    all_runs = discover_caduceus_runs(root, loo_fold=None)
    fold0 = discover_caduceus_runs(root, loo_fold=0)
    assert any("hashfrag" in r.key for r in fold0) or any(
        "hashfrag" in r.key for r in all_runs
    )
    for r in fold0:
        if r.fold is not None:
            assert r.fold == 0
    # dims table sanity
    assert CADUCEUS_LAYER_DIMS["pooled"] == 256
    assert CADUCEUS_LAYER_DIMS["stage0"] == 512
