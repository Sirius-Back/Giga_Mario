"""Optional smoke: load real run2 checkpoint and embed 8 sequences."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.embed.discover import discover_legnet_runs

RUNS = Path("runs_unif/legnet")
VENDOR = Path("software/human_legnet")


def _run2():
    if not RUNS.is_dir():
        return None
    for r in discover_legnet_runs(RUNS):
        if r.run_name == "run2_legnet_random" and r.fold is None:
            return r
    return None


def _has_lightning() -> bool:
    try:
        import lightning.pytorch  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not VENDOR.is_dir(), reason="vendor missing")
@pytest.mark.skipif(not _has_lightning(), reason="lightning not installed (use conda env legnet)")
@pytest.mark.skipif(_run2() is None, reason="run2_legnet_random checkpoint missing")
def test_real_checkpoint_eight_seqs():
    import torch

    from src.embed.legnet_extract import LegNetLayerExtractor, load_lit_model
    from src.embed.validate import load_tsv_index

    run = _run2()
    assert run is not None
    idx, _ = load_tsv_index(run.legnet_tsv)
    seqs = [idx[k] for k in sorted(idx)[:8]]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    lit = load_lit_model(run, map_location=device)
    with LegNetLayerExtractor(lit.model, device=device) as ex:
        out = ex.extract_batch(seqs)
    assert out["pooled"].shape == (8, 256)
    assert out["stage1_2"].shape == (8, 416)
