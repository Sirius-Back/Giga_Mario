"""LegNet layer dims, hook consistency, AGCT RC."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

VENDOR = Path("software/human_legnet")
pytestmark = pytest.mark.skipif(
    not VENDOR.is_dir(), reason="software/human_legnet missing"
)


@pytest.fixture(scope="module")
def legnet():
    from src.embed.legnet_extract import build_default_legnet

    torch.manual_seed(0)
    m = build_default_legnet(in_ch=4)
    m.eval()
    return m


def test_layer_dims_and_map_length(legnet):
    from src.embed import LAYER_DIMS, SEQ_LEN
    from src.embed.legnet_extract import LegNetLayerExtractor, encode_agct

    seq = "ACGT" * (SEQ_LEN // 4) + "A" * (SEQ_LEN % 4)
    x = encode_agct(seq).unsqueeze(0)
    with LegNetLayerExtractor(legnet, device="cpu") as ex:
        # Inspect raw stage maps
        _ = ex.forward_once(x)
        assert ex._feats["_stage0"].shape == (1, 80, SEQ_LEN // 2)
        assert ex._feats["_stage3"].shape[2] == 14  # 230→14 after 4 pools
        out = ex.extract_tensor(x, rc_average=False)
    for k, d in LAYER_DIMS.items():
        assert out[k].shape == (1, d), (k, out[k].shape)


def test_pooled_matches_manual(legnet):
    from src.embed import SEQ_LEN
    from src.embed.legnet_extract import (
        LegNetLayerExtractor,
        encode_agct,
        pooled_manual,
    )

    rng = np.random.default_rng(1)
    bases = np.array(list("AGCT"))
    seq = "".join(rng.choice(bases, size=SEQ_LEN))
    x = encode_agct(seq).unsqueeze(0)
    with LegNetLayerExtractor(legnet, device="cpu", layers=("pooled",)) as ex:
        got = ex.extract_tensor(x, rc_average=False)["pooled"]
    ref = pooled_manual(legnet, x)
    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-5)


def test_agct_rc_flip_and_average(legnet):
    from src.embed import SEQ_LEN
    from src.embed.legnet_extract import (
        LegNetLayerExtractor,
        encode_agct,
        reverse_complement_onehot,
    )

    seq = "A" * 100 + "G" * 50 + "C" * 40 + "T" * 40
    assert len(seq) == SEQ_LEN
    x = encode_agct(seq).unsqueeze(0)
    # Channel 0=A, 3=T: RC of poly-A prefix → poly-T on channel 3 at end
    rc = reverse_complement_onehot(x)
    assert torch.allclose(rc[0, 3, -100:], torch.ones(100))
    with LegNetLayerExtractor(legnet, device="cpu", layers=("pooled",)) as ex:
        avg = ex.extract_tensor(x, rc_average=True)["pooled"]
        a = ex.forward_once(x)["pooled"]
        b = ex.forward_once(rc)["pooled"]
        ref = 0.5 * (a + b)
    assert torch.allclose(avg, ref, atol=1e-6)


def test_extract_batch_numpy_shapes(legnet):
    from src.embed import DEFAULT_LAYERS, LAYER_DIMS, SEQ_LEN
    from src.embed.legnet_extract import LegNetLayerExtractor

    seqs = ["ACGTN" * 46]  # 230
    assert all(len(s) == SEQ_LEN for s in seqs)
    with LegNetLayerExtractor(legnet, device="cpu", layers=DEFAULT_LAYERS) as ex:
        out = ex.extract_batch(seqs)
    for k in DEFAULT_LAYERS:
        assert out[k].shape == (1, LAYER_DIMS[k])
        assert out[k].dtype == np.float32
