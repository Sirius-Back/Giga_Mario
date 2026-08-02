"""VGAE pack / model / homology firewall / early-stop policy tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.splits.vgae.assign import size_constrained_assign
from src.splits.vgae.graph_data import (
    assert_no_homology_features,
    build_compositional_features,
    pack_region_graph,
)
from src.splits.vgae.homology_loss import compute_l_hom, load_homology_groups, sd_random_from_labels
from src.splits.vgae.model import ClassicVGAE, soft_role_probs


def test_assert_no_homology_features_blocks_leakage() -> None:
    with pytest.raises(ValueError, match="homology"):
        assert_no_homology_features(["GC_pct", "orthogroup_id"])
    assert_no_homology_features(["GC_pct", "kmer_AAAAA"])


def test_build_compositional_features_shapes() -> None:
    ids = ["a", "b", "c"]
    seqs = {"a": "ACGTAC", "b": "GGGGGG", "c": "ATATAT"}
    x, names = build_compositional_features(ids, seqs, k=2)
    assert x.shape == (3, 1 + 16)
    assert names[0] == "GC_pct"
    assert_no_homology_features(names)


def test_size_constrained_assign_ratios() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(100, 3))
    labels = size_constrained_assign(scores, ratios=(3, 1, 1), seed=42)
    assert len(labels) == 100
    assert labels.count("train") == 60
    assert labels.count("test") == 20
    assert labels.count("val") == 20


def test_classic_vgae_forward_and_losses() -> None:
    n, f = 32, 8
    x = torch.randn(n, f)
    # line graph edges
    u = torch.arange(0, n - 1, dtype=torch.long)
    v = u + 1
    edge_index = torch.stack([u, v], dim=0)
    edge_weight = torch.ones(n - 1)
    model = ClassicVGAE(f, hidden_dim=16, latent_dim=8)
    out = model(x, edge_index, edge_weight)
    assert out["z"].shape == (n, 8)
    recon = ClassicVGAE.recon_loss_neg_sample(out["z"], edge_index, edge_weight)
    kl = ClassicVGAE.kl_loss(out["mu"], out["logstd"])
    assert torch.isfinite(recon)
    assert torch.isfinite(kl)
    soft = soft_role_probs(out["role_logits"])
    assert soft.shape == (n, 3)
    assert torch.allclose(soft.sum(dim=-1), torch.ones(n), atol=1e-5)


def test_sd_random_formula() -> None:
    # One group all in train, global 60/20/20 → high sd
    labels = ["train"] * 6 + ["test"] * 2 + ["val"] * 2
    groups = [np.arange(6)]  # first 6 all train
    mean, vals = sd_random_from_labels(labels, groups)
    assert mean > 0
    assert vals[0] > 0


def test_early_stop_policy_min_epochs() -> None:
    """Documented policy: no stop before min_epochs=25."""
    min_epochs, patience = 25, 10
    stale = 0
    stopped_at = None
    best = 1e9
    # Simulate hard L_hom never improving after epoch 1
    for epoch in range(1, 40):
        l_hom = 1.0
        if l_hom < best - 1e-6:
            best = l_hom
            stale = 0
        elif epoch >= min_epochs:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            stopped_at = epoch
            break
    assert stopped_at == 34  # first eligible epoch 25 → patience 10 hits at 34


def test_pack_region_graph_tiny(tmp_path: Path) -> None:
    # Tiny synthetic contingency graph + MARKED
    gdir = tmp_path / "graph"
    gdir.mkdir()
    ids = ["1", "2", "3", "4"]
    (gdir / "ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    np.savez_compressed(
        gdir / "contingency_graph.npz",
        cluster_ids=np.array([0, 0, 1, 1], dtype=np.int32),
        edge_u=np.array([0, 2], dtype=np.int32),
        edge_v=np.array([1, 3], dtype=np.int32),
        edge_w=np.array([2, 3], dtype=np.int32),
    )
    (gdir / "contingency_graph_meta.json").write_text(
        '{"k": 2, "n_ids": 4, "n_clusters": 2, "n_edges": 2}\n', encoding="utf-8"
    )
    marked = tmp_path / "MARKED"
    marked.mkdir()
    for rid, seq in zip(ids, ["ACGTAC", "ACGTAA", "GGGCCC", "GGGCCT"]):
        (marked / f"{rid}.fa").write_text(f">{rid}\n{seq}\n", encoding="utf-8")
    pack = pack_region_graph(gdir, marked, tmp_path / "pack", k=2)
    assert pack.n_nodes == 4
    assert pack.meta["homology_in_encoder"] is False
    assert (tmp_path / "pack" / "feature_meta.json").is_file()
    # Injected homology column must fail
    with pytest.raises(ValueError):
        assert_no_homology_features(list(pack.feature_names) + ["paralog_flag"])
