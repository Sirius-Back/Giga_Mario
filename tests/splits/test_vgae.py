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
from src.splits.vgae.homology_loss import (
    EmaTermNorm,
    compute_l_hom,
    load_homology_groups,
    sd_random_from_labels,
    select_groups_epoch_stable,
    soft_sd_random,
    soft_sd_random_weighted,
)
from src.splits.vgae.graph_data import project_features_fixed
from src.splits.vgae.model import (
    ClassicVGAE,
    gumbel_softmax_roles,
    gumbel_tau_schedule,
    kl_beta_schedule,
    soft_role_probs,
)
from src.splits.vgae.train import compose_objective


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


def test_gat_and_sage_vgae_forward() -> None:
    from src.splits.vgae.model import build_vgae

    n, f = 24, 6
    x = torch.randn(n, f)
    u = torch.arange(0, n - 1, dtype=torch.long)
    v = u + 1
    edge_index = torch.stack([u, v], dim=0)
    edge_weight = torch.ones(n - 1)
    for arch in ("gcn", "gat", "sage"):
        model = build_vgae(arch, f, hidden_dim=16, latent_dim=8, gat_heads=2)
        out = model(x, edge_index, edge_weight)
        assert out["z"].shape == (n, 8)
        assert out["role_logits"].shape == (n, 3)
        loss = type(model).recon_loss_neg_sample(out["z"], edge_index, edge_weight)
        loss = loss + type(model).kl_loss(out["mu"], out["logstd"])
        loss.backward()
        assert torch.isfinite(loss)


def test_contrastive_info_nce_and_augment() -> None:
    from src.splits.vgae.contrastive import augment_graph_view, info_nce_pairwise
    from src.splits.vgae.model import build_vgae, uses_contrastive

    n, f = 32, 8
    x = torch.randn(n, f, requires_grad=True)
    u = torch.arange(0, n - 1, dtype=torch.long)
    v = u + 1
    ei = torch.stack([u, v], dim=0)
    ew = torch.ones(n - 1)
    x1, ei1, ew1 = augment_graph_view(x, ei, ew, edge_drop=0.3, feat_mask=0.2)
    x2, ei2, ew2 = augment_graph_view(x, ei, ew, edge_drop=0.3, feat_mask=0.2)
    assert x1.shape == x.shape
    model = build_vgae("gcl", f, hidden_dim=16, latent_dim=8)
    assert uses_contrastive("gcl") and uses_contrastive("gcl_gat")
    assert model.architecture == "gcl"
    o1 = model(x1, ei1, ew1)
    o2 = model(x2, ei2, ew2)
    loss = info_nce_pairwise(o1["z"], o2["z"], temperature=0.5, max_nodes=16)
    loss.backward()
    assert torch.isfinite(loss)


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


def test_gumbel_softmax_roles_shape_and_sum() -> None:
    logits = torch.randn(16, 3)
    soft = gumbel_softmax_roles(logits, tau=0.7, hard=False)
    assert soft.shape == (16, 3)
    assert torch.allclose(soft.sum(dim=-1), torch.ones(16), atol=1e-5)
    # Legacy soft_role_probs still works unchanged
    plain = soft_role_probs(logits)
    assert plain.shape == (16, 3)


def test_kl_and_gumbel_schedules() -> None:
    assert kl_beta_schedule(1, beta_max=0.05, t_anneal=15) == pytest.approx(
        0.05 / 15
    )
    assert kl_beta_schedule(15, beta_max=0.05, t_anneal=15) == pytest.approx(0.05)
    assert kl_beta_schedule(100, beta_max=0.05, t_anneal=15) == pytest.approx(0.05)
    assert gumbel_tau_schedule(1, tau_start=1.0, tau_end=0.3, t_anneal=20) == pytest.approx(
        1.0
    )
    assert gumbel_tau_schedule(21, tau_start=1.0, tau_end=0.3, t_anneal=20) == pytest.approx(
        0.3
    )


def test_soft_sd_random_weighted_and_stable_subset() -> None:
    n = 40
    soft = torch.softmax(torch.randn(n, 3), dim=-1)
    groups = [np.arange(i, i + 4) for i in range(0, 36, 4)]  # 9 groups size 4
    a = soft_sd_random(soft, groups, max_groups=None)
    b = soft_sd_random_weighted(
        soft, groups, max_groups=None, subset_seed=0, weight_power=0.5
    )
    assert torch.isfinite(a) and torch.isfinite(b)
    # Epoch-stable subset: same seed → same groups
    s0 = select_groups_epoch_stable(groups, max_groups=3, seed=7)
    s1 = select_groups_epoch_stable(groups, max_groups=3, seed=7)
    assert len(s0) == 3
    assert all(np.array_equal(x, y) for x, y in zip(s0, s1))
    # Weighted L_hom path
    from src.splits.vgae.homology_loss import HomologyGroups

    hg = HomologyGroups(
        orthogroup=np.full(n, -1, dtype=np.int64),
        paragroup=np.full(n, -1, dtype=np.int64),
        ortho_groups=tuple(groups[:4]),
        para_groups=tuple(groups[4:]),
    )
    out = compute_l_hom(
        soft, hg, soft=True, weighted=True, max_groups=8, subset_seed=3
    )
    assert torch.isfinite(out["l_hom"])
    # Legacy unweighted still works
    legacy = compute_l_hom(soft, hg, soft=True)
    assert torch.isfinite(legacy["l_hom"])


def test_robust_and_log_balance_aggs() -> None:
    n = 40
    soft = torch.softmax(torch.randn(n, 3), dim=-1).detach().requires_grad_(True)
    groups = [np.arange(i, i + 4) for i in range(0, 36, 4)]
    from src.splits.vgae.homology_loss import (
        HomologyGroups,
        evaluate_split_all_aggs,
        soft_sd_random_agg,
        sd_group_balance_report,
    )

    hg = HomologyGroups(
        orthogroup=np.full(n, -1, dtype=np.int64),
        paragroup=np.full(n, -1, dtype=np.int64),
        ortho_groups=tuple(groups[:4]),
        para_groups=tuple(groups[4:]),
    )
    for agg in ("robust", "log_balance", "weighted", "mean"):
        out = compute_l_hom(soft, hg, soft=True, agg=agg, max_groups=8, subset_seed=1)
        assert torch.isfinite(out["l_hom"])
        out["l_hom"].backward(retain_graph=True)
    labels = ["train"] * 24 + ["test"] * 8 + ["val"] * 8
    hard = evaluate_split_all_aggs(labels, hg)
    assert set(hard) >= {"mean", "weighted", "robust", "log_balance"}
    bal = sd_group_balance_report(labels, hg)
    assert bal["ortho"]["n_groups"] >= 1
    z = soft_sd_random_agg(soft, groups, agg="robust", max_groups=None, subset_seed=0)
    assert torch.isfinite(z)



def test_ema_term_norm_and_homology_first_compose() -> None:
    ema = EmaTermNorm(decay=0.9)
    recon = torch.tensor(80.0, requires_grad=True)
    kl = torch.tensor(20.0, requires_grad=True)
    l_hom = torch.tensor(-0.2, requires_grad=True)
    size = torch.tensor(0.01, requires_grad=True)
    composed = compose_objective(
        recon=recon,
        kl=kl,
        l_hom=l_hom,
        size=size,
        loss_mode="homology_first",
        epoch=1,
        beta_kl=1.0,
        lambda_hom=25.0,
        lambda_size=1.0,
        alpha_recon=0.3,
        beta_kl_max=0.05,
        kl_anneal_epochs=15,
        ema=ema,
    )
    loss = composed["loss"]
    assert torch.isfinite(loss)
    # After first update, ema ≈ |term|; recon_norm ≈ 1
    assert composed["recon_norm"] == pytest.approx(1.0, abs=1e-5)
    # Homology term magnitude should dominate early (β tiny, α_r=0.3)
    assert abs(composed["term_hom"]) > abs(composed["term_recon"])
    # Legacy path still available
    legacy = compose_objective(
        recon=recon.detach(),
        kl=kl.detach(),
        l_hom=l_hom.detach(),
        size=size.detach(),
        loss_mode="legacy",
        epoch=1,
        beta_kl=1.0,
        lambda_hom=1.0,
        lambda_size=1.0,
        alpha_recon=1.0,
        beta_kl_max=1.0,
        kl_anneal_epochs=0,
        ema=None,
    )
    assert torch.isfinite(legacy["loss"])


def test_project_features_fixed_preserves_gc() -> None:
    rng = np.random.default_rng(0)
    x = rng.random((20, 1 + 16), dtype=np.float32)
    names = ("GC_pct",) + tuple(f"kmer_{i}" for i in range(16))
    x2, names2, meta = project_features_fixed(x, names, project_dim=8, seed=1)
    assert meta["applied"] is True
    assert x2.shape == (20, 8)
    assert names2[0] == "GC_pct"
    assert np.allclose(x2[:, 0], x[:, 0])


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
