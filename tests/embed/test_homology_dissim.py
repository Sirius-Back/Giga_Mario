"""Synthetic tests for ortholog/paralog embedding dissimilarity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.embed.homology_dissim import (
    HomDissimScores,
    marked_id_from_seq_id,
    mean_pairwise_cosine_distance,
    score_store_layer,
    split_method_from_run,
    write_per_store_tsv,
    write_ranking_tsv,
)
from src.embed.store import EmbedStore
from src.splits.vgae.homology_loss import HomologyGroups


def test_marked_id_from_seq_id():
    assert marked_id_from_seq_id("ENCSR161RSX__3") == "3"
    assert marked_id_from_seq_id("SRX1__42") == "42"
    assert marked_id_from_seq_id("99") == "99"
    assert marked_id_from_seq_id("bad") is None
    assert marked_id_from_seq_id("") is None


def test_split_method_from_run():
    assert split_method_from_run("run2_legnet_random") == "random"
    assert split_method_from_run("run24_legnet_paralogs_only") == "paralogs_only"
    assert (
        split_method_from_run("run31_legnet_pangenome_k7_w0_100_loo5/fold0")
        == "pangenome_k7_w0_100_loo5"
    )
    assert split_method_from_run("other") == "other"


def test_mean_pairwise_cosine_distance_identical():
    x = np.ones((4, 8), dtype=np.float64)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    d, n = mean_pairwise_cosine_distance(x)
    assert n == 6
    assert d == pytest.approx(0.0, abs=1e-9)


def test_mean_pairwise_cosine_distance_orthogonal():
    x = np.eye(3, dtype=np.float64)
    d, n = mean_pairwise_cosine_distance(x)
    assert n == 3
    assert d == pytest.approx(1.0, abs=1e-9)


def _make_groups(
    n: int,
    ortho_lists: list[list[int]],
    para_lists: list[list[int]],
) -> HomologyGroups:
    ortho = np.full(n, -1, dtype=np.int64)
    para = np.full(n, -1, dtype=np.int64)
    for gid, idxs in enumerate(ortho_lists):
        for i in idxs:
            ortho[i] = gid
    for gid, idxs in enumerate(para_lists):
        for i in idxs:
            para[i] = gid
    return HomologyGroups(
        orthogroup=ortho,
        paragroup=para,
        ortho_groups=tuple(np.asarray(g, dtype=np.int64) for g in ortho_lists),
        para_groups=tuple(np.asarray(g, dtype=np.int64) for g in para_lists),
    )


def test_score_store_layer_ortho_tight_para_spread(tmp_path: Path):
    """Orthologs share a direction; paralogs are near-orthogonal → D_hom_emb > 0."""
    rng = np.random.default_rng(0)
    d = 16
    # 3 OG pairs along axes 0,1,2 (tight within pair)
    # 1 PG of 4 points on distinct axes (spread)
    n = 10
    x = np.zeros((n, d), dtype=np.float32)
    # OG0: rows 0,1
    x[0, 0] = 1.0
    x[1, 0] = 1.0
    # OG1: rows 2,3
    x[2, 1] = 1.0
    x[3, 1] = 1.0
    # OG2: rows 4,5
    x[4, 2] = 1.0
    x[5, 2] = 1.0
    # PG0: rows 6,7,8,9 — mutually orthogonal-ish
    for i, ax in enumerate((3, 4, 5, 6)):
        x[6 + i, ax] = 1.0
    # small noise
    x += 0.01 * rng.normal(size=x.shape).astype(np.float32)

    ids = np.asarray([f"S__{i}" for i in range(n)], dtype=object)
    roles = np.zeros(n, dtype=np.int8)  # all train
    store = EmbedStore(
        out_dir=tmp_path / "store",
        ids=ids,
        roles=roles,
        layers={"pooled": x},
    )
    groups = _make_groups(
        n,
        ortho_lists=[[0, 1], [2, 3], [4, 5]],
        para_lists=[[6, 7, 8, 9]],
    )
    scores = score_store_layer(store, "pooled", groups, run_name="run99_legnet_synth")
    assert scores.split_method == "synth"
    assert scores.mean_d_ortho < scores.mean_d_para
    assert scores.D_hom_emb > 0.3
    assert scores.n_og == 3
    assert scores.n_pg == 1


def test_write_ranking_tsv(tmp_path: Path):
    rows = [
        HomDissimScores(
            run="run1_legnet_a",
            split_method="a",
            layer="pooled",
            n_ids=10,
            n_mapped=8,
            coverage=0.8,
            n_og=2,
            n_pg=2,
            mean_d_ortho=0.2,
            mean_d_para=0.5,
            sem_d_ortho=0.01,
            sem_d_para=0.02,
            D_hom_emb=0.3,
            n_pairs_ortho=5,
            n_pairs_para=5,
        ),
        HomDissimScores(
            run="run2_legnet_b",
            split_method="b",
            layer="pooled",
            n_ids=10,
            n_mapped=8,
            coverage=0.8,
            n_og=2,
            n_pg=2,
            mean_d_ortho=0.1,
            mean_d_para=0.6,
            sem_d_ortho=0.01,
            sem_d_para=0.02,
            D_hom_emb=0.5,
            n_pairs_ortho=5,
            n_pairs_para=5,
        ),
        HomDissimScores(
            run="run1_legnet_a",
            split_method="a",
            layer="stage0",
            n_ids=10,
            n_mapped=8,
            coverage=0.8,
            n_og=2,
            n_pg=2,
            mean_d_ortho=0.2,
            mean_d_para=0.4,
            sem_d_ortho=0.01,
            sem_d_para=0.02,
            D_hom_emb=0.2,
            n_pairs_ortho=5,
            n_pairs_para=5,
        ),
    ]
    per = write_per_store_tsv(rows, tmp_path / "per.tsv")
    rank = write_ranking_tsv(rows, tmp_path / "ranking.tsv", layer="pooled")
    assert per.is_file()
    text = rank.read_text(encoding="utf-8").strip().splitlines()
    assert text[0].startswith("run\t")
    # best D_hom_emb first
    assert "run2_legnet_b" in text[1]
    assert text[1].endswith("\t1") or "\t1" in text[1]
