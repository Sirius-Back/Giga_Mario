"""Unit tests for LOO CV cluster packing and round labels."""
from __future__ import annotations

from src.pipeline.loo_cv import (
    assign_clusters_to_cv_folds,
    build_cv_master_rows,
    split_rows_for_loo_round,
    summarize_split_rows,
)


def _rows(n_clusters: int = 10, per: int = 4) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    i = 0
    for c in range(n_clusters):
        for _ in range(per):
            out.append(
                {
                    "region": str(i),
                    "cluster": str(c),
                    "fold": str(c),
                    "train_test": "train",
                }
            )
            i += 1
    # zsv
    for j in range(3):
        out.append(
            {
                "region": f"z{j}",
                "cluster": "zsv",
                "fold": "zsv",
                "train_test": "zsv",
            }
        )
    return out


def test_assign_clusters_balanced():
    rows = _rows(10, 4)
    m = assign_clusters_to_cv_folds(rows, n_cv=5, seed=42)
    assert len(m) == 10
    assert set(m.values()) == {0, 1, 2, 3, 4}


def test_loo_round_ratios_approx_3_1_1():
    rows = _rows(20, 5)  # 100 regions + zsv
    m = assign_clusters_to_cv_folds(rows, n_cv=5, seed=42)
    master = build_cv_master_rows(rows, m)
    for holdout in range(5):
        split = split_rows_for_loo_round(master, holdout=holdout, n_cv=5)
        s = summarize_split_rows(split)
        assert s["zsv"] == 3
        total = s["train"] + s["test"] + s["val"]
        assert total == 100
        # ≈ 60/20/20 within 15% absolute of total
        assert abs(s["train"] / total - 0.6) < 0.15
        assert abs(s["test"] / total - 0.2) < 0.15
        assert abs(s["val"] / total - 0.2) < 0.15
        # homology fold preserved on non-zsv
        non_z = [r for r in split if r["train_test"] != "zsv"]
        assert all(r["fold"] != "" for r in non_z)
