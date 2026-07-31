"""Shared train:test:val ratio contract for /split and /split-generate."""
from __future__ import annotations

from src.splits.common import TEST_FRACTION, VAL_FRACTION_OF_TRAINPOOL, train_test_val_weights
from src.splits.sbs.assign import _assign_folds_to_train_test


def test_default_weights_match_caduceus_fractions() -> None:
    w = train_test_val_weights(None)
    assert abs(w["test"] - TEST_FRACTION) < 1e-12
    assert abs(w["val"] - (1.0 - TEST_FRACTION) * VAL_FRACTION_OF_TRAINPOOL) < 1e-12
    assert abs(sum(w.values()) - 1.0) < 1e-12


def test_explicit_ratios_are_train_test_val_order() -> None:
    w = train_test_val_weights((1, 1, 3))
    assert w == {"train": 1.0, "test": 1.0, "val": 3.0}


def test_sbs_fold_sizes_uses_train_test_val_order() -> None:
    # Ten equal folds; weights 1:1:3 train:test:val → ~2 train, ~2 test, ~6 val folds.
    fold_ids = [str(i) for i in range(10)]
    sizes = {fid: 100 for fid in fold_ids}
    labels = _assign_folds_to_train_test(
        fold_ids, seed=42, fold_strata=None, ratios=(1, 1, 3), fold_sizes=sizes
    )
    from collections import Counter

    c = Counter(labels.values())
    assert c["train"] == 2
    assert c["test"] == 2
    assert c["val"] == 6
