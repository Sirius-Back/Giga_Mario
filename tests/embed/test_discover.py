"""discover_legnet_runs skips BAD and finds fold units."""

from __future__ import annotations

import json
from pathlib import Path

from src.embed.discover import discover_legnet_runs


def _unit(root: Path) -> None:
    direct = root / "direct"
    best = direct / "best_model"
    best.mkdir(parents=True)
    (direct / "final_model").mkdir(exist_ok=True)
    (best / "pearson-epoch=00-val_pearson=0.1.ckpt").write_bytes(b"x")
    (best / "best_meta.json").write_text("{}", encoding="utf-8")
    cfg = {
        "stem_ch": 64,
        "stem_ks": 11,
        "ef_ks": 9,
        "ef_block_sizes": [80, 96, 112, 128],
        "resize_factor": 4,
        "pool_sizes": [2, 2, 2, 2],
        "use_reverse_channel": False,
        "reverse_augment": False,
        "use_shift": False,
        "max_shift": None,
        "max_lr": 0.01,
        "weight_decay": 0.1,
        "model_dir": str(direct),
        "data_path": "x",
        "epoch_num": 1,
        "device": 0,
        "seed": 1,
        "train_batch_size": 1,
        "valid_batch_size": 1,
        "num_workers": 0,
    }
    (direct / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (root / "split.csv").write_text(
        "ID|train_test|fold\n1|train|0\n2|test|0\n", encoding="utf-8"
    )
    leg = root / "legnet_input"
    leg.mkdir()
    seq = "A" * 230
    (leg / "all.tsv").write_text(
        "seq_id\tseq\tmean_value\tfold\trev\n"
        f"GENOME__1\t{seq}\t0\t3\t0\n"
        f"GENOME__2\t{seq}\t1\t1\t0\n",
        encoding="utf-8",
    )


def test_discover_skips_bad_and_finds_loo(tmp_path: Path):
    root = tmp_path / "legnet"
    root.mkdir()
    _unit(root / "run2_legnet_random")
    bad = root / "run13_legnet_kmer_k7_BAD_random_reassign_20260731"
    _unit(bad)
    loo = root / "run31_legnet_pangenome_k7_w0_100_loo5"
    loo.mkdir()
    _unit(loo / "fold0")
    _unit(loo / "fold1")

    found = discover_legnet_runs(root)
    keys = {r.key for r in found}
    assert "run2_legnet_random" in keys
    assert "run31_legnet_pangenome_k7_w0_100_loo5/fold0" in keys
    assert "run31_legnet_pangenome_k7_w0_100_loo5/fold1" in keys
    assert not any("BAD" in k for k in keys)

    fold0 = discover_legnet_runs(root, loo_fold=0)
    keys0 = {r.key for r in fold0}
    assert "run2_legnet_random" in keys0
    assert "run31_legnet_pangenome_k7_w0_100_loo5/fold0" in keys0
    assert "run31_legnet_pangenome_k7_w0_100_loo5/fold1" not in keys0
