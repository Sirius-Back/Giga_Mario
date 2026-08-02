"""GCN cascade resolve / reuse / infer (splits/GCN.md)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.pipeline.common import write_csv
from src.splits.gcn import (
    AmbiguousGcnModelError,
    MissingGcnModelError,
    infer_gcn_labeling,
    resolve_gcn_model,
    run_gcn_split_assign,
)
from src.splits.vgae.model import ClassicVGAE


def _write_mini_model_dir(
    root: Path,
    name: str,
    *,
    with_split: bool = True,
    with_ckpt: bool = True,
    grain: str = "region",
    k: int = 5,
    stage: str = "1",
) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    pack = d / "pack"
    pack.mkdir(parents=True, exist_ok=True)
    n, f = 10, 4
    ids = [str(i) for i in range(n)]
    x = np.random.default_rng(0).normal(size=(n, f)).astype(np.float32)
    np.savez_compressed(
        pack / "node_features.npz",
        x=x,
        feature_names=np.asarray([f"f{i}" for i in range(f)], dtype=object),
    )
    u = np.arange(0, n - 1, dtype=np.int32)
    v = u + 1
    w = np.ones(n - 1, dtype=np.float32)
    np.savez_compressed(
        pack / "edges_weighted.npz",
        edge_u=u,
        edge_v=v,
        edge_w=w,
        edge_w_raw=np.ones(n - 1, dtype=np.int32),
    )
    (pack / "ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    meta = {
        "format": "gigamario_vgae_pack_v1",
        "grain": grain,
        "k": k,
        "n_nodes": n,
        "feature_names": [f"f{i}" for i in range(f)],
        "homology_in_encoder": False,
    }
    (pack / "feature_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    (d / "train_meta.json").write_text(
        json.dumps(
            {"grain": grain, "k": k, "stage": stage, "loss_mode": "legacy"},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if with_ckpt:
        ckpt_dir = d / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model = ClassicVGAE(f, hidden_dim=8, latent_dim=4, n_roles=3)
        torch.save({"model": model.state_dict(), "epoch": 1}, ckpt_dir / "best.pt")
    if with_split:
        rows = []
        for i, rid in enumerate(ids):
            lab = "train" if i < 6 else ("test" if i < 8 else "val")
            rows.append({"ID": rid, "train_test": lab, "fold": f"gcn_{lab}"})
        write_csv(d / "split.csv", rows, ["ID", "train_test", "fold"])
    return d


def test_resolve_exact_name(tmp_path: Path) -> None:
    root = tmp_path / "VGAE"
    _write_mini_model_dir(root, "stage1_region_k5")
    got = resolve_gcn_model("stage1_region_k5", vgae_root=root)
    assert got.name == "stage1_region_k5"


def test_resolve_ambiguous_description(tmp_path: Path) -> None:
    root = tmp_path / "VGAE"
    _write_mini_model_dir(root, "stage1_region_k5", stage="1")
    _write_mini_model_dir(root, "stage1_region_k5_lossfix", stage="1")
    with pytest.raises(AmbiguousGcnModelError):
        resolve_gcn_model("stage1 region k5", vgae_root=root)


def test_resolve_missing(tmp_path: Path) -> None:
    root = tmp_path / "VGAE"
    root.mkdir()
    with pytest.raises(MissingGcnModelError):
        resolve_gcn_model("no_such_model", vgae_root=root)


def test_reuse_labeling_cascade(tmp_path: Path) -> None:
    root = tmp_path / "VGAE"
    _write_mini_model_dir(root, "stage1_region_k5", with_split=True)
    out = tmp_path / "out"
    summary = run_gcn_split_assign(
        outdir=out,
        model="stage1_region_k5",
        vgae_root=root,
        seed=42,
    )
    assert summary["source"] == "reuse"
    split = Path(summary["split_csv"])
    assert split.is_file()
    text = split.read_text(encoding="utf-8")
    assert "train_test" in text
    assert text.count("\n") >= 10


def test_infer_when_split_missing(tmp_path: Path) -> None:
    root = tmp_path / "VGAE"
    model_dir = _write_mini_model_dir(
        root, "stage1_region_k5", with_split=False, with_ckpt=True
    )
    out = tmp_path / "out"
    summary = run_gcn_split_assign(
        outdir=out,
        model="stage1_region_k5",
        vgae_root=root,
        seed=7,
        ratios=(3.0, 1.0, 1.0),
        device="cpu",
    )
    assert summary["source"] == "infer"
    split = Path(summary["split_csv"])
    assert split.is_file()
    assert (model_dir / "split.csv").is_file()
    lines = [
        ln for ln in split.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    assert len(lines) == 11  # header + 10
    labels = [ln.split("|")[1] for ln in lines[1:]]
    assert labels.count("train") == 6
    assert labels.count("test") == 2
    assert labels.count("val") == 2


def test_infer_helper_direct(tmp_path: Path) -> None:
    root = tmp_path / "VGAE"
    model_dir = _write_mini_model_dir(
        root, "mini", with_split=False, with_ckpt=True
    )
    path = infer_gcn_labeling(
        model_dir, outdir=tmp_path / "infer_out", seed=1, device="cpu"
    )
    assert path.is_file()


def test_train_path_requires_inputs(tmp_path: Path) -> None:
    root = tmp_path / "VGAE"
    root.mkdir()
    with pytest.raises(MissingGcnModelError):
        run_gcn_split_assign(
            outdir=tmp_path / "out",
            model="brand_new_model",
            vgae_root=root,
        )
