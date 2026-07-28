#!/usr/bin/env python3
"""Tests for src.get_mpra soft vs continuous modes."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.get_mpra import bin_fractions, log2_tpm_plus_one, read_wide_row, run_get_mpra


def _write_tpm(path: Path, genes: list[str], values: list[float]) -> None:
    path.write_text(
        ",".join(genes) + "\n" + ",".join(f"{v}" for v in values) + "\n",
        encoding="utf-8",
    )


def test_soft_mode_unchanged_default(tmp_path: Path) -> None:
    tpm = tmp_path / "tpm"
    out = tmp_path / "mpra_soft"
    tpm.mkdir()
    _write_tpm(tpm / "a.csv", ["g1", "g2", "g3"], [0.0, 1.0, 3.0])
    meta = run_get_mpra(tpm, out, n_bins=18, shared_scale=True, mode="soft")
    assert meta["mode"] == "soft"
    assert meta["n_bins"] == 18
    genes, vals = read_wide_row(out / "a.csv")
    assert genes == ["g1", "g2", "g3"]
    logs = log2_tpm_plus_one([0.0, 1.0, 3.0])
    expected = bin_fractions(logs, ymin=min(logs), ymax=max(logs), n_bins=18)
    assert vals == pytest.approx(expected, abs=1e-5)
    side = json.loads((out / "get_mpra_scale.json").read_text(encoding="utf-8"))
    assert side["mode"] == "soft"


def test_continuous_mode_raw_log2(tmp_path: Path) -> None:
    tpm = tmp_path / "tpm"
    out = tmp_path / "mpra_cont"
    tpm.mkdir()
    values = [0.0, 1.0, 3.0, 7.0]
    _write_tpm(tpm / "b.csv", ["g1", "g2", "g3", "g4"], values)
    meta = run_get_mpra(tpm, out, mode="continuous", shared_scale=True)
    assert meta["mode"] == "continuous"
    assert meta["n_bins"] is None
    assert meta["scale_01"] is False
    _, vals = read_wide_row(out / "b.csv")
    expected = [math.log2(v + 1.0) for v in values]
    assert vals == pytest.approx(expected, abs=1e-5)
    side = json.loads((out / "get_mpra_scale.json").read_text(encoding="utf-8"))
    assert "log2(TPM+1)" in side["transform"]


def test_continuous_scale_01(tmp_path: Path) -> None:
    tpm = tmp_path / "tpm"
    out = tmp_path / "mpra_01"
    tpm.mkdir()
    values = [0.0, 1.0, 3.0]
    _write_tpm(tpm / "c.csv", ["g1", "g2", "g3"], values)
    meta = run_get_mpra(
        tpm, out, mode="continuous", scale_01_flag=True, shared_scale=True
    )
    assert meta["scale_01"] is True
    _, vals = read_wide_row(out / "c.csv")
    assert min(vals) == pytest.approx(0.0, abs=1e-6)
    assert max(vals) == pytest.approx(1.0, abs=1e-6)


def test_scale_01_requires_continuous() -> None:
    with pytest.raises(ValueError, match="continuous"):
        run_get_mpra(Path("."), Path("."), mode="soft", scale_01_flag=True)
