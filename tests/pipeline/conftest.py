"""Shared fixtures for pipeline I/O contract tests."""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MINI_RAW = ROOT / "tests" / "fixtures" / "mini_raw"
READY_V2_MOCK = ROOT / "tests" / "fixtures" / "ready_v2_mock"


@pytest.fixture
def mini_raw() -> Path:
    assert (MINI_RAW / "gtf").is_dir()
    assert (MINI_RAW / "fna").is_dir()
    assert (MINI_RAW / "tpm").is_dir()
    return MINI_RAW


@pytest.fixture
def ready_v2_mock() -> Path:
    assert (READY_V2_MOCK / "ready.csv").is_file()
    return READY_V2_MOCK


@pytest.fixture
def id_csv(tmp_path: Path, mini_raw: Path) -> Path:
    from src.pipeline.id_gen import run_id_gen

    out = tmp_path / "ids"
    return run_id_gen(mini_raw / "gtf", gtf_column="transcript", outdir=out)
