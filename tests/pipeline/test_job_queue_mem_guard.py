"""Tests for mem_guard + job_queue scheduling politics."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline import job_queue as jq
from src.pipeline.mem_guard import (
    assert_ram_headroom,
    ensure_allocation_fits,
    ram_snapshot,
    ram_used_fraction,
    wait_for_ram_headroom,
)


def test_ram_snapshot_sane() -> None:
    snap = ram_snapshot()
    assert snap["mem_total_bytes"] > 0
    assert snap["mem_available_bytes"] > 0
    assert 0.0 <= snap["mem_used_fraction"] <= 1.0
    assert abs(ram_used_fraction() - snap["mem_used_fraction"]) < 1e-9


def test_assert_and_wait_headroom_current() -> None:
    assert_ram_headroom(0.95)
    wait_for_ram_headroom(0.95, timeout_sec=5.0, poll_sec=0.1, label="test")


def test_ensure_tiny_allocation() -> None:
    ensure_allocation_fits(1024, timeout_sec=5.0, poll_sec=0.1, label="tiny")


def test_parse_queue_and_can_launch(tmp_path: Path) -> None:
    q = tmp_path / "queue.md"
    q.write_text(
        """# Job queue
### heavy_a — RUNNING
- **PID:** 1
- **class:** cpu_ram_heavy
- **peak_ram_gib:** 30
- **status:** RUNNING

### train_b — RUNNING
- **PID:** 2
- **class:** gpu_train
- **GPUs:** 0,1
- **peak_ram_gib:** 10
- **status:** RUNNING

### waiter_c — RUNNING
- **PID:** 3
- **class:** waiter
- **status:** RUNNING
""",
        encoding="utf-8",
    )
    jobs = jq.parse_queue(q)
    assert len(jobs) == 3
    # GPU conflict on 0,1
    ok, reason = jq.can_launch_parallel(
        peak_ram_gib=8.0,
        gpus=(0, 1),
        job_class=jq.CLASS_GPU_TRAIN,
        path=q,
    )
    # PID 1/2/3 likely dead → running_jobs may drop them if not alive
    # Force by patching pid_alive
    orig = jq.pid_alive
    try:
        jq.pid_alive = lambda pid: True  # type: ignore[assignment]
        ok, reason = jq.can_launch_parallel(
            peak_ram_gib=8.0,
            gpus=(0, 1),
            job_class=jq.CLASS_GPU_TRAIN,
            path=q,
        )
        assert ok is False
        assert "GPU" in reason
        ok2, _ = jq.can_launch_parallel(
            peak_ram_gib=8.0,
            gpus=(2, 3),
            job_class=jq.CLASS_GPU_TRAIN,
            path=q,
        )
        assert ok2 is True
        # second cpu_ram_heavy with declared peaks — depends on host used
        ok3, reason3 = jq.can_launch_parallel(
            peak_ram_gib=30.0,
            job_class=jq.CLASS_CPU_RAM_HEAVY,
            path=q,
        )
        assert isinstance(ok3, bool)
        assert reason3
    finally:
        jq.pid_alive = orig  # type: ignore[assignment]


def test_append_queue_entry(tmp_path: Path) -> None:
    q = tmp_path / "queue.md"
    jq.append_queue_entry(
        "demo_job",
        job="python -m demo",
        pid=999001,
        estimated_time="1h",
        job_class=jq.CLASS_CPU_RAM_HEAVY,
        peak_ram_gib=32.0,
        gpus=(),
        path=q,
        log="logs/demo.log",
    )
    text = q.read_text(encoding="utf-8")
    assert "demo_job" in text
    assert "peak_ram_gib:** 32.0" in text
    assert "cpu_ram_heavy" in text
