"""Helpers for project-root ``queue.md`` (local-job-queue politics).

Agents and long runners use this to:
- register large jobs
- decide **parallel vs sequential** from RAM peaks + GPU indices
- wait for headroom without instant death
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from src.pipeline.mem_guard import (
    DEFAULT_MAX_USED_FRACTION,
    ram_snapshot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUEUE_PATH = PROJECT_ROOT / "queue.md"

# Job classes for scheduling politics (see local-job-queue.mdc).
CLASS_CPU_RAM_HEAVY = "cpu_ram_heavy"  # full-panel k-mer / large dense matrices
CLASS_GPU_TRAIN = "gpu_train"
CLASS_WAITER = "waiter"  # sleeps until deps; negligible RAM
CLASS_LIGHT = "light"


@dataclass(frozen=True)
class QueueJob:
    name: str
    status: str
    pid: int | None
    gpus: tuple[int, ...]
    job_class: str
    peak_ram_gib: float | None
    raw_block: str


_HEADING_RE = re.compile(
    r"^###\s+(?P<name>.+?)\s+[—-]\s+(?P<status>RUNNING|COMPLETED|FAILED|WAITED-OUT)\s*$",
    re.MULTILINE,
)
_PID_RE = re.compile(r"\*\*PID:\*\*\s*(?P<pid>\d+)", re.IGNORECASE)
_GPU_RE = re.compile(
    r"\*\*GPUs?:\*\*\s*([0-9,\s]+)"
    r"|"
    r"(?:CUDA_VISIBLE_DEVICES)[=:\s]*([0-9,\s]+)"
    r"|"
    r"(?:GPU|GPUs)[:\s]+([0-9,\s]+)",
    re.IGNORECASE,
)
_CLASS_RE = re.compile(r"\*\*class:\*\*\s*([a-z_]+)", re.IGNORECASE)
_PEAK_RE = re.compile(
    r"\*\*peak_ram_gib:\*\*\s*([0-9.]+)",
    re.IGNORECASE,
)


def queue_path(root: Path | None = None) -> Path:
    return (root or PROJECT_ROOT) / "queue.md"


def parse_queue(path: Path | None = None) -> list[QueueJob]:
    """Parse queue.md entries (best-effort; never raises on malformed blocks)."""
    p = path or queue_path()
    if not p.is_file():
        return []
    text = p.read_text(encoding="utf-8")
    matches = list(_HEADING_RE.finditer(text))
    jobs: list[QueueJob] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        pid_m = _PID_RE.search(block)
        pid = int(pid_m.group("pid")) if pid_m else None
        gpus: tuple[int, ...] = ()
        gpu_m = _GPU_RE.search(block)
        if gpu_m:
            raw = next(g for g in gpu_m.groups() if g)
            nums = [int(x) for x in re.findall(r"\d+", raw)]
            gpus = tuple(nums)
        class_m = _CLASS_RE.search(block)
        job_class = class_m.group(1).lower() if class_m else CLASS_LIGHT
        peak_m = _PEAK_RE.search(block)
        peak = float(peak_m.group(1)) if peak_m else None
        jobs.append(
            QueueJob(
                name=m.group("name").strip(),
                status=m.group("status").strip().upper(),
                pid=pid,
                gpus=gpus,
                job_class=job_class,
                peak_ram_gib=peak,
                raw_block=block,
            )
        )
    return jobs


def pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def running_jobs(path: Path | None = None) -> list[QueueJob]:
    """RUNNING entries whose PID is missing (treat as active) or still alive."""
    out: list[QueueJob] = []
    for j in parse_queue(path):
        if j.status != "RUNNING":
            continue
        if j.pid is None or pid_alive(j.pid):
            out.append(j)
    return out


def gpus_conflict(a: Sequence[int], b: Sequence[int]) -> bool:
    if not a or not b:
        return False
    return bool(set(a) & set(b))


def can_launch_parallel(
    *,
    peak_ram_gib: float,
    gpus: Sequence[int] = (),
    job_class: str = CLASS_LIGHT,
    max_used_fraction: float = DEFAULT_MAX_USED_FRACTION,
    path: Path | None = None,
) -> tuple[bool, str]:
    """Return (ok, reason) for launching alongside current queue + host RAM.

    Politics:
    - Hard: projected host used ≤ max_used_fraction after adding peak_ram_gib.
    - GPU: refuse if requested GPUs overlap an alive RUNNING gpu_train.
    - cpu_ram_heavy: allow parallel only if peaks still fit under the RAM cap;
      prefer serialize when another cpu_ram_heavy is alive and peaks are unknown.
    - waiter / light: OK if RAM + GPU OK.
    """
    snap = ram_snapshot()
    total_gib = float(snap["mem_total_gib"])
    used_gib = total_gib * float(snap["mem_used_fraction"])
    projected = (used_gib + max(0.0, peak_ram_gib)) / total_gib if total_gib else 1.0
    if projected > max_used_fraction:
        return (
            False,
            f"projected used {100 * projected:.1f}% > {100 * max_used_fraction:.0f}% "
            f"(now {snap['mem_used_pct']:.1f}% + peak {peak_ram_gib:.1f} GiB)",
        )

    alive = running_jobs(path)
    for j in alive:
        if job_class == CLASS_GPU_TRAIN and j.job_class == CLASS_GPU_TRAIN:
            if gpus_conflict(gpus, j.gpus):
                return False, f"GPU conflict with {j.name} (PID {j.pid}) gpus={j.gpus}"
        if job_class == CLASS_CPU_RAM_HEAVY and j.job_class == CLASS_CPU_RAM_HEAVY:
            other_peak = j.peak_ram_gib
            if other_peak is None:
                return (
                    False,
                    f"another cpu_ram_heavy running ({j.name}, PID {j.pid}) "
                    "without peak_ram_gib — serialize",
                )
            both = used_gib + peak_ram_gib + other_peak
            # other_peak may already be partly in used_gib; use max of
            # (used+new) vs (used+new+other) conservatively only add new peak.
            # Conservative: assume other peak still fully resident beyond current used.
            # Too conservative kills parallelism — use: used + this_peak <= cap
            # already checked; only serialize if this_peak + other_peak + baseline
            # baseline ≈ used - min(other_rss_unknown). Simpler rule:
            if (used_gib + peak_ram_gib + 0.5 * other_peak) / total_gib > max_used_fraction:
                return (
                    False,
                    f"two cpu_ram_heavy peaks may exceed cap "
                    f"(this {peak_ram_gib:.1f} + other {other_peak:.1f} GiB)",
                )
    return True, "ok"


def wait_until_launchable(
    *,
    peak_ram_gib: float,
    gpus: Sequence[int] = (),
    job_class: str = CLASS_LIGHT,
    max_used_fraction: float = DEFAULT_MAX_USED_FRACTION,
    timeout_sec: float | None = 6 * 3600,
    poll_sec: float = 30.0,
    path: Path | None = None,
    label: str = "queue_launch",
) -> None:
    """Poll until :func:`can_launch_parallel` is true (RAM + GPU politics)."""
    t0 = time.monotonic()
    while True:
        ok, reason = can_launch_parallel(
            peak_ram_gib=peak_ram_gib,
            gpus=gpus,
            job_class=job_class,
            max_used_fraction=max_used_fraction,
            path=path,
        )
        if ok:
            print(f"[job_queue] {label}: launchable — {reason}", flush=True)
            return
        print(f"[job_queue] {label}: not launchable yet — {reason}", flush=True)
        if timeout_sec is not None and (time.monotonic() - t0) >= timeout_sec:
            raise TimeoutError(f"[job_queue] {label}: timed out — {reason}")
        time.sleep(poll_sec)


def append_queue_entry(
    name: str,
    *,
    job: str,
    pid: int,
    estimated_time: str,
    status: str = "RUNNING",
    resources: str = "",
    log: str = "",
    job_class: str = CLASS_LIGHT,
    peak_ram_gib: float | None = None,
    gpus: Iterable[int] = (),
    path: Path | None = None,
) -> None:
    """Append a queue.md entry (create file with header if missing)."""
    p = path or queue_path()
    if not p.is_file():
        p.write_text(
            "# Job queue\n\n"
            "Living log of **large** local CPU / RAM / GPU jobs.\n"
            "Policy: `.cursor/rules/local-job-queue.mdc`.\n\n"
            "<!-- Append new jobs below this line -->\n\n",
            encoding="utf-8",
        )
    gpu_s = ",".join(str(g) for g in gpus)
    lines = [
        f"### {name} — {status}",
        f"- **launch time:** {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        f"- **job:** `{job}`",
        f"- **PID:** {pid}",
        f"- **estimated time:** {estimated_time}",
        f"- **class:** {job_class}",
    ]
    if peak_ram_gib is not None:
        lines.append(f"- **peak_ram_gib:** {peak_ram_gib:.1f}")
    if gpu_s:
        lines.append(f"- **GPUs:** {gpu_s}")
    if resources:
        lines.append(f"- **resources:** {resources}")
    if log:
        lines.append(f"- **log:** {log}")
    lines.append(f"- **status:** {status}")
    with p.open("a", encoding="utf-8") as fh:
        fh.write("\n" + "\n".join(lines) + "\n")
