"""Host RAM headroom helpers (local-job-queue: used ≤ 95% of MemTotal).

Hard constraint is **total host RAM occupancy**, not CPU and not a RAM/CPU
ratio. Long jobs should **wait** for headroom instead of exiting immediately.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any


DEFAULT_MAX_USED_FRACTION = 0.95
DEFAULT_POLL_SEC = 15.0


def read_meminfo() -> dict[str, int]:
    """Return selected /proc/meminfo fields in **bytes**."""
    out: dict[str, int] = {}
    text = Path("/proc/meminfo").read_text(encoding="utf-8")
    for line in text.splitlines():
        if not line.startswith(("MemTotal:", "MemAvailable:", "MemFree:")):
            continue
        key, rest = line.split(":", 1)
        out[key] = int(rest.split()[0]) * 1024
    if "MemTotal" not in out or "MemAvailable" not in out:
        raise RuntimeError("failed to parse MemTotal/MemAvailable from /proc/meminfo")
    return out


def ram_snapshot() -> dict[str, Any]:
    """Structured MemTotal / MemAvailable / used fraction for logging."""
    info = read_meminfo()
    total = float(info["MemTotal"])
    avail = float(info["MemAvailable"])
    used_frac = max(0.0, min(1.0, (total - avail) / total))
    return {
        "mem_total_bytes": int(info["MemTotal"]),
        "mem_available_bytes": int(info["MemAvailable"]),
        "mem_used_fraction": used_frac,
        "mem_total_gib": total / 2**30,
        "mem_available_gib": avail / 2**30,
        "mem_used_pct": 100.0 * used_frac,
    }


def ram_used_fraction() -> float:
    """Fraction of MemTotal currently unavailable (0–1)."""
    return float(ram_snapshot()["mem_used_fraction"])


def _format_snap(snap: dict[str, Any] | None = None) -> str:
    s = snap or ram_snapshot()
    return (
        f"used={s['mem_used_pct']:.1f}% "
        f"avail={s['mem_available_gib']:.1f} GiB / "
        f"total={s['mem_total_gib']:.1f} GiB"
    )


def assert_ram_headroom(max_used_fraction: float = DEFAULT_MAX_USED_FRACTION) -> None:
    """Raise MemoryError if host RAM used fraction exceeds ``max_used_fraction``.

    Prefer :func:`wait_for_ram_headroom` inside long jobs so transient pressure
    does not kill the process instantly.
    """
    if not (0.0 < max_used_fraction < 1.0):
        raise ValueError("max_used_fraction must be in (0, 1)")
    snap = ram_snapshot()
    if snap["mem_used_fraction"] > max_used_fraction:
        raise MemoryError(
            f"host RAM {_format_snap(snap)} "
            f"(limit used<={100 * max_used_fraction:.0f}%) — "
            "refuse (local-job-queue headroom)"
        )


def kill_cursor_indexers(
    *,
    min_used_fraction: float = 0.94,
    kill_file_watchers_if_still_high: bool = True,
    max_file_watchers: int = 4,
) -> list[int]:
    """Kill Cursor ``rg`` indexers (by PID) when host RAM used ≥ threshold.

    Never uses ``pkill -f``. Optionally trims heaviest ``fileWatcher`` forks if
    used is still ≥ 94.5% after rg kills. Does not touch training/python jobs.
    """
    import os
    import signal

    killed: list[int] = []
    snap = ram_snapshot()
    if snap["mem_used_fraction"] < float(min_used_fraction):
        return killed

    # 1) Cursor / VS Code ripgrep file indexers
    try:
        for pid_s, cmd in _iter_proc_cmdlines():
            if "/@vscode/ripgrep/bin/rg" not in cmd:
                continue
            pid = int(pid_s)
            if pid == os.getpid():
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                killed.append(pid)
            except ProcessLookupError:
                continue
            except PermissionError:
                continue
    except Exception as exc:  # noqa: BLE001 — best-effort reclaim
        print(f"[mem_guard] kill_cursor_indexers rg scan failed: {exc}", flush=True)

    if killed:
        print(
            f"[mem_guard] killed cursor rg PIDs={killed} "
            f"({_format_snap(snap)})",
            flush=True,
        )
        time.sleep(1.5)

    if not kill_file_watchers_if_still_high:
        return killed

    snap2 = ram_snapshot()
    if snap2["mem_used_fraction"] < 0.945:
        return killed

    # 2) Heaviest fileWatcher node forks (RSS descending)
    watchers: list[tuple[int, int]] = []  # (rss_kb, pid)
    try:
        for pid_s, cmd in _iter_proc_cmdlines():
            if "bootstrap-fork --type=fileWatcher" not in cmd:
                continue
            if "cursor-server" not in cmd and "vscode-server" not in cmd:
                continue
            pid = int(pid_s)
            rss = _rss_kb(pid)
            if rss is not None:
                watchers.append((rss, pid))
    except Exception as exc:  # noqa: BLE001
        print(f"[mem_guard] fileWatcher scan failed: {exc}", flush=True)
        return killed

    watchers.sort(reverse=True)
    fw_killed: list[int] = []
    for _rss, pid in watchers[: max(0, int(max_file_watchers))]:
        try:
            os.kill(pid, signal.SIGTERM)
            fw_killed.append(pid)
            killed.append(pid)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue
    if fw_killed:
        print(
            f"[mem_guard] killed fileWatcher PIDs={fw_killed} "
            f"({_format_snap(snap2)})",
            flush=True,
        )
    return killed


def _iter_proc_cmdlines() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    proc = Path("/proc")
    for ent in proc.iterdir():
        if not ent.name.isdigit():
            continue
        try:
            raw = (ent / "cmdline").read_bytes().replace(b"\x00", b" ")
            cmd = raw.decode("utf-8", errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        rows.append((ent.name, cmd))
    return rows


def _rss_kb(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return None
    return None


def wait_for_ram_headroom(
    max_used_fraction: float = DEFAULT_MAX_USED_FRACTION,
    *,
    timeout_sec: float | None = 6 * 3600,
    poll_sec: float = DEFAULT_POLL_SEC,
    label: str = "ram_headroom",
    kill_indexers: bool = True,
) -> None:
    """Block until used RAM ≤ ``max_used_fraction``, then return.

    Raises MemoryError only after ``timeout_sec`` (None = wait forever).
    When ``kill_indexers`` is True, attempts to free Cursor ``rg`` / heavy
    fileWatchers on each wait iteration (by PID, never ``pkill -f``).
    """
    if not (0.0 < max_used_fraction < 1.0):
        raise ValueError("max_used_fraction must be in (0, 1)")
    if poll_sec <= 0:
        raise ValueError("poll_sec must be > 0")
    t0 = time.monotonic()
    warned = False
    while True:
        if kill_indexers:
            kill_cursor_indexers(min_used_fraction=min(0.94, max_used_fraction))
        snap = ram_snapshot()
        if snap["mem_used_fraction"] <= max_used_fraction:
            if warned:
                print(f"[mem_guard] {label}: headroom OK ({_format_snap(snap)})", flush=True)
            return
        if not warned:
            print(
                f"[mem_guard] {label}: waiting — {_format_snap(snap)} "
                f"(limit used<={100 * max_used_fraction:.0f}%)",
                flush=True,
            )
            warned = True
        if timeout_sec is not None and (time.monotonic() - t0) >= timeout_sec:
            raise MemoryError(
                f"[mem_guard] {label}: timed out after {timeout_sec:.0f}s — "
                f"{_format_snap(snap)} (limit used<={100 * max_used_fraction:.0f}%)"
            )
        time.sleep(poll_sec)


def ensure_allocation_fits(
    nbytes: int,
    *,
    max_used_fraction: float = DEFAULT_MAX_USED_FRACTION,
    safety: float = 1.20,
    timeout_sec: float | None = 6 * 3600,
    poll_sec: float = DEFAULT_POLL_SEC,
    label: str = "alloc",
    kill_indexers: bool = True,
) -> None:
    """Wait until ``nbytes * safety`` fits under the 95% MemTotal cap.

    Uses MemAvailable vs required bytes, and also checks that
    (MemTotal - MemAvailable + need) / MemTotal ≤ max_used_fraction.
    """
    if nbytes < 0:
        raise ValueError("nbytes must be >= 0")
    if safety < 1.0:
        raise ValueError("safety must be >= 1.0")
    need = int(nbytes * safety)
    t0 = time.monotonic()
    warned = False
    while True:
        if kill_indexers:
            kill_cursor_indexers(min_used_fraction=min(0.94, max_used_fraction))
        snap = ram_snapshot()
        total = int(snap["mem_total_bytes"])
        avail = int(snap["mem_available_bytes"])
        used_after = (total - avail) + need
        frac_after = used_after / total if total else 1.0
        if avail >= need and frac_after <= max_used_fraction:
            if warned:
                print(
                    f"[mem_guard] {label}: OK need={need / 2**30:.1f} GiB "
                    f"({_format_snap(snap)})",
                    flush=True,
                )
            return
        if not warned:
            print(
                f"[mem_guard] {label}: waiting for {need / 2**30:.1f} GiB "
                f"(safety={safety}); {_format_snap(snap)}; "
                f"projected used={100 * frac_after:.1f}%",
                flush=True,
            )
            warned = True
        if timeout_sec is not None and (time.monotonic() - t0) >= timeout_sec:
            raise MemoryError(
                f"[mem_guard] {label}: timed out waiting for {need / 2**30:.1f} GiB — "
                f"{_format_snap(snap)}; projected used={100 * frac_after:.1f}%"
            )
        time.sleep(poll_sec)
