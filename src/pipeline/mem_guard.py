"""Host RAM headroom helpers (local-job-queue: used ≤ 95% of MemTotal)."""
from __future__ import annotations

from pathlib import Path


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


def ram_used_fraction() -> float:
    """Fraction of MemTotal currently unavailable (0–1)."""
    info = read_meminfo()
    total = float(info["MemTotal"])
    avail = float(info["MemAvailable"])
    return max(0.0, min(1.0, (total - avail) / total))


def assert_ram_headroom(max_used_fraction: float = 0.95) -> None:
    """Raise MemoryError if host RAM used fraction exceeds ``max_used_fraction``."""
    if not (0.0 < max_used_fraction < 1.0):
        raise ValueError("max_used_fraction must be in (0, 1)")
    used = ram_used_fraction()
    if used > max_used_fraction:
        info = read_meminfo()
        raise MemoryError(
            f"host RAM used {100 * used:.1f}% of MemTotal "
            f"(limit {100 * max_used_fraction:.0f}%); "
            f"MemAvailable={info['MemAvailable'] / 2**30:.1f} GiB / "
            f"MemTotal={info['MemTotal'] / 2**30:.1f} GiB — "
            "refuse to continue (local-job-queue headroom)"
        )
