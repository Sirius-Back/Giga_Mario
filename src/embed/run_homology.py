"""CLI: LegNet embed homology dissimilarity (paralog vs ortholog).

Example::

  python -m src.embed.run_homology \\
    --embed-root results/embed_legnet \\
    --out results/embed_legnet/homology_dissim
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.embed.homology_dissim import (
    DEFAULT_LAYERS_HOM,
    run_all_stores,
)
from src.pipeline.job_queue import (
    CLASS_CPU_RAM_HEAVY,
    append_queue_entry,
    queue_path,
)
from src.pipeline.mem_guard import wait_for_ram_headroom
from src.splits.vgae.homology_loss import DEFAULT_HASH_TABLE

ROOT = Path(__file__).resolve().parents[2]


def _append_queue(**kwargs: Any) -> None:
    try:
        append_queue_entry(**kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: queue.md append failed: {exc}", flush=True)


def _update_queue(name: str, status: str, *, note: str = "") -> None:
    try:
        p = queue_path()
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        lines = [
            f"### {name} — {status}",
            f"- **update time:** {ts}",
            f"- **status:** {status}",
        ]
        if note:
            lines.append(f"- **note:** {note}")
        with p.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: queue.md update failed: {exc}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--embed-root",
        type=Path,
        default=ROOT / "results" / "embed_legnet",
        help="Directory of LegNet embed stores",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dir (default: <embed-root>/homology_dissim)",
    )
    p.add_argument(
        "--hash-table",
        type=Path,
        default=DEFAULT_HASH_TABLE,
        help="Compara gene_ortho_para_hash.tsv",
    )
    p.add_argument(
        "--layers",
        default=",".join(DEFAULT_LAYERS_HOM),
        help="Comma-separated layers",
    )
    p.add_argument(
        "--primary-layer",
        default="pooled",
        help="Layer used for ranking.tsv",
    )
    p.add_argument("--max-groups", type=int, default=8192)
    p.add_argument("--max-members", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--peak-ram-gib",
        type=float,
        default=12.0,
        help="Declared peak for queue / headroom wait",
    )
    args = p.parse_args(argv)

    embed_root = Path(args.embed_root)
    out = Path(args.out) if args.out is not None else embed_root / "homology_dissim"
    layers = tuple(s.strip() for s in str(args.layers).split(",") if s.strip())
    if not layers:
        raise SystemExit("no layers specified")
    if args.primary_layer not in layers:
        raise SystemExit(
            f"--primary-layer {args.primary_layer!r} not in layers {layers}"
        )

    wait_for_ram_headroom(label="homology_dissim")

    job = f"embed_homology_dissim_{int(time.time())}"
    _append_queue(
        name=job,
        status="RUNNING",
        job=f"python -m src.embed.run_homology out={out}",
        pid=os.getpid(),
        estimated_time="45m",
        job_class=CLASS_CPU_RAM_HEAVY,
        peak_ram_gib=float(args.peak_ram_gib),
        gpus=(),
        log=str(out / "run.log"),
    )
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run.log"
    try:
        written = run_all_stores(
            embed_root,
            out,
            layers=layers,
            hash_table=Path(args.hash_table),
            max_groups=int(args.max_groups) if args.max_groups > 0 else None,
            max_members=int(args.max_members) if args.max_members > 0 else None,
            seed=int(args.seed),
            primary_layer=str(args.primary_layer),
        )
        msg = ", ".join(f"{k}={v}" for k, v in written.items())
        log_path.write_text(msg + "\n", encoding="utf-8")
        print(f"[homology_dissim] done: {msg}", flush=True)
        _update_queue(job, "COMPLETED", note=msg)
    except Exception as exc:
        _update_queue(job, "FAILED", note=str(exc))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
