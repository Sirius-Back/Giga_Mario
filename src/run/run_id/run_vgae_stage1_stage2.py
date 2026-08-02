"""Orchestrate VGAE Stage1 (region) then Stage2 (hash) under VGAE/."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ids", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--skip-stage2", action="store_true")
    args = ap.parse_args(argv)

    py = sys.executable
    common = [
        py,
        "-m",
        "src.splits.vgae",
        "--graph-dir",
        "runs_unif/legnet/run37_legnet_pangenome_k5_wm100_100/graph",
        "--marked-dir",
        "ready_legnet/MARKED",
        "--k",
        "5",
        "--seed",
        "42",
        "--min-epochs",
        "25",
        "--patience",
        "10",
        "--max-epochs",
        str(args.max_epochs),
        "--wait-poll-sec",
        "600",
    ]
    if args.max_ids is not None:
        common.extend(["--max-ids", str(args.max_ids)])
    if args.device:
        common.extend(["--device", args.device])

    _run(common + ["--stage", "1", "--out", "VGAE/stage1_region_k5"])
    if not args.skip_stage2:
        _run(common + ["--stage", "2", "--out", "VGAE/stage2_hash_k5"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
