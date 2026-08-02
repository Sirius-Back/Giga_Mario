"""Retrain VGAE Stage1→Stage2 with homology_first loss under VGAE/*_lossfix.

Reuses packs from stage*_k5 (symlink); never clobbers prior artifacts.
Waits for a free GPU via train.resolve_device.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT1 = ROOT / "VGAE" / "stage1_region_k5_lossfix"
OUT2 = ROOT / "VGAE" / "stage2_hash_k5_lossfix"
SRC1 = ROOT / "VGAE" / "stage1_region_k5" / "pack"
SRC2 = ROOT / "VGAE" / "stage2_hash_k5" / "pack"
GRAPH = ROOT / "runs_unif" / "legnet" / "run37_legnet_pangenome_k5_wm100_100" / "graph"
MARKED = ROOT / "ready_legnet" / "MARKED"
CHECK_BIN = ROOT / "mag" / "src" / "split_check_othoparagroup" / "split_check_othoparagroup"
HASH_TABLE = ROOT / "mag" / "homology_graph" / "maps" / "gene_ortho_para_hash.tsv"


def _link_pack(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in src.iterdir():
        target = dst / name.name
        if target.exists() or target.is_symlink():
            continue
        try:
            target.symlink_to(name.resolve())
        except OSError:
            shutil.copy2(name, target)


def _run_checker(split_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not CHECK_BIN.is_file():
        print(f"[lossfix] checker binary missing: {CHECK_BIN}", flush=True)
        return
    cmd = [
        str(CHECK_BIN),
        "--split",
        str(split_csv),
        "--hash-table",
        str(HASH_TABLE),
        "--out",
        str(out_dir),
    ]
    print("[lossfix] checker:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=False)


def main() -> int:
    if not SRC1.is_dir():
        raise FileNotFoundError(f"missing Stage1 pack: {SRC1}")
    if not SRC2.is_dir():
        raise FileNotFoundError(f"missing Stage2 pack: {SRC2}")

    _link_pack(SRC1, OUT1 / "pack")
    _link_pack(SRC2, OUT2 / "pack")

    common = [
        sys.executable,
        "-u",
        "-m",
        "src.splits.vgae",
        "--loss-mode",
        "homology_first",
        "--k",
        "5",
        "--seed",
        "42",
        "--min-epochs",
        "25",
        "--patience",
        "10",
        "--max-epochs",
        "200",
        "--wait-poll-sec",
        "600",
        "--marked-dir",
        str(MARKED),
        "--graph-dir",
        str(GRAPH),
    ]

    print("[lossfix] Stage1 train…", flush=True)
    r1 = subprocess.run(
        common + ["--stage", "1", "--out", str(OUT1)],
        cwd=str(ROOT),
        check=False,
    )
    if r1.returncode != 0:
        return int(r1.returncode)

    _run_checker(OUT1 / "split.csv", ROOT / "VGAE" / "checks" / "stage1_lossfix")

    print("[lossfix] Stage2 train…", flush=True)
    ids = GRAPH / "ids.txt"
    r2 = subprocess.run(
        common
        + [
            "--stage",
            "2",
            "--out",
            str(OUT2),
            "--ids-file",
            str(ids),
        ],
        cwd=str(ROOT),
        check=False,
    )
    if r2.returncode != 0:
        return int(r2.returncode)

    _run_checker(OUT2 / "split.csv", ROOT / "VGAE" / "checks" / "stage2_lossfix")

    meta = {
        "stage1": str(OUT1 / "train_meta.json"),
        "stage2": str(OUT2 / "train_meta.json"),
    }
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
