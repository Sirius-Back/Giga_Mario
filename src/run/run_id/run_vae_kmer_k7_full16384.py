"""GPU MLP-VAE on full run13 k=7 embeds (16384-d, no projection)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "VAE" / "mlp_vae_kmer_k7_full16384_lossfix"
FEATURES = ROOT / "runs_unif" / "legnet" / "run13_legnet_kmer_k7" / "feature_table.npz"
CHECK_BIN = ROOT / "mag" / "src" / "split_check_othoparagroup" / "split_check_othoparagroup"
HASH_TABLE = ROOT / "mag" / "homology_graph" / "maps" / "gene_ortho_para_hash.tsv"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.splits.vae",
        "--out",
        str(OUT),
        "--features",
        str(FEATURES),
        "--k",
        "7",
        "--seed",
        "42",
        "--device",
        "cuda:0",
        "--prefer-gpu",
        "--keep-memmap",
        "--batch-size",
        "2048",
        "--peak-ram-gib",
        "40",
        "--wait-poll-sec",
        "120",
        "--min-epochs",
        "25",
        "--patience",
        "10",
        "--max-epochs",
        "200",
        "--source-label",
        "run13_legnet_kmer_k7",
    ]
    print("[run] ", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=str(ROOT), check=False).returncode
    if rc != 0:
        return int(rc)
    check_out = ROOT / "VAE" / "checks" / "k7_full16384_lossfix"
    check_out.mkdir(parents=True, exist_ok=True)
    if CHECK_BIN.is_file():
        subprocess.run(
            [
                str(CHECK_BIN),
                "--split",
                str(OUT / "split.csv"),
                "--hash-table",
                str(HASH_TABLE),
                "--outdir",
                str(check_out),
                "--model",
                "vae",
                "--run-id",
                "mlp_vae_kmer_k7_full16384_lossfix",
            ],
            check=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
