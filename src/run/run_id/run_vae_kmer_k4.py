"""Run MLP-VAE k=4 homology_first baseline under VAE/mlp_vae_kmer_k4_lossfix."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "VAE" / "mlp_vae_kmer_k4_lossfix"
FEATURES = ROOT / "runs_unif" / "legnet" / "run11_legnet_kmer_k4" / "feature_table.csv"
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
        "4",
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
        "--peak-ram-gib",
        "8",
        "--source-label",
        "run11_legnet_kmer_k4",
    ]
    print("[run] ", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=str(ROOT), check=False).returncode
    if rc != 0:
        return int(rc)
    check_out = ROOT / "VAE" / "checks" / "k4_lossfix"
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
                "mlp_vae_kmer_k4_lossfix",
            ],
            check=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
