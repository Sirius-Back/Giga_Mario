"""MMseqs2 backends for SBS.

Production path: ``easy-cluster`` → member→representative TSV → dense cluster
ids (no dense n×n matrix).

Legacy / small-n: ``MMseqsDistanceBackend`` via ``easy-search`` →
``1 - pident/100`` distance matrix (must not be used for full-panel clustering).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

from src.splits.sbs.distance import DistanceMatrix

DEFAULT_MIN_SEQ_ID = 0.8
DEFAULT_SENSITIVITY = 7.5


def find_mmseqs(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"mmseqs binary not found: {path}")
        return path
    which = shutil.which("mmseqs")
    if which is not None:
        return Path(which)
    # Common project env (bio_tools) when PATH is bare.
    for candidate in (
        Path.home() / "miniconda3" / "envs" / "bio_tools" / "bin" / "mmseqs",
        Path.home() / "mambaforge" / "envs" / "bio_tools" / "bin" / "mmseqs",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "mmseqs binary not found on PATH. Install MMseqs2 (e.g. "
        "`conda install -n bio_tools -c bioconda mmseqs2`) or pass "
        "mmseqs_bin=... to the MMseqs backend."
    )


def parse_cluster_tsv(path: Path) -> dict[str, str]:
    """Parse ``*_cluster.tsv`` (representative\\tmember) → member → representative."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MMseqs cluster TSV missing: {path}")
    member_to_rep: dict[str, str] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            rep, member = parts[0].strip(), parts[1].strip()
            if not rep or not member:
                continue
            member_to_rep[member] = rep
    if not member_to_rep:
        raise ValueError(f"no cluster assignments in {path}")
    return member_to_rep


def cluster_map_to_dense_ids(
    member_to_rep: Mapping[str, str],
    *,
    ids: list[str],
) -> dict[str, int]:
    """Map region ids → dense 0..C-1 cluster ints (singletons for missing)."""
    reps = sorted({member_to_rep[m] for m in member_to_rep})
    rep_to_cid = {rep: i for i, rep in enumerate(reps)}
    out: dict[str, int] = {}
    next_singleton = len(rep_to_cid)
    for rid in ids:
        if rid in member_to_rep:
            out[rid] = rep_to_cid[member_to_rep[rid]]
        else:
            out[rid] = next_singleton
            next_singleton += 1
    return out


def write_multifasta(sequences: Mapping[str, str], out_fa: Path) -> Path:
    """Write ``>{id}\\n{seq}`` multifasta (stable sorted id order)."""
    out_fa = Path(out_fa)
    out_fa.parent.mkdir(parents=True, exist_ok=True)
    with out_fa.open("w", encoding="utf-8") as fh:
        for rid in sorted(sequences):
            seq = sequences[rid]
            if not seq:
                raise ValueError(f"empty sequence for ID {rid!r}")
            fh.write(f">{rid}\n{seq}\n")
    return out_fa


def run_mmseqs_easy_cluster(
    fasta: Path,
    *,
    work: Path,
    mmseqs_bin: str | Path | None = None,
    threads: int = 8,
    sensitivity: float = DEFAULT_SENSITIVITY,
    min_seq_id: float = DEFAULT_MIN_SEQ_ID,
    force: bool = False,
    cov_mode: int = 0,
    cluster_mode: int = 0,
) -> Path:
    """Run ``mmseqs easy-cluster``; return path to ``*_cluster.tsv``.

    When ``force=false`` and the cluster TSV already exists and is non-empty,
    the CLI is skipped (resume).
    """
    mmseqs = find_mmseqs(mmseqs_bin)
    fasta = Path(fasta)
    if not fasta.is_file() or fasta.stat().st_size == 0:
        raise FileNotFoundError(f"input FASTA missing or empty: {fasta}")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    prefix = work / "clu"
    cluster_tsv = Path(str(prefix) + "_cluster.tsv")
    if cluster_tsv.is_file() and cluster_tsv.stat().st_size > 0 and not force:
        return cluster_tsv

    tmp = work / "tmp"
    if tmp.exists() and force:
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(mmseqs),
        "easy-cluster",
        str(fasta),
        str(prefix),
        str(tmp),
        "--threads",
        str(int(threads)),
        "-s",
        str(float(sensitivity)),
        "--min-seq-id",
        str(float(min_seq_id)),
        "--cov-mode",
        str(int(cov_mode)),
        "--cluster-mode",
        str(int(cluster_mode)),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "mmseqs easy-cluster failed "
            f"(exit {proc.returncode}): {proc.stderr[-2000:]}"
        )
    if not cluster_tsv.is_file() or cluster_tsv.stat().st_size == 0:
        raise FileNotFoundError(
            f"mmseqs easy-cluster did not write cluster TSV: {cluster_tsv}"
        )
    return cluster_tsv


class MMseqsDistanceBackend:
    """All-vs-all MMseqs easy-search → distance = 1 − percent identity / 100.

    Legacy / small-n only — do not use for full-panel clustering.
    """

    name = "mmseqs"

    def __init__(
        self,
        *,
        mmseqs_bin: str | Path | None = None,
        threads: int = 1,
        sensitivity: float = DEFAULT_SENSITIVITY,
        min_seq_id: float = 0.0,
    ) -> None:
        self.mmseqs_bin = mmseqs_bin
        self.threads = threads
        self.sensitivity = sensitivity
        self.min_seq_id = min_seq_id

    def compute(self, sequences: Mapping[str, str]) -> DistanceMatrix:
        mmseqs = find_mmseqs(self.mmseqs_bin)
        ids = tuple(sorted(sequences))
        n = len(ids)
        with tempfile.TemporaryDirectory(prefix="sbs_mmseqs_") as tmp:
            tmp_path = Path(tmp)
            fasta = tmp_path / "input.fa"
            write_multifasta(sequences, fasta)
            out_tsv = tmp_path / "allvsall.tsv"
            cmd = [
                str(mmseqs),
                "easy-search",
                str(fasta),
                str(fasta),
                str(out_tsv),
                str(tmp_path / "tmp"),
                "--threads",
                str(self.threads),
                "-s",
                str(self.sensitivity),
                "--min-seq-id",
                str(self.min_seq_id),
                "--format-output",
                "query,target,pident",
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "mmseqs easy-search failed "
                    f"(exit {proc.returncode}): {proc.stderr[-2000:]}"
                )
            matrix = np.ones((n, n), dtype=float)
            np.fill_diagonal(matrix, 0.0)
            index = {rid: i for i, rid in enumerate(ids)}
            if out_tsv.is_file():
                with out_tsv.open(encoding="utf-8") as fh:
                    for line in fh:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) < 3:
                            continue
                        q, t, pident_s = parts[0], parts[1], parts[2]
                        if q not in index or t not in index:
                            continue
                        try:
                            pident = float(pident_s)
                        except ValueError:
                            continue
                        dist = max(0.0, 1.0 - pident / 100.0)
                        i, j = index[q], index[t]
                        if dist < matrix[i, j]:
                            matrix[i, j] = dist
            return DistanceMatrix(
                ids=ids,
                matrix=matrix,
                metric="one_minus_pident",
                backend=self.name,
                extras={"mmseqs_bin": str(mmseqs)},
            )
