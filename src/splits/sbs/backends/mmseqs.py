"""MMseqs2 distance / cluster backend for SBS.

When the ``mmseqs`` binary is available, builds a temporary DB and runs
``easy-search`` (all-vs-all) to derive distances as ``1 - pident/100``.
Cluster-native output is retained in ``extras`` for future cluster-first
strategies. If MMseqs is missing, ``compute`` raises ``FileNotFoundError``
with install guidance (do not invent distances).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

from src.splits.sbs.distance import DistanceMatrix


def find_mmseqs(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"mmseqs binary not found: {path}")
        return path
    which = shutil.which("mmseqs")
    if which is None:
        raise FileNotFoundError(
            "mmseqs binary not found on PATH. Install MMseqs2 or pass "
            "mmseqs_bin=... to MMseqsDistanceBackend."
        )
    return Path(which)


class MMseqsDistanceBackend:
    """All-vs-all MMseqs easy-search → distance = 1 − percent identity / 100."""

    name = "mmseqs"

    def __init__(
        self,
        *,
        mmseqs_bin: str | Path | None = None,
        threads: int = 1,
        sensitivity: float = 7.5,
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
            with fasta.open("w", encoding="utf-8") as fh:
                for rid in ids:
                    fh.write(f">{rid}\n{sequences[rid]}\n")
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
            # Default distance = 1 (no hit); identity hit → 1 - pident/100
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
                        # Keep strongest (smallest distance) hit
                        if dist < matrix[i, j]:
                            matrix[i, j] = dist
            return DistanceMatrix(
                ids=ids,
                matrix=matrix,
                metric="one_minus_pident",
                backend=self.name,
                extras={"mmseqs_bin": str(mmseqs)},
            )
