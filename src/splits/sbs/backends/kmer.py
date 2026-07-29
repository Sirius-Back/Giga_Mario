"""K-mer composition feature backend for SBS (DSK-backed).

Caption: ``splits/kmer.md``. Observed k-mers only (no dense 4^k allocation).
DSK (GATB) is used for ``k >= 3``. DSK rejects ``k <= 2``; those sizes use an
in-process overlapping counter with the same FeatureTable contract.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.splits.sbs.features import FeatureTable

__all__ = (
    "DSK_MIN_K",
    "KmerFeatureBackend",
    "count_kmers_dsk",
    "count_kmers_local",
    "find_dsk",
    "find_dsk2ascii",
    "normalize_k_list",
    "parse_dsk_ascii",
)

DSK_MIN_K = 3


def find_dsk(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"dsk binary not found: {path}")
        return path
    which = shutil.which("dsk")
    if which is None:
        raise FileNotFoundError(
            "dsk binary not found on PATH. Install GATB/DSK or pass dsk_bin=..."
        )
    return Path(which)


def find_dsk2ascii(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"dsk2ascii binary not found: {path}")
        return path
    which = shutil.which("dsk2ascii")
    if which is None:
        raise FileNotFoundError(
            "dsk2ascii binary not found on PATH. Install GATB/DSK or pass "
            "dsk2ascii_bin=..."
        )
    return Path(which)


def normalize_k_list(k: int | Sequence[int]) -> tuple[int, ...]:
    """Validate and return a sorted unique tuple of positive k-mer sizes."""
    if isinstance(k, (int, np.integer)):
        values = [int(k)]
    else:
        values = [int(x) for x in k]
    if not values:
        raise ValueError("k must contain at least one k-mer size")
    bad = [x for x in values if x < 1]
    if bad:
        raise ValueError(f"k-mer sizes must be >= 1; got {bad}")
    return tuple(sorted(set(values)))


def parse_dsk_ascii(text: str) -> dict[str, int]:
    """Parse ``dsk2ascii`` lines ``KMER COUNT``; sum duplicate keys."""
    counts: dict[str, int] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"malformed dsk2ascii line: {line!r}")
        kmer = parts[0].upper()
        try:
            abundance = int(parts[1])
        except ValueError as exc:
            raise ValueError(f"non-integer abundance in dsk2ascii line: {line!r}") from exc
        if abundance < 0:
            raise ValueError(f"negative abundance in dsk2ascii line: {line!r}")
        counts[kmer] = counts.get(kmer, 0) + abundance
    return counts


def count_kmers_local(sequence: str, k: int) -> dict[str, int]:
    """Overlapping ACGT-only k-mer counts (observed keys only)."""
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    seq = "".join(ch for ch in sequence.upper() if not ch.isspace())
    if len(seq) < k:
        return {}
    counts: Counter[str] = Counter()
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        if any(base not in "ACGT" for base in kmer):
            continue
        counts[kmer] += 1
    return dict(counts)


def count_kmers_dsk(
    sequence: str,
    k: int,
    *,
    workdir: Path,
    dsk_bin: Path,
    dsk2ascii_bin: Path,
    abundance_min: int = 1,
    threads: int = 1,
) -> dict[str, int]:
    """Run DSK + dsk2ascii on one sequence; return observed k-mer counts."""
    if k < DSK_MIN_K:
        raise ValueError(
            f"DSK does not support k <= {DSK_MIN_K - 1}; got k={k}. "
            "Use the in-process counter for those sizes."
        )
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    fasta = workdir / "seq.fa"
    out_prefix = workdir / "counts"
    tmp_dir = workdir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    seq = "".join(ch for ch in sequence.upper() if not ch.isspace())
    if not seq:
        return {}
    fasta.write_text(f">seq\n{seq}\n", encoding="utf-8")
    cmd_dsk = [
        str(dsk_bin),
        "-file",
        str(fasta),
        "-kmer-size",
        str(int(k)),
        "-abundance-min",
        str(int(abundance_min)),
        "-out",
        str(out_prefix),
        "-out-tmp",
        str(tmp_dir),
        "-nb-cores",
        str(max(1, int(threads))),
        "-verbose",
        "0",
    ]
    proc = subprocess.run(cmd_dsk, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"dsk failed (exit {proc.returncode}) for k={k}: "
            f"{(proc.stderr or proc.stdout or '').strip()[:500]}"
        )
    h5 = Path(str(out_prefix) + ".h5")
    if not h5.is_file():
        # Some builds write without suffix when -out already ends oddly.
        candidates = list(workdir.glob("counts*.h5"))
        if not candidates:
            raise FileNotFoundError(f"dsk did not write HDF5 counts under {workdir}")
        h5 = candidates[0]
    ascii_path = workdir / "counts.txt"
    cmd_ascii = [
        str(dsk2ascii_bin),
        "-file",
        str(h5),
        "-out",
        str(ascii_path),
        "-nb-cores",
        "1",
        "-verbose",
        "0",
    ]
    proc2 = subprocess.run(cmd_ascii, capture_output=True, text=True, check=False)
    if proc2.returncode != 0 or not ascii_path.is_file():
        raise RuntimeError(
            f"dsk2ascii failed (exit {proc2.returncode}): "
            f"{(proc2.stderr or proc2.stdout or '').strip()[:500]}"
        )
    return parse_dsk_ascii(ascii_path.read_text(encoding="utf-8", errors="replace"))


def _feature_name(k: int, kmer: str, *, multi_k: bool) -> str:
    if multi_k:
        return f"k{k}_{kmer}"
    return f"kmer_{kmer}"


def _relative(counts: Mapping[str, float] | Mapping[str, int]) -> dict[str, float]:
    total = float(sum(float(v) for v in counts.values()))
    if total <= 0.0:
        return {key: 0.0 for key in counts}
    return {key: float(val) / total for key, val in counts.items()}


class KmerFeatureBackend:
    """Per-region k-mer composition features via DSK (and local for k < 3)."""

    name = "kmer"

    def __init__(
        self,
        k: int | Sequence[int] = 5,
        *,
        normalize: str = "relative",
        log_transform: bool = False,
        dsk_bin: str | Path | None = None,
        dsk2ascii_bin: str | Path | None = None,
        abundance_min: int = 1,
        threads: int = 1,
        allow_local_for_small_k: bool = True,
    ) -> None:
        self.k_list = normalize_k_list(k)
        if normalize not in {"relative", "none"}:
            raise ValueError(
                f"normalize must be 'relative' or 'none'; got {normalize!r}"
            )
        self.normalize = normalize
        self.log_transform = bool(log_transform)
        self.dsk_bin = dsk_bin
        self.dsk2ascii_bin = dsk2ascii_bin
        self.abundance_min = int(abundance_min)
        if self.abundance_min < 1:
            raise ValueError("abundance_min must be >= 1")
        self.threads = max(1, int(threads))
        self.allow_local_for_small_k = bool(allow_local_for_small_k)

    def _count_one(
        self,
        sequence: str,
        k: int,
        *,
        work_root: Path,
        dsk: Path | None,
        dsk2ascii: Path | None,
        region_tag: str,
    ) -> dict[str, int]:
        if k < DSK_MIN_K:
            if not self.allow_local_for_small_k:
                raise ValueError(
                    f"k={k} requires DSK but DSK rejects k < {DSK_MIN_K}; "
                    "set allow_local_for_small_k=True or choose k >= 3"
                )
            return count_kmers_local(sequence, k)
        assert dsk is not None and dsk2ascii is not None
        region_dir = work_root / f"k{k}_{region_tag}"
        region_dir.mkdir(parents=True, exist_ok=True)
        return count_kmers_dsk(
            sequence,
            k,
            workdir=region_dir,
            dsk_bin=dsk,
            dsk2ascii_bin=dsk2ascii,
            abundance_min=self.abundance_min,
            threads=self.threads,
        )

    def compute(self, sequences: Mapping[str, str]) -> FeatureTable:
        ids = tuple(sorted(sequences))
        if len(ids) < 1:
            raise ValueError("need >=1 sequences for k-mer features")
        multi_k = len(self.k_list) > 1
        needs_dsk = any(k >= DSK_MIN_K for k in self.k_list)
        needs_local = any(k < DSK_MIN_K for k in self.k_list)
        dsk: Path | None = None
        dsk2ascii: Path | None = None
        if needs_dsk:
            dsk = find_dsk(self.dsk_bin)
            dsk2ascii = find_dsk2ascii(self.dsk2ascii_bin)
        if needs_local:
            warnings.warn(
                f"DSK does not support k < {DSK_MIN_K}; using in-process "
                f"overlapping counts for k in "
                f"{[k for k in self.k_list if k < DSK_MIN_K]}",
                UserWarning,
                stacklevel=2,
            )

        per_id_k_counts: dict[str, dict[int, dict[str, int]]] = {}
        vocab: dict[int, set[str]] = {k: set() for k in self.k_list}

        with tempfile.TemporaryDirectory(prefix="sbs_kmer_dsk_") as tmp:
            work_root = Path(tmp)
            for idx, rid in enumerate(ids):
                tag = f"{idx:06d}"
                per_id_k_counts[rid] = {}
                for k in self.k_list:
                    counts = self._count_one(
                        sequences[rid],
                        k,
                        work_root=work_root,
                        dsk=dsk,
                        dsk2ascii=dsk2ascii,
                        region_tag=tag,
                    )
                    per_id_k_counts[rid][k] = counts
                    vocab[k].update(counts)

        feature_names: list[str] = []
        kmer_index: list[tuple[int, str]] = []
        for k in self.k_list:
            for kmer in sorted(vocab[k]):
                feature_names.append(_feature_name(k, kmer, multi_k=multi_k))
                kmer_index.append((k, kmer))

        if not feature_names:
            # All sequences shorter than min k — still return a valid empty-feature
            # table shape is invalid for clustering; raise early.
            raise ValueError(
                f"no observed k-mers for k={list(self.k_list)}; "
                "sequences may be shorter than k"
            )

        mat = np.zeros((len(ids), len(feature_names)), dtype=float)
        for i, rid in enumerate(ids):
            col = 0
            for k in self.k_list:
                raw = per_id_k_counts[rid][k]
                values = (
                    _relative(raw)
                    if self.normalize == "relative"
                    else {kk: float(vv) for kk, vv in raw.items()}
                )
                if self.log_transform:
                    values = {kk: float(np.log1p(vv)) for kk, vv in values.items()}
                for kmer in sorted(vocab[k]):
                    mat[i, col] = float(values.get(kmer, 0.0))
                    col += 1

        return FeatureTable(
            ids=ids,
            feature_names=tuple(feature_names),
            matrix=mat,
            backend=self.name,
            extras={
                "k": list(self.k_list),
                "normalize": self.normalize,
                "log_transform": self.log_transform,
                "dsk_min_k": DSK_MIN_K,
                "n_features": len(feature_names),
            },
        )
