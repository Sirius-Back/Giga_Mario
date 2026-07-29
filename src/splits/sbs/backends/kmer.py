"""K-mer composition feature backend for SBS.

Production default (``engine="auto"``): fast in-process counter for ``k >= 2``
(native C++ when built, else optimized Python). GATB DSK is optional
(``engine="dsk"``) and only supports ``k >= 3``.

Caption: ``splits/kmer.md``. Observed k-mers only (no dense FeatureTable of
absent k-mers — dense buffers are an internal counting device).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np

from src.splits.sbs.features import FeatureTable

__all__ = (
    "DSK_MIN_K",
    "NATIVE_MAX_K",
    "KmerFeatureBackend",
    "count_kmers_dsk",
    "count_kmers_local",
    "count_kmers",
    "find_dsk",
    "find_dsk2ascii",
    "normalize_k_list",
    "parse_dsk_ascii",
)

DSK_MIN_K = 3
NATIVE_MAX_K = 12  # matches C++ KMER_COUNT_MAX_K (dense 4^k)

KmerEngine = Literal["auto", "native", "python", "dsk"]


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
    """Overlapping ACGT-only k-mer counts (observed keys only). Pure Python."""
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
            "Use engine='auto'/'native'/'python' for those sizes."
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


def count_kmers(
    sequence: str,
    k: int,
    *,
    engine: KmerEngine = "auto",
) -> dict[str, int]:
    """Count overlapping ACGT k-mers (abundance, not presence).

    ``auto`` / ``native`` / ``python`` work for any ``k >= 1`` (native dense
    path covers ``1..12``). ``dsk`` requires ``k >= 3`` and a writable temp dir
    via ``count_kmers_dsk`` — use ``KmerFeatureBackend(engine='dsk')`` for that.
    """
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if engine == "dsk":
        raise ValueError(
            "count_kmers(engine='dsk') needs per-call workdirs; "
            "use KmerFeatureBackend(engine='dsk') instead"
        )
    if engine in {"auto", "native"} and k <= NATIVE_MAX_K:
        from src.splits.sbs.backends.native import try_get_native_counter

        native = try_get_native_counter()
        if native is not None:
            return native.count(sequence, k)
        if engine == "native":
            raise RuntimeError(
                "native k-mer counter unavailable; build with "
                "`python -m src.splits.sbs.backends.native.build`"
            )
    return count_kmers_local(sequence, k)


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
    """Per-region k-mer composition features (abundance counts → FeatureTable)."""

    name = "kmer"

    def __init__(
        self,
        k: int | Sequence[int] = 5,
        *,
        normalize: str = "relative",
        log_transform: bool = False,
        engine: KmerEngine = "auto",
        dsk_bin: str | Path | None = None,
        dsk2ascii_bin: str | Path | None = None,
        abundance_min: int = 1,
        threads: int = 1,
        allow_local_for_small_k: bool = True,  # legacy alias; ignored if engine set
    ) -> None:
        self.k_list = normalize_k_list(k)
        if normalize not in {"relative", "none"}:
            raise ValueError(
                f"normalize must be 'relative' or 'none'; got {normalize!r}"
            )
        if engine not in {"auto", "native", "python", "dsk"}:
            raise ValueError(
                f"engine must be auto|native|python|dsk; got {engine!r}"
            )
        self.normalize = normalize
        self.log_transform = bool(log_transform)
        self.engine: KmerEngine = engine
        self.dsk_bin = dsk_bin
        self.dsk2ascii_bin = dsk2ascii_bin
        self.abundance_min = int(abundance_min)
        if self.abundance_min < 1:
            raise ValueError("abundance_min must be >= 1")
        self.threads = max(1, int(threads))
        self.allow_local_for_small_k = bool(allow_local_for_small_k)
        _ = allow_local_for_small_k  # retained for call-site compatibility

    def _resolve_engine(self) -> KmerEngine:
        if self.engine != "auto":
            return self.engine
        # Prefer native in-process for all supported k (includes k=2).
        if all(k <= NATIVE_MAX_K for k in self.k_list):
            from src.splits.sbs.backends.native import try_get_native_counter

            if try_get_native_counter() is not None:
                return "native"
            return "python"
        # Mixed / very large k: python observed Counter (never invent DSK default).
        return "python"

    def _count_one(
        self,
        sequence: str,
        k: int,
        *,
        engine: KmerEngine,
        work_root: Path | None,
        dsk: Path | None,
        dsk2ascii: Path | None,
        region_tag: str,
    ) -> dict[str, int]:
        if engine == "dsk":
            if k < DSK_MIN_K:
                if not self.allow_local_for_small_k:
                    raise ValueError(
                        f"k={k} cannot use DSK (min k={DSK_MIN_K}); "
                        "use engine='auto' or 'native'"
                    )
                return count_kmers(sequence, k, engine="auto")
            assert work_root is not None and dsk is not None and dsk2ascii is not None
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
        if engine == "native":
            return count_kmers(sequence, k, engine="native")
        if engine == "python":
            return count_kmers_local(sequence, k)
        # auto already resolved
        return count_kmers(sequence, k, engine="auto")

    def compute(self, sequences: Mapping[str, str]) -> FeatureTable:
        ids = tuple(sorted(sequences))
        if len(ids) < 1:
            raise ValueError("need >=1 sequences for k-mer features")
        multi_k = len(self.k_list) > 1
        engine = self._resolve_engine()

        dsk: Path | None = None
        dsk2ascii: Path | None = None
        if engine == "dsk":
            dsk = find_dsk(self.dsk_bin)
            dsk2ascii = find_dsk2ascii(self.dsk2ascii_bin)
            small = [k for k in self.k_list if k < DSK_MIN_K]
            if small:
                warnings.warn(
                    f"DSK does not support k < {DSK_MIN_K}; using in-process "
                    f"counts for k in {small} within engine='dsk'",
                    UserWarning,
                    stacklevel=2,
                )

        per_id_k_counts: dict[str, dict[int, dict[str, int]]] = {}
        vocab: dict[int, set[str]] = {k: set() for k in self.k_list}

        tmp_ctx = (
            tempfile.TemporaryDirectory(prefix="sbs_kmer_dsk_")
            if engine == "dsk"
            else None
        )
        try:
            work_root = Path(tmp_ctx.name) if tmp_ctx is not None else None
            for idx, rid in enumerate(ids):
                tag = f"{idx:06d}"
                per_id_k_counts[rid] = {}
                for k in self.k_list:
                    counts = self._count_one(
                        sequences[rid],
                        k,
                        engine=engine,
                        work_root=work_root,
                        dsk=dsk,
                        dsk2ascii=dsk2ascii,
                        region_tag=tag,
                    )
                    per_id_k_counts[rid][k] = counts
                    vocab[k].update(counts)
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()

        feature_names: list[str] = []
        for k in self.k_list:
            for kmer in sorted(vocab[k]):
                feature_names.append(_feature_name(k, kmer, multi_k=multi_k))

        if not feature_names:
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
                "engine": engine,
                "dsk_min_k": DSK_MIN_K,
                "native_max_k": NATIVE_MAX_K,
                "n_features": len(feature_names),
            },
        )
