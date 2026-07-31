"""MAFFT alignments + per-position consensus rates for orthoparagroups."""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

DNA = frozenset("ACGT")


@dataclass(frozen=True)
class SeqRecord:
    header: str
    sequence: str
    role: str  # ortholog | paralog


def parse_fasta(path: Path) -> list[SeqRecord]:
    """Parse FASTA; role from header prefix ``ortholog|`` / ``paralog|``."""
    text = path.read_text(encoding="utf-8", errors="replace")
    records: list[SeqRecord] = []
    header: str | None = None
    chunks: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append(_make_record(header, "".join(chunks)))
            header = line[1:].strip()
            chunks = []
        else:
            chunks.append(line.strip())
    if header is not None:
        records.append(_make_record(header, "".join(chunks)))
    if not records:
        raise ValueError(f"Empty FASTA: {path}")
    return records


def _make_record(header: str, seq: str) -> SeqRecord:
    seq_u = seq.upper().replace("U", "T")
    if not seq_u:
        raise ValueError(f"Empty sequence for header {header!r}")
    role = "ortholog" if header.startswith("ortholog") else "paralog" if header.startswith("paralog") else "unknown"
    if role == "unknown":
        raise ValueError(f"Header must start with ortholog| or paralog|: {header!r}")
    return SeqRecord(header=header, sequence=seq_u, role=role)


def write_fasta(records: Iterable[SeqRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(f">{rec.header}\n")
            seq = rec.sequence
            for i in range(0, len(seq), 80):
                fh.write(seq[i : i + 80] + "\n")


def run_mafft(
    infile: Path,
    outfile: Path,
    *,
    mafft_bin: str = "mafft",
    threads: int = 1,
    timeout_s: int = 600,
) -> None:
    """Align ``infile`` with MAFFT ``--auto``; write aligned FASTA to ``outfile``."""
    if not infile.is_file() or infile.stat().st_size == 0:
        raise FileNotFoundError(f"MAFFT input missing/empty: {infile}")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        mafft_bin,
        "--auto",
        "--quiet",
        "--thread",
        str(max(1, int(threads))),
        str(infile),
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"mafft not found ({mafft_bin!r}). Install bioconda mafft or pass --mafft."
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"MAFFT failed for {infile} (rc={proc.returncode}): {proc.stderr[-2000:]}"
        )
    if not proc.stdout.strip():
        raise RuntimeError(f"MAFFT produced empty stdout for {infile}")
    outfile.write_text(proc.stdout, encoding="utf-8")


def consensus_rate(column_chars: list[str]) -> float:
    """Majority ACGT count / n_sequences (gaps dilute the rate).

    Returns NaN when fewer than 2 sequences or no non-gap bases.
    """
    n = len(column_chars)
    if n < 2:
        return float("nan")
    bases = [c for c in column_chars if c in DNA]
    if not bases:
        return float("nan")
    maj = Counter(bases).most_common(1)[0][1]
    return maj / float(n)


def column_metrics(
    aligned: list[SeqRecord],
) -> pd.DataFrame:
    """Per-position overall / ortholog / paralog consensus rates."""
    if not aligned:
        raise ValueError("No aligned sequences")
    L = len(aligned[0].sequence)
    if any(len(r.sequence) != L for r in aligned):
        raise ValueError("Aligned sequences have unequal lengths")

    roles = [r.role for r in aligned]
    matrix = [list(r.sequence) for r in aligned]
    n_all = len(aligned)
    ortho_idx = [i for i, role in enumerate(roles) if role == "ortholog"]
    para_idx = [i for i, role in enumerate(roles) if role == "paralog"]

    rows: list[dict[str, Any]] = []
    for pos in range(L):
        col_all = [matrix[i][pos] for i in range(n_all)]
        col_o = [matrix[i][pos] for i in ortho_idx]
        col_p = [matrix[i][pos] for i in para_idx]
        n_gap = sum(1 for c in col_all if c == "-" or c not in DNA)
        rows.append(
            {
                "position": pos + 1,  # 1-based
                "n_seqs": n_all,
                "n_orthologs": len(ortho_idx),
                "n_paralogs": len(para_idx),
                "n_non_gap": n_all - n_gap,
                "gap_fraction": n_gap / float(n_all),
                "overall_consensus_rate": consensus_rate(col_all),
                "orthologs_consensus_rate": consensus_rate(col_o),
                "paralogs_consensus_rate": consensus_rate(col_p),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class SizeNormModel:
    """Linear model: rate ~ a + b * log(n_group) + c * gap_fraction (train residual)."""

    target: str
    n_feature: str
    intercept: float
    coef_log_n: float
    coef_gap: float
    sigma: float
    n_train_rows: int
    n_train_clusters: int
    train_fraction: float
    seed: int

    def predict(self, log_n: np.ndarray, gap_fraction: np.ndarray) -> np.ndarray:
        return self.intercept + self.coef_log_n * log_n + self.coef_gap * gap_fraction

    def normalize_residual(self, rate: np.ndarray, log_n: np.ndarray, gap: np.ndarray) -> np.ndarray:
        pred = self.predict(log_n, gap)
        return rate - pred

    def normalize_ratio(self, rate: np.ndarray, log_n: np.ndarray, gap: np.ndarray) -> np.ndarray:
        pred = self.predict(log_n, gap)
        return rate / np.clip(pred, 1e-6, None)


def _fit_ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Ordinary least squares with intercept column already in X; return beta, residual sigma."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    sigma = float(np.sqrt(np.sum(resid**2) / dof))
    return beta, sigma


def fit_size_norm_models(
    position_tables: dict[str, pd.DataFrame],
    *,
    train_fraction: float = 0.7,
    seed: int = 42,
) -> tuple[dict[str, SizeNormModel], dict[str, Any]]:
    """Fit separate size-normalization models for overall / orthologs / paralogs.

    Training uses a random subset of *clusters* (not rows). Each model predicts
    the consensus rate from ``log(n_group)`` and ``gap_fraction`` so that larger
    alignments are not systematically scored as less conserved.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0,1), got {train_fraction}")
    names = sorted(position_tables)
    if len(names) < 4:
        raise ValueError(f"Need >=4 clusters to fit norm models, got {len(names)}")
    rng = random.Random(seed)
    n_train = max(2, int(round(len(names) * train_fraction)))
    n_train = min(n_train, len(names) - 1)
    train_names = set(rng.sample(names, n_train))
    held_out = sorted(set(names) - train_names)

    specs = [
        ("overall_consensus_rate", "n_seqs", "overall"),
        ("orthologs_consensus_rate", "n_orthologs", "orthologs"),
        ("paralogs_consensus_rate", "n_paralogs", "paralogs"),
    ]
    models: dict[str, SizeNormModel] = {}
    for target, n_col, key in specs:
        ys: list[float] = []
        log_ns: list[float] = []
        gaps: list[float] = []
        for name in train_names:
            df = position_tables[name]
            n_vals = df[n_col].to_numpy(dtype=float)
            rates = df[target].to_numpy(dtype=float)
            gap_vals = df["gap_fraction"].to_numpy(dtype=float)
            for rate, n, gap in zip(rates, n_vals, gap_vals, strict=True):
                if not math.isfinite(float(rate)) or float(n) < 2:
                    continue
                ys.append(float(rate))
                log_ns.append(math.log(float(n)))
                gaps.append(float(gap))
        if len(ys) < 50:
            raise ValueError(f"Too few training rows for {target}: {len(ys)}")
        y = np.asarray(ys, dtype=float)
        X = np.column_stack(
            [
                np.ones(len(y)),
                np.asarray(log_ns, dtype=float),
                np.asarray(gaps, dtype=float),
            ]
        )
        beta, sigma = _fit_ols(y, X)
        models[key] = SizeNormModel(
            target=target,
            n_feature=n_col,
            intercept=float(beta[0]),
            coef_log_n=float(beta[1]),
            coef_gap=float(beta[2]),
            sigma=sigma,
            n_train_rows=len(y),
            n_train_clusters=len(train_names),
            train_fraction=train_fraction,
            seed=seed,
        )
    meta = {
        "train_clusters": sorted(train_names),
        "held_out_clusters": held_out,
        "n_clusters": len(names),
        "train_fraction": train_fraction,
        "seed": seed,
    }
    return models, meta


def apply_size_norm(df: pd.DataFrame, models: dict[str, SizeNormModel]) -> pd.DataFrame:
    """Add residual and ratio normalizations for the three consensus rates."""
    out = df.copy()
    mapping = [
        ("overall", "overall_consensus_rate", "n_seqs"),
        ("orthologs", "orthologs_consensus_rate", "n_orthologs"),
        ("paralogs", "paralogs_consensus_rate", "n_paralogs"),
    ]
    for key, rate_col, n_col in mapping:
        model = models[key]
        rate = out[rate_col].to_numpy(dtype=float)
        n = out[n_col].to_numpy(dtype=float)
        gap = out["gap_fraction"].to_numpy(dtype=float)
        log_n = np.log(np.clip(n, 2, None))
        valid = np.isfinite(rate) & (n >= 2)
        resid = np.full(len(out), np.nan, dtype=float)
        ratio = np.full(len(out), np.nan, dtype=float)
        z = np.full(len(out), np.nan, dtype=float)
        if valid.any():
            resid[valid] = model.normalize_residual(rate[valid], log_n[valid], gap[valid])
            ratio[valid] = model.normalize_ratio(rate[valid], log_n[valid], gap[valid])
            z[valid] = resid[valid] / max(model.sigma, 1e-12)
        out[f"{rate_col}_norm_residual"] = resid
        out[f"{rate_col}_norm_ratio"] = ratio
        out[f"{rate_col}_norm_z"] = z
        out[f"{rate_col}_expected"] = np.where(
            valid, model.predict(log_n, gap), np.nan
        )
    return out


def models_to_jsonable(
    models: dict[str, SizeNormModel],
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {k: asdict(v) for k, v in models.items()}
    if meta is not None:
        out["_meta"] = meta
    return out


def load_models_json(path: Path) -> tuple[dict[str, SizeNormModel], dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    meta = dict(raw.pop("_meta", {}))
    models = {k: SizeNormModel(**v) for k, v in raw.items()}
    return models, meta


def align_one_cluster(
    fna_path: Path,
    aln_path: Path,
    *,
    mafft_bin: str,
    threads: int = 1,
) -> list[SeqRecord]:
    """Run MAFFT on a cluster FASTA; return aligned records (roles preserved)."""
    raw = parse_fasta(fna_path)
    # MAFFT preserves headers after first whitespace; keep full header as single token
    safe = [
        SeqRecord(header=r.header.replace(" ", "_"), sequence=r.sequence, role=r.role)
        for r in raw
    ]
    with tempfile.TemporaryDirectory(prefix="mafft_opg_") as td:
        inp = Path(td) / "in.fa"
        write_fasta(safe, inp)
        run_mafft(inp, aln_path, mafft_bin=mafft_bin, threads=threads)
    aligned = parse_fasta(aln_path)
    # restore role from header prefix
    return aligned


def process_cluster(
    fna_path: Path,
    outdir: Path,
    *,
    mafft_bin: str,
    threads: int = 1,
) -> tuple[str, pd.DataFrame]:
    """Align one cluster and compute raw per-position consensus metrics."""
    stem = fna_path.stem  # cluster_0
    aln_path = outdir / f"{stem}.aln.fa"
    aligned = align_one_cluster(fna_path, aln_path, mafft_bin=mafft_bin, threads=threads)
    metrics = column_metrics(aligned)
    metrics.insert(0, "cluster", stem)
    return stem, metrics


def discover_fna(indir: Path) -> list[Path]:
    paths = sorted(indir.glob("cluster_*.fna"))
    if not paths:
        raise FileNotFoundError(f"No cluster_*.fna under {indir}")
    return paths
