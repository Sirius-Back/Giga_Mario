#!/usr/bin/env python3
"""TPM → LegNet-style soft-classification bin fractions (MPRA targets).

Wide TPM CSVs (header = gene_id, one numeric row) → same layout with values
replaced by continuous bin fractions on an 18-bin scale.

Transform (per file by default):
  1. log2(TPM + 1)
  2. global min/max of those log values (within the scale scope)
  3. map into [0, n_bins - 1] with n_bins equal steps (default 18 → [0, 17])

Example:
  python -m src.get_mpra --tpm prokaryotes/tpm --outfolder prokaryotes/mpra
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

DEFAULT_N_BINS = 18


def read_wide_row(path: Path) -> tuple[list[str], list[float]]:
    """Read wide TPM CSV → (gene_ids in header order, values)."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Empty or missing TPM CSV: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        raise ValueError(f"TPM CSV needs header + data row: {path}")
    header, values = rows[0], rows[1]
    if len(header) != len(values):
        raise ValueError(
            f"Header/value length mismatch in {path}: {len(header)} vs {len(values)}"
        )
    if not header:
        raise ValueError(f"No genes in {path}")
    out_vals: list[float] = []
    for i, (g, v) in enumerate(zip(header, values)):
        if not g:
            raise ValueError(f"Empty gene_id at column {i} in {path}")
        try:
            out_vals.append(float(v))
        except ValueError as exc:
            raise ValueError(f"Non-numeric TPM for {g!r} in {path}: {v!r}") from exc
    return list(header), out_vals


def write_wide_row(path: Path, genes: Sequence[str], values: Sequence[float]) -> None:
    """Write wide CSV preserving gene order (same format as prokaryotes/tpm)."""
    path = Path(path)
    if len(genes) != len(values):
        raise ValueError(f"genes/values length mismatch: {len(genes)} vs {len(values)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(list(genes))
        w.writerow([f"{float(v):.6f}" for v in values])


def log2_tpm_plus_one(tpm: Sequence[float]) -> list[float]:
    out: list[float] = []
    for v in tpm:
        if v < 0:
            raise ValueError(f"Negative TPM not allowed: {v}")
        out.append(math.log2(v + 1.0))
    return out


def bin_fractions(
    log_vals: Sequence[float],
    *,
    ymin: float,
    ymax: float,
    n_bins: int = DEFAULT_N_BINS,
) -> list[float]:
    """Map log-transformed targets onto [0, n_bins - 1] (18 bins → [0, 17]).

    Equal steps of width (ymax - ymin) / n_bins; continuous fraction is the
    position along that grid (LegNet soft-classification ``bin`` coordinate).
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    if ymax < ymin:
        raise ValueError(f"ymax < ymin: {ymax} < {ymin}")
    if ymax == ymin:
        # Degenerate dynamic range → place all mass at bin 0
        return [0.0] * len(log_vals)
    scale = float(n_bins - 1) / (ymax - ymin)
    return [(x - ymin) * scale for x in log_vals]


def discover_tpm_csvs(tpm_dir: Path) -> list[Path]:
    tpm_dir = Path(tpm_dir)
    if not tpm_dir.is_dir():
        raise FileNotFoundError(f"TPM folder missing: {tpm_dir}")
    files = sorted(p for p in tpm_dir.glob("*.csv") if p.is_file() and p.stat().st_size > 0)
    if not files:
        raise FileNotFoundError(f"No non-empty *.csv under {tpm_dir}")
    return files


def compute_global_log_range(
    files: Sequence[Path],
) -> tuple[float, float, int]:
    """Min/max of log2(TPM+1) across all genes in all files; total gene slots."""
    ymin = math.inf
    ymax = -math.inf
    n = 0
    for path in files:
        _, vals = read_wide_row(path)
        logs = log2_tpm_plus_one(vals)
        n += len(logs)
        ymin = min(ymin, min(logs))
        ymax = max(ymax, max(logs))
    if n == 0:
        raise ValueError("No TPM values found for global scale")
    return float(ymin), float(ymax), n


def transform_file(
    path: Path,
    *,
    ymin: float,
    ymax: float,
    n_bins: int,
) -> tuple[list[str], list[float], list[float]]:
    genes, tpm = read_wide_row(path)
    logs = log2_tpm_plus_one(tpm)
    fracs = bin_fractions(logs, ymin=ymin, ymax=ymax, n_bins=n_bins)
    return genes, logs, fracs


def run_get_mpra(
    tpm_dir: Path,
    outfolder: Path,
    *,
    n_bins: int = DEFAULT_N_BINS,
    shared_scale: bool = True,
) -> dict[str, Any]:
    """Convert all wide TPM CSVs under ``tpm_dir`` into bin-fraction CSVs.

    Parameters
    ----------
    shared_scale:
        If True (default), one global log2 min/max across all input files.
        If False, each file uses its own min/max.
    """
    tpm_dir = Path(tpm_dir)
    outfolder = Path(outfolder)
    files = discover_tpm_csvs(tpm_dir)
    outfolder.mkdir(parents=True, exist_ok=True)

    global_ymin: float | None = None
    global_ymax: float | None = None
    if shared_scale:
        global_ymin, global_ymax, _ = compute_global_log_range(files)

    per_file: list[dict[str, Any]] = []
    for path in files:
        if shared_scale:
            assert global_ymin is not None and global_ymax is not None
            ymin, ymax = global_ymin, global_ymax
        else:
            _, tpm = read_wide_row(path)
            logs = log2_tpm_plus_one(tpm)
            ymin, ymax = min(logs), max(logs)

        genes, logs, fracs = transform_file(path, ymin=ymin, ymax=ymax, n_bins=n_bins)
        out_path = outfolder / path.name
        write_wide_row(out_path, genes, fracs)
        per_file.append(
            {
                "input": str(path),
                "output": str(out_path),
                "n_genes": len(genes),
                "log2_min": min(logs),
                "log2_max": max(logs),
                "scale_ymin": ymin,
                "scale_ymax": ymax,
                "bin_frac_min": min(fracs) if fracs else None,
                "bin_frac_max": max(fracs) if fracs else None,
            }
        )

    meta: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tpm_dir": str(tpm_dir),
        "outfolder": str(outfolder),
        "n_bins": n_bins,
        "shared_scale": shared_scale,
        "transform": "bin_fraction = (log2(TPM+1) - ymin) / (ymax - ymin) * (n_bins - 1)",
        "global_log2_min": global_ymin,
        "global_log2_max": global_ymax,
        "n_files": len(per_file),
        "files": per_file,
    }
    meta_path = outfolder / "get_mpra_scale.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    meta["scale_json"] = str(meta_path)
    return meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Transform wide TPM CSVs to LegNet soft-classification bin fractions "
            f"(default {DEFAULT_N_BINS} bins on log2(TPM+1))."
        )
    )
    p.add_argument(
        "--tpm",
        type=Path,
        required=True,
        help="Input folder of wide TPM *.csv (header=gene_id, one data row)",
    )
    p.add_argument(
        "--outfolder",
        "--out",
        dest="outfolder",
        type=Path,
        required=True,
        help="Output folder (same filenames / wide CSV layout)",
    )
    p.add_argument(
        "--n-bins",
        type=int,
        default=DEFAULT_N_BINS,
        help=f"Number of equal bins (default {DEFAULT_N_BINS})",
    )
    p.add_argument(
        "--per-file-scale",
        action="store_true",
        help="Use per-file log2 min/max instead of one global scale across all files",
    )
    args = p.parse_args(argv)

    if args.n_bins < 2:
        print(f"ERROR: --n-bins must be >= 2, got {args.n_bins}", file=sys.stderr)
        return 2
    if not args.tpm.is_dir():
        print(f"ERROR: TPM folder missing: {args.tpm}", file=sys.stderr)
        return 2

    try:
        meta = run_get_mpra(
            args.tpm,
            args.outfolder,
            n_bins=args.n_bins,
            shared_scale=not args.per_file_scale,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Wrote {meta['n_files']} MPRA target CSVs → {meta['outfolder']} "
        f"(n_bins={meta['n_bins']}, shared_scale={meta['shared_scale']})"
    )
    if meta.get("global_log2_min") is not None:
        print(
            f"  global log2(TPM+1) range: "
            f"[{meta['global_log2_min']:.6f}, {meta['global_log2_max']:.6f}]"
        )
    print(f"  scale sidecar: {meta['scale_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
