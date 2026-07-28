#!/usr/bin/env python3
"""TPM → LegNet-style MPRA targets (soft bin fractions or continuous log2).

Wide TPM CSVs (header = gene_id, one numeric row) → same layout with values
replaced by either:

- ``soft`` (default): continuous bin fractions on an 18-bin scale ``[0, n_bins-1]``
- ``continuous``: ``log2(TPM+1)`` (optionally min–max scaled to ``[0, 1]``)

Transform (per file by default):
  1. log2(TPM + 1)
  2. soft: global/per-file min/max → map into [0, n_bins - 1]
     continuous: write log2 values as-is, or scale to [0, 1] with ``--scale-01``

Example:
  python -m src.get_mpra --tpm prokaryotes/tpm --outfolder prokaryotes/mpra
  python -m src.get_mpra --tpm prokaryotes/tpm --outfolder out --mode continuous
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
MODES = ("soft", "continuous")


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


def scale_01(
    log_vals: Sequence[float],
    *,
    ymin: float,
    ymax: float,
) -> list[float]:
    """Min–max scale log values onto ``[0, 1]`` (degenerate → all zeros)."""
    if ymax < ymin:
        raise ValueError(f"ymax < ymin: {ymax} < {ymin}")
    if ymax == ymin:
        return [0.0] * len(log_vals)
    scale = 1.0 / (ymax - ymin)
    return [(x - ymin) * scale for x in log_vals]


def transform_file(
    path: Path,
    *,
    ymin: float,
    ymax: float,
    n_bins: int,
    mode: str = "soft",
    scale_01_flag: bool = False,
) -> tuple[list[str], list[float], list[float]]:
    """Return ``(genes, logs, targets)`` for one wide TPM CSV."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    genes, tpm = read_wide_row(path)
    logs = log2_tpm_plus_one(tpm)
    if mode == "soft":
        targets = bin_fractions(logs, ymin=ymin, ymax=ymax, n_bins=n_bins)
    elif scale_01_flag:
        targets = scale_01(logs, ymin=ymin, ymax=ymax)
    else:
        targets = list(logs)
    return genes, logs, targets


def run_get_mpra(
    tpm_dir: Path,
    outfolder: Path,
    *,
    n_bins: int = DEFAULT_N_BINS,
    shared_scale: bool = True,
    mode: str = "soft",
    scale_01_flag: bool = False,
) -> dict[str, Any]:
    """Convert all wide TPM CSVs under ``tpm_dir`` into MPRA-style target CSVs.

    Parameters
    ----------
    shared_scale:
        If True (default), one global log2 min/max across all input files.
        If False, each file uses its own min/max.
        Used for ``soft`` always; for ``continuous`` only when ``scale_01_flag``.
    mode:
        ``soft`` — LegNet soft-classification bin fractions on ``[0, n_bins-1]``.
        ``continuous`` — ``log2(TPM+1)``, or ``[0, 1]`` if ``scale_01_flag``.
    scale_01_flag:
        Only for ``continuous``: min–max scale log2 onto ``[0, 1]``.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if scale_01_flag and mode != "continuous":
        raise ValueError("--scale-01 / scale_01_flag only valid with mode=continuous")

    tpm_dir = Path(tpm_dir)
    outfolder = Path(outfolder)
    files = discover_tpm_csvs(tpm_dir)
    outfolder.mkdir(parents=True, exist_ok=True)

    need_scale = mode == "soft" or scale_01_flag
    global_ymin: float | None = None
    global_ymax: float | None = None
    if need_scale and shared_scale:
        global_ymin, global_ymax, _ = compute_global_log_range(files)

    per_file: list[dict[str, Any]] = []
    for path in files:
        if need_scale:
            if shared_scale:
                assert global_ymin is not None and global_ymax is not None
                ymin, ymax = global_ymin, global_ymax
            else:
                _, tpm = read_wide_row(path)
                logs = log2_tpm_plus_one(tpm)
                ymin, ymax = min(logs), max(logs)
        else:
            # continuous raw log2 — scale bounds unused; keep placeholders for meta
            ymin, ymax = 0.0, 0.0

        genes, logs, targets = transform_file(
            path,
            ymin=ymin,
            ymax=ymax,
            n_bins=n_bins,
            mode=mode,
            scale_01_flag=scale_01_flag,
        )
        out_path = outfolder / path.name
        write_wide_row(out_path, genes, targets)
        entry: dict[str, Any] = {
            "input": str(path),
            "output": str(out_path),
            "n_genes": len(genes),
            "log2_min": min(logs),
            "log2_max": max(logs),
            "target_min": min(targets) if targets else None,
            "target_max": max(targets) if targets else None,
        }
        if need_scale:
            entry["scale_ymin"] = ymin
            entry["scale_ymax"] = ymax
        if mode == "soft":
            entry["bin_frac_min"] = entry["target_min"]
            entry["bin_frac_max"] = entry["target_max"]
        per_file.append(entry)

    if mode == "soft":
        transform = "bin_fraction = (log2(TPM+1) - ymin) / (ymax - ymin) * (n_bins - 1)"
    elif scale_01_flag:
        transform = "continuous = (log2(TPM+1) - ymin) / (ymax - ymin)  # [0, 1]"
    else:
        transform = "continuous = log2(TPM+1)"

    meta: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tpm_dir": str(tpm_dir),
        "outfolder": str(outfolder),
        "mode": mode,
        "scale_01": scale_01_flag,
        "n_bins": n_bins if mode == "soft" else None,
        "shared_scale": shared_scale if need_scale else None,
        "transform": transform,
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
            "Transform wide TPM CSVs to LegNet MPRA targets: "
            f"soft bin fractions (default {DEFAULT_N_BINS} bins) or continuous log2(TPM+1)."
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
        "--mode",
        choices=list(MODES),
        default="soft",
        help="soft = bin fractions [0, n_bins-1] (default); continuous = log2(TPM+1)",
    )
    p.add_argument(
        "--n-bins",
        type=int,
        default=DEFAULT_N_BINS,
        help=f"Number of equal bins for mode=soft (default {DEFAULT_N_BINS})",
    )
    p.add_argument(
        "--per-file-scale",
        action="store_true",
        help="Use per-file log2 min/max instead of one global scale (soft / continuous --scale-01)",
    )
    p.add_argument(
        "--scale-01",
        action="store_true",
        dest="scale_01",
        help="With mode=continuous: min–max scale log2(TPM+1) onto [0, 1]",
    )
    args = p.parse_args(argv)

    if args.mode == "soft" and args.n_bins < 2:
        print(f"ERROR: --n-bins must be >= 2, got {args.n_bins}", file=sys.stderr)
        return 2
    if args.scale_01 and args.mode != "continuous":
        print("ERROR: --scale-01 requires --mode continuous", file=sys.stderr)
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
            mode=args.mode,
            scale_01_flag=args.scale_01,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"Wrote {meta['n_files']} MPRA target CSVs → {meta['outfolder']} "
        f"(mode={meta['mode']}, n_bins={meta['n_bins']}, shared_scale={meta['shared_scale']})"
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
