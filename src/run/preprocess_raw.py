"""Preprocess eukaryote ``raw/`` panel → gene±100 bp windows + log10(TPM+1).

Supports ``--to-type caduceus|legnet`` (same window / transform / mappings contract).

Run from the project root:
  conda run -n legnet python -m src.run.preprocess_raw \\
    --gtf raw/gtf --fna raw/fna --target raw/tpm \\
    --outdir ready_legnet --environment gene \\
    --window '{"pos1":-100,"pos2":100}' --to-type legnet \\
    --mappings raw/random_borzoi_expr_file_mappings.csv \\
    --transform log10p1
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.pipeline.adapt import run_adapt
from src.pipeline.id_gen import run_id_gen
from src.pipeline.parse_data import run_parse_data
from src.pipeline.parse_target import VALUE_TRANSFORMS, run_parse_target
from src.preprocess_report import collect_preprocess_checks, write_parse_md
from src.preprocessing import genome_prefix

STALE_ARTIFACTS = (
    "ID.csv",
    "MARKED",
    "PARSED",
    "PREDICT",
    "intersect.csv",
    "parse_data_stats.json",
    "prepare_fold.csv",
    "fold.csv",
    "parse.md",
    "split.csv",
    "SPLIT",
)

LOCKED_WINDOW = {"pos1": -100, "pos2": 100}


def _require_nonempty_dir(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label} directory: {path}")
    if not any(path.iterdir()):
        raise ValueError(f"{label} directory is empty: {path}")
    return path


def _require_mappings(path: Path, gtf: Path, fna: Path, target: Path) -> Path:
    """Fail early unless every mapping row has FNA+GTF+TPM on disk."""
    import csv

    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Mappings CSV missing or empty: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    required = {"id", "tpm", "genome"}
    if not rows:
        raise ValueError(f"Mappings CSV has no rows: {path}")
    if missing := required - set(rows[0].keys()):
        raise ValueError(f"Mappings CSV missing columns {sorted(missing)}: {path}")

    fna_ids = {
        genome_prefix(p.name)
        for p in fna.iterdir()
        if p.is_file() and p.suffix in {".fna", ".fa", ".fasta"} and p.stat().st_size > 0
    }
    gtf_ids = {
        genome_prefix(p.name)
        for p in gtf.iterdir()
        if p.is_file() and (p.name.endswith(".gtf") or p.name.endswith(".gtf.gz")) and p.stat().st_size > 0
    }
    issues: list[str] = []
    seen_ids: set[str] = set()
    seen_genomes: set[str] = set()
    for row in rows:
        sid = (row.get("id") or "").strip()
        genome = genome_prefix((row.get("genome") or "").strip())
        tpm_name = Path((row.get("tpm") or "").strip()).name
        if not sid:
            issues.append("empty sample id")
            continue
        if sid in seen_ids:
            issues.append(f"duplicate sample id {sid}")
        seen_ids.add(sid)
        if genome in seen_genomes:
            issues.append(f"duplicate genome {genome}")
        seen_genomes.add(genome)
        if genome not in fna_ids:
            issues.append(f"{sid}: missing FNA for {genome}")
        if genome not in gtf_ids:
            issues.append(f"{sid}: missing GTF for {genome}")
        tpm_path = target / tpm_name
        if not tpm_path.is_file() or tpm_path.stat().st_size == 0:
            issues.append(f"{sid}: missing TPM {tpm_path}")
    if issues:
        raise FileNotFoundError(
            "Mappings not complete for raw preprocess:\n  - " + "\n  - ".join(issues)
        )
    return path


def _clear_stale_artifacts(outdir: Path) -> None:
    for name in STALE_ARTIFACTS:
        path = outdir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess raw/ → ready_* (gene±100 bp, log10(TPM+1), caduceus|legnet)"
    )
    parser.add_argument("--gtf", required=True, type=Path)
    parser.add_argument("--fna", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path, help="Wide TPM folder")
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--environment", required=True, choices=["gene", "random"])
    parser.add_argument("--window", required=True, help="JSON signed offsets")
    parser.add_argument("--to-type", required=True, choices=["caduceus", "legnet"])
    parser.add_argument("--mappings", required=True, type=Path)
    parser.add_argument(
        "--transform",
        default="log10p1",
        choices=sorted(VALUE_TRANSFORMS),
        help="Label transform (default log10p1 for this panel)",
    )
    parser.add_argument("--gtf-column", default="gene")
    parser.add_argument(
        "--prepare-fold",
        type=Path,
        default=None,
        help="Optional prepare_fold.csv for ZSV → fold.csv",
    )
    args = parser.parse_args(argv)

    gtf = _require_nonempty_dir(args.gtf, "GTF")
    fna = _require_nonempty_dir(args.fna, "FNA")
    target = _require_nonempty_dir(args.target, "TARGET/TPM")
    mappings = _require_mappings(args.mappings, gtf=gtf, fna=fna, target=target)

    try:
        window = json.loads(args.window)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--window must be valid JSON: {exc.msg}") from exc
    if window != LOCKED_WINDOW:
        raise ValueError(f"preprocess_raw is locked to window {LOCKED_WINDOW}, got {window}")
    if args.environment != "gene":
        raise ValueError("preprocess_raw is locked to environment=gene")
    if args.to_type not in {"caduceus", "legnet"}:
        raise ValueError("preprocess_raw to_type must be caduceus|legnet")
    if args.transform != "log10p1":
        raise ValueError("preprocess_raw is locked to transform=log10p1")

    outdir = Path(args.outdir)
    if outdir.exists():
        # User-requested: remove current outdir before rebuild.
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    id_csv = run_id_gen(gtf, gtf_column=args.gtf_column, outdir=outdir)
    run_adapt(
        gtf,
        fna,
        outdir=outdir,
        id_csv=id_csv,
        environment=args.environment,
        window=window,
    )
    run_parse_data(outdir / "MARKED", outdir=outdir, to_type=args.to_type)
    run_parse_target(
        target,
        outdir=outdir,
        id_csv=id_csv,
        to_type=args.to_type,
        mappings=mappings,
        transform=args.transform,
    )

    require_fold = args.prepare_fold is not None
    if require_fold:
        from src.pipeline.generate_fold import run_generate_fold

        run_generate_fold(id_csv, args.prepare_fold, outdir=outdir)

    write_parse_md(outdir, id_csv=id_csv, require_fold=require_fold)
    checks = collect_preprocess_checks(
        outdir=outdir, id_csv=id_csv, require_fold=require_fold
    )
    if not checks["ok"]:
        raise RuntimeError(f"Preprocess report failed validation: {checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
