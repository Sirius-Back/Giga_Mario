"""Prepare the locked 3+1 genome continuous-target run0 LegNet panel.

Run from the project root:
  python -m src.run.preprocess_run0 \
    --gtf run/run0/input/gtf --fna run/run0/input/fna --target run/run0/input/tpm \
    --outdir run/run0 --environment gene \
    --window '{"pos1":-100,"pos2":100}' --to-type legnet
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from src.get_mpra import discover_tpm_csvs, run_get_mpra
from src.pipeline.adapt import run_adapt
from src.pipeline.generate_fold import run_generate_fold
from src.pipeline.id_gen import run_id_gen
from src.pipeline.parse_data import run_parse_data
from src.pipeline.parse_target import run_parse_target
from src.preprocess_report import collect_preprocess_checks, write_parse_md
from src.preprocessing import genome_prefix

PANEL_GENOMES = (
    "GCF_000210835.1_ASM21083v1",
    "GCF_000006845.1_ASM684v1",
    "GCF_000005845.2_ASM584v2",
    "GCF_000009045.1_ASM904v1",
)
EXPECTED_GENOMES = frozenset(genome_prefix(name) for name in PANEL_GENOMES)
ZSV_GENOME = genome_prefix("GCF_000210835.1_ASM21083v1")
STALE_ARTIFACTS = (
    "TARGET",
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
    "legnet_input",
    "adversarial",
)


def _require_nonempty_dir(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label} directory: {path}")
    if not any(path.iterdir()):
        raise ValueError(f"{label} directory is empty: {path}")
    return path


def _nonempty_genomes(folder: Path, patterns: tuple[str, ...], label: str) -> set[str]:
    paths = [
        path
        for pattern in patterns
        for path in folder.glob(pattern)
        if path.is_file() and path.stat().st_size > 0
    ]
    genomes = {genome_prefix(path.name) for path in paths}
    if len(genomes) != len(paths):
        raise ValueError(f"{label} contains duplicate files for at least one genome")
    if genomes != EXPECTED_GENOMES:
        missing = sorted(EXPECTED_GENOMES - genomes)
        unexpected = sorted(genomes - EXPECTED_GENOMES)
        raise ValueError(
            f"{label} must contain exactly the locked 3+1 subset; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return genomes


def _require_panel_inputs(gtf: Path, fna: Path, tpm: Path) -> None:
    gtf_ids = _nonempty_genomes(gtf, ("*.gtf", "*.gtf.gz"), "GTF")
    fna_ids = {
        genome
        for genome in _nonempty_genomes(
            fna, ("*.fna", "*.fa", "*.fasta", "*.fna.gz", "*.fa.gz", "*.fasta.gz"), "FNA"
        )
    }
    tpm_ids = _nonempty_genomes(tpm, ("*.csv",), "TARGET/TPM")
    if gtf_ids != fna_ids or gtf_ids != tpm_ids:
        raise ValueError("Locked GTF, FNA, and TPM genome subsets do not match")


def _write_prepare_fold(id_csv: Path, outdir: Path) -> Path:
    """Write the single locked ZSV rule using the actual ID.csv genome value."""
    import csv

    with id_csv.open(newline="", encoding="utf-8") as fh:
        genomes = {row["genome"] for row in csv.DictReader(fh, delimiter="|")}
    if {genome_prefix(g) for g in genomes} != EXPECTED_GENOMES:
        raise ValueError("ID.csv genomes do not match the locked 3+1 subset")
    matches = sorted(g for g in genomes if genome_prefix(g) == ZSV_GENOME)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one ID.csv genome matching {ZSV_GENOME}; found {matches or 'none'}"
        )
    path = outdir / "prepare_fold.csv"
    path.write_text(
        "identificator|column|fold\n"
        + f"{matches[0]}|genome|zsv\n",
        encoding="utf-8",
    )
    return path


def _clear_stale_artifacts(outdir: Path) -> None:
    """Remove prior generated stages while preserving the locked input symlinks."""
    for name in STALE_ARTIFACTS:
        path = outdir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare run0 prokaryote LegNet panel")
    parser.add_argument("--gtf", required=True, type=Path)
    parser.add_argument("--fna", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path, help="Source wide TPM folder")
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--environment", required=True, choices=["gene", "random"])
    parser.add_argument("--window", required=True, help="JSON signed offsets")
    parser.add_argument("--to-type", required=True, choices=["caduceus", "legnet"])
    parser.add_argument("--gtf-column", default="gene")
    args = parser.parse_args(argv)

    gtf = _require_nonempty_dir(args.gtf, "GTF")
    fna = _require_nonempty_dir(args.fna, "FNA")
    tpm = _require_nonempty_dir(args.target, "TARGET/TPM")
    _require_panel_inputs(gtf, fna, tpm)
    discover_tpm_csvs(tpm)
    try:
        window = json.loads(args.window)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--window must be valid JSON: {exc.msg}") from exc
    if window != {"pos1": -100, "pos2": 100}:
        raise ValueError("run0 is locked to window {'pos1': -100, 'pos2': 100}")
    if args.environment != "gene" or args.to_type != "legnet":
        raise ValueError("run0 is locked to environment=gene and to_type=legnet")

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    _clear_stale_artifacts(outdir)
    mpra_target = outdir / "TARGET"
    run_get_mpra(tpm, mpra_target, mode="continuous")
    id_csv = run_id_gen(gtf, gtf_column=args.gtf_column, outdir=outdir)
    run_adapt(gtf, fna, outdir=outdir, id_csv=id_csv, environment=args.environment, window=window)
    run_parse_data(outdir / "MARKED", outdir=outdir, to_type=args.to_type)
    run_parse_target(mpra_target, outdir=outdir, id_csv=id_csv, to_type=args.to_type)
    prepare_fold = _write_prepare_fold(id_csv, outdir)
    run_generate_fold(id_csv, prepare_fold, outdir=outdir)
    write_parse_md(outdir, id_csv=id_csv, require_fold=True)
    checks = collect_preprocess_checks(outdir=outdir, id_csv=id_csv, require_fold=True)
    if not checks["ok"]:
        raise RuntimeError(f"Preprocess report failed validation: {checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
