#!/usr/bin/env python3
"""Adapt: gene±flank DNA windows + continuous TPM → Caduceus-ready adapt/.

Does NOT perform train/val/test splitting.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # type: ignore

try:
    from pyfaidx import Fasta  # type: ignore
except ImportError:
    Fasta = None  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.default.yaml"


@dataclass
class GenomeUnit:
    genome_id: str
    fold: str | None
    fasta: Path
    genes: Path
    tpm: Path
    gtf: Path | None = None
    species: str | None = None
    source: str = ""


@dataclass
class AuditResult:
    mode: str  # split | raw | empty
    units: list[GenomeUnit] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def load_config(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML required: pip/conda install pyyaml")
    cfg = yaml.safe_load(path.read_text()) or {}
    return cfg


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "sample"


def find_named(dir_path: Path, names: list[str]) -> Path | None:
    for n in names:
        p = dir_path / n
        if p.exists():
            return p
    # glob soft match
    for n in names:
        hits = list(dir_path.glob(n))
        if hits:
            return hits[0]
    return None


def audit_repo(root: Path, cfg: dict[str, Any], input_spec: str) -> AuditResult:
    paths = cfg.get("paths", {})
    files = cfg.get("files", {})
    fasta_names = files.get("fasta_names", ["genome.fna"])
    genes_names = files.get("genes_names", ["genes.tsv"])
    tpm_names = files.get("tpm_names", ["expression_tpm.csv"])
    gtf_names = files.get("gtf_names", ["annotation.gtf"])

    result = AuditResult(mode="empty")

    def add_unit(gdir: Path, fold: str | None, genome_id: str | None = None) -> None:
        gid = genome_id or gdir.name
        fa = find_named(gdir, fasta_names)
        ge = find_named(gdir, genes_names)
        tp = find_named(gdir, tpm_names)
        gt = find_named(gdir, gtf_names)
        if not fa or not ge or not tp:
            miss = []
            if not fa:
                miss.append(f"fasta in {gdir}")
            if not ge:
                miss.append(f"genes.tsv in {gdir}")
            if not tp:
                miss.append(f"TPM in {gdir}")
            result.missing.extend(miss)
            return
        result.units.append(
            GenomeUnit(
                genome_id=gid,
                fold=fold,
                fasta=fa,
                genes=ge,
                tpm=tp,
                gtf=gt,
                source=str(gdir),
            )
        )

    # Resolve input root
    if input_spec == "auto":
        split_root = root / paths.get("prefer_split_root", "data_splits/full")
        if split_root.is_dir() and any((split_root / f).is_dir() for f in ("train", "val", "test")):
            input_path = split_root
            result.notes.append(f"auto: using split root {split_root}")
        else:
            ref = root / paths.get("reformat_root", "data/reformat/random_full")
            input_path = ref if ref.is_dir() else root / paths.get("raw_genomes_root", "data/raw/genomes")
            result.notes.append(f"auto: no splits; using {input_path}")
    else:
        input_path = (root / input_spec).resolve() if not Path(input_spec).is_absolute() else Path(input_spec)

    if not input_path.exists():
        result.missing.append(f"input path missing: {input_path}")
        return result

    # Split layout?
    fold_dirs = [d for d in ("train", "val", "validation", "test") if (input_path / d).is_dir()]
    if fold_dirs:
        result.mode = "split"
        for fold in fold_dirs:
            fold_norm = "val" if fold == "validation" else fold
            for gdir in sorted(p for p in (input_path / fold).iterdir() if p.is_dir()):
                add_unit(gdir, fold_norm)
        # enrich species from fold_manifest if present
        man = root / paths.get("split_manifest", "data_splits/full/fold_manifest.tsv")
        if man.is_file():
            by_id = {r["sample_id"]: r for r in csv.DictReader(man.open(), delimiter="\t")}
            for u in result.units:
                meta = by_id.get(u.genome_id)
                if meta:
                    u.species = meta.get("species")
        return result

    # Flat genome dirs
    result.mode = "raw"
    subdirs = sorted(p for p in input_path.iterdir() if p.is_dir())
    if subdirs:
        for gdir in subdirs:
            add_unit(gdir, None)
    else:
        # maybe input_path itself is one genome
        add_unit(input_path, None, genome_id=input_path.name)
    return result


def load_tpm(path: Path) -> dict[str, float]:
    """Support wide CSV (header=genes, one value row) or long gene,tpm TSV/CSV."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return {}
    # detect delimiter
    delim = "," if lines[0].count(",") >= lines[0].count("\t") else "\t"
    header = lines[0].split(delim)
    if len(lines) >= 2 and len(header) > 2 and not any(
        h.lower() in {"gene", "gene_id", "gene_name", "tpm"} for h in header[:3]
    ):
        # wide matrix: one row of values
        values = lines[1].split(delim)
        out: dict[str, float] = {}
        for g, v in zip(header, values):
            try:
                out[g] = float(v)
            except ValueError:
                continue
        return out
    # long format
    reader = csv.DictReader(lines, delimiter=delim)
    out = {}
    for rec in reader:
        key = rec.get("gene_name") or rec.get("gene_id") or rec.get("gene") or rec.get("id")
        val = rec.get("TPM") or rec.get("tpm") or rec.get("value")
        if key is None or val is None:
            continue
        try:
            out[key] = float(val)
        except ValueError:
            continue
    return out


def open_fasta(path: Path):
    if Fasta is None:
        raise RuntimeError("pyfaidx is required (conda activate caduceus_env)")
    return Fasta(str(path), as_raw=True, sequence_always_upper=True)


def chrom_lookup(fasta) -> dict[str, str]:
    keys = list(fasta.keys())
    lut = {k: k for k in keys}
    for k in keys:
        lut.setdefault(k.split()[0], k)
        if "." in k:
            lut.setdefault(k.split(".")[0], k)
    return lut


def revcomp(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]


def iter_genes(path: Path) -> Iterator[dict[str, str]]:
    with path.open(encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for rec in reader:
            yield rec


def extract_window_fa(
    fasta,
    chrom_key: str,
    gene_start: int,
    gene_end: int,
    flank: int,
    *,
    rc: bool = False,
) -> tuple[str, int, int] | None:
    """1-based inclusive gene coords → flanked window via pyfaidx slice."""
    if gene_start <= 0 or gene_end <= 0 or gene_end < gene_start:
        return None
    win_start = gene_start - flank
    win_end = gene_end + flank
    if win_start < 1:
        return None
    chrom_len = len(fasta[chrom_key])
    if win_end > chrom_len:
        return None
    # pyfaidx uses 0-based half-open slicing
    seq = str(fasta[chrom_key][win_start - 1 : win_end]).upper()
    if not seq or len(seq) != (win_end - win_start + 1):
        return None
    seq = re.sub(r"[^ACGTN]", "N", seq)
    if rc:
        seq = revcomp(seq)
    return seq, win_start, win_end


def process_unit(
    unit: GenomeUnit,
    *,
    window_size: int,
    flank: int,
    rc_export: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    samples: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    counts: dict[str, int] = Counter()
    max_gene_len = window_size - 2 * flank
    if max_gene_len <= 0:
        raise ValueError("window_size must be > 2*flank")

    tpm = load_tpm(unit.tpm)
    fasta = open_fasta(unit.fasta)
    lut = chrom_lookup(fasta)
    seen_genes: set[str] = set()

    for g in iter_genes(unit.genes):
        counts["genes_total"] += 1
        gene_id = (g.get("gene_id") or g.get("gene_name") or "").strip()
        gene_name = (g.get("gene_name") or gene_id).strip()
        if not gene_id:
            excluded.append(
                {
                    "genome": unit.genome_id,
                    "fold": unit.fold or "",
                    "gene_id": "",
                    "length": "",
                    "reason": "missing_gene_id",
                }
            )
            counts["rejected"] += 1
            continue
        if gene_id in seen_genes:
            excluded.append(
                {
                    "genome": unit.genome_id,
                    "fold": unit.fold or "",
                    "gene_id": gene_id,
                    "length": "",
                    "reason": "duplicate_gene_id",
                }
            )
            counts["rejected"] += 1
            continue
        seen_genes.add(gene_id)

        strand = (g.get("strand") or "+").strip()
        if strand not in {"+", "-", "."}:
            excluded.append(
                {
                    "genome": unit.genome_id,
                    "fold": unit.fold or "",
                    "gene_id": gene_id,
                    "length": "",
                    "reason": "invalid_strand",
                }
            )
            counts["rejected"] += 1
            continue
        if strand == ".":
            strand = "+"

        try:
            tss = int(g["TSS"])
            tes = int(g["TES"])
        except (KeyError, ValueError, TypeError):
            excluded.append(
                {
                    "genome": unit.genome_id,
                    "fold": unit.fold or "",
                    "gene_id": gene_id,
                    "length": "",
                    "reason": "invalid_coordinates",
                }
            )
            counts["rejected"] += 1
            continue

        gene_start = min(tss, tes)
        gene_end = max(tss, tes)
        gene_len = gene_end - gene_start + 1
        if gene_len > max_gene_len:
            excluded.append(
                {
                    "genome": unit.genome_id,
                    "fold": unit.fold or "",
                    "gene_id": gene_id,
                    "length": gene_len,
                    "reason": "gene_too_long",
                }
            )
            counts["rejected"] += 1
            counts["rejected_too_long"] += 1
            continue

        tpm_val = tpm.get(gene_name)
        if tpm_val is None:
            tpm_val = tpm.get(gene_id)
        if tpm_val is None:
            excluded.append(
                {
                    "genome": unit.genome_id,
                    "fold": unit.fold or "",
                    "gene_id": gene_id,
                    "length": gene_len,
                    "reason": "missing_tpm",
                }
            )
            counts["rejected"] += 1
            continue

        chrom = (g.get("chromosome") or g.get("chrom") or "").strip()
        key = lut.get(chrom) or lut.get(chrom.split(".")[0]) if chrom else None
        if key is None:
            excluded.append(
                {
                    "genome": unit.genome_id,
                    "fold": unit.fold or "",
                    "gene_id": gene_id,
                    "length": gene_len,
                    "reason": "chrom_not_in_fasta",
                }
            )
            counts["rejected"] += 1
            continue

        got = extract_window_fa(fasta, key, gene_start, gene_end, flank, rc=False)
        if got is None:
            excluded.append(
                {
                    "genome": unit.genome_id,
                    "fold": unit.fold or "",
                    "gene_id": gene_id,
                    "length": gene_len,
                    "reason": "window_out_of_bounds",
                }
            )
            counts["rejected"] += 1
            continue
        seq, win_start, win_end = got
        if not seq or set(seq) <= {"N"}:
            excluded.append(
                {
                    "genome": unit.genome_id,
                    "fold": unit.fold or "",
                    "gene_id": gene_id,
                    "length": gene_len,
                    "reason": "empty_or_invalid_sequence",
                }
            )
            counts["rejected"] += 1
            continue

        sample_id = sanitize(f"{unit.genome_id}__{gene_id}")
        row = {
            "sample_id": sample_id,
            "genome": unit.genome_id,
            "fold": unit.fold or "",
            "species": unit.species or "",
            "chromosome": chrom,
            "gene_id": gene_id,
            "gene_name": gene_name,
            "gene_start": gene_start,
            "gene_end": gene_end,
            "window_start": win_start,
            "window_end": win_end,
            "strand": strand,
            "sequence": seq,
            "TPM": float(tpm_val),
            "window_length": len(seq),
            "gene_length": gene_len,
            "orientation": "forward",
            "flank_bp": flank,
        }
        samples.append(row)
        counts["accepted"] += 1

        if rc_export:
            rc_seq = revcomp(seq)
            rc_id = sanitize(f"{sample_id}__rc")
            rc_row = dict(row)
            rc_row.update(
                {
                    "sample_id": rc_id,
                    "sequence": rc_seq,
                    "orientation": "reverse_complement",
                }
            )
            samples.append(rc_row)
            counts["accepted_rc"] += 1

    fasta.close()
    return samples, excluded, dict(counts)


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore", delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_caduceus_ready(out: Path, samples: list[dict[str, Any]]) -> None:
    """Fold-preserving sequence files + continuous labels for /caduceus."""
    root = out / "caduceus_ready"
    if root.exists():
        shutil.rmtree(root)
    by_fold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        fold = s.get("fold") or "all"
        by_fold[fold].append(s)
    for fold, rows in by_fold.items():
        seq_dir = root / fold / "sequences"
        seq_dir.mkdir(parents=True, exist_ok=True)
        labels = []
        for r in rows:
            fn = seq_dir / f"{r['sample_id']}.txt"
            fn.write_text(r["sequence"] + "\n", encoding="utf-8")
            labels.append(
                {
                    "sample_id": r["sample_id"],
                    "path": str(fn.relative_to(root)),
                    "TPM": r["TPM"],
                    "genome": r["genome"],
                    "gene_id": r["gene_id"],
                    "window_length": r["window_length"],
                    "strand": r["strand"],
                }
            )
        write_tsv(
            root / fold / "labels.tsv",
            labels,
            ["sample_id", "path", "TPM", "genome", "gene_id", "window_length", "strand"],
        )
    (root / "README.md").write_text(
        "\n".join(
            [
                "# Caduceus-ready adapt export",
                "",
                "Continuous TPM regression labels (not GenomicBenchmarks high/low classes).",
                "",
                "Layout:",
                "",
                "```",
                "caduceus_ready/{fold}/sequences/<sample_id>.txt",
                "caduceus_ready/{fold}/labels.tsv",
                "```",
                "",
                "`fold` is preserved from `@split` when present; otherwise `all`.",
                "",
                "Each `.txt` is a raw DNA string (A/C/G/T/N), length = gene + 2×flank.",
                "Tokenizer: Caduceus character-level (1 bp = 1 token); pad/truncate in the trainer.",
                "",
                "See docs/caduceus_format.md.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_qc_report(
    path: Path,
    audit: AuditResult,
    stats: dict[str, Any],
    excluded: list[dict[str, Any]],
) -> None:
    reasons = Counter(r["reason"] for r in excluded)
    lines = [
        "# Adapt QC report",
        "",
        f"- Mode: `{audit.mode}`",
        f"- Genomes processed: {len(audit.units)}",
        f"- Accepted windows: {stats.get('accepted_total', 0)}",
        f"- Rejected genes: {stats.get('rejected_total', 0)}",
        "",
        "## Audit notes",
        "",
    ]
    for n in audit.notes:
        lines.append(f"- {n}")
    if audit.missing:
        lines += ["", "## Missing (non-fatal per-genome unless empty run)", ""]
        for m in audit.missing:
            lines.append(f"- {m}")
    lines += ["", "## Exclusion reasons", ""]
    for reason, n in sorted(reasons.items(), key=lambda x: -x[1]):
        lines.append(f"- `{reason}`: {n}")
    lines += ["", "## Per-genome statistics", ""]
    for gid, st in stats.get("per_genome", {}).items():
        acc = st.get("accepted", 0)
        rej = st.get("rejected", 0)
        tot = acc + rej
        pct = (100.0 * rej / tot) if tot else 0.0
        lines.append(f"- `{gid}`: accepted={acc} rejected={rej} reject%={pct:.1f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset_readme(path: Path, cfg: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# Adapt dataset",
                "",
                "Caduceus-ready DNA windows for continuous TPM prediction.",
                "",
                f"- Window strategy: `{cfg.get('window_strategy')}`",
                f"- Flank: {cfg.get('flank_bp')} bp upstream + downstream",
                f"- window_size (max): {cfg.get('window_size')}",
                f"- Long genes: reject if length > window_size - 2×flank",
                f"- Orientation default: {cfg.get('orientation_default')}",
                f"- Target: continuous {cfg.get('target')}",
                "",
                "This dataset does **not** define train/val/test splits.",
                "Fold columns are copied from an upstream `@split` when present.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_method_decisions(path: Path, cfg: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# METHOD_DECISIONS (adapt run)",
                "",
                "### Window strategy (LOCKED baseline v1)",
                "",
                "- **Decision:** One window = one gene = 200 bp upstream + gene body + 200 bp downstream; reject genes longer than `window_size - 400`; no chunking / overlap / partial genes; DNA only; continuous TPM; forward orientation by default.",
                f"- **window_size:** {cfg.get('window_size')}",
                f"- **flank_bp:** {cfg.get('flank_bp')}",
                f"- **rc_export:** {cfg.get('rc_export')}",
                "- **Status:** Locked (user adapt skill baseline)",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--input", default="auto", help="auto | split/raw path")
    ap.add_argument("--out", type=Path, default=Path("adapt"))
    ap.add_argument("--window-size", type=int, default=None)
    ap.add_argument("--flank", type=int, default=None)
    ap.add_argument("--rc-export", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--max-units",
        type=int,
        default=None,
        help="Optional cap on genome units (smoke tests only)",
    )
    args = ap.parse_args(argv)

    root = args.root.resolve()
    cfg = load_config(args.config if args.config.is_absolute() else root / args.config)
    window_size = int(args.window_size or cfg.get("window_size", 8192))
    flank = int(args.flank or cfg.get("flank_bp", 200))
    rc_export = bool(args.rc_export or cfg.get("rc_export", False))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    cfg = {
        **cfg,
        "window_size": window_size,
        "flank_bp": flank,
        "rc_export": rc_export,
        "seed": seed,
    }

    if window_size <= 2 * flank:
        print(
            f"ERROR: window_size ({window_size}) must be > 2*flank ({2 * flank})",
            file=sys.stderr,
        )
        return 2

    audit = audit_repo(root, cfg, args.input)
    if not audit.units:
        print("ERROR: no usable genome units found.", file=sys.stderr)
        for m in audit.missing:
            print(f"  missing: {m}", file=sys.stderr)
        for n in audit.notes:
            print(f"  note: {n}", file=sys.stderr)
        return 2

    if args.max_units is not None:
        audit.units = audit.units[: max(0, args.max_units)]
        audit.notes.append(f"max_units={args.max_units} applied")

    out = args.out if args.out.is_absolute() else root / args.out
    if out.exists():
        # idempotent restart: replace tree
        shutil.rmtree(out)
    out.mkdir(parents=True)

    all_samples: list[dict[str, Any]] = []
    all_excluded: list[dict[str, Any]] = []
    per_genome: dict[str, Any] = {}
    manifest_rows: list[dict[str, Any]] = []

    for unit in audit.units:
        samples, excluded, counts = process_unit(
            unit, window_size=window_size, flank=flank, rc_export=rc_export
        )
        all_samples.extend(samples)
        all_excluded.extend(excluded)
        key = f"{unit.fold + ':' if unit.fold else ''}{unit.genome_id}"
        per_genome[key] = counts
        manifest_rows.append(
            {
                "genome": unit.genome_id,
                "fold": unit.fold or "",
                "species": unit.species or "",
                "fasta": str(unit.fasta),
                "genes": str(unit.genes),
                "tpm": str(unit.tpm),
                "gtf": str(unit.gtf) if unit.gtf else "",
                "accepted": counts.get("accepted", 0),
                "rejected": counts.get("rejected", 0),
                "source": unit.source,
            }
        )
        print(
            f"{key}: accepted={counts.get('accepted', 0)} "
            f"rejected={counts.get('rejected', 0)} "
            f"too_long={counts.get('rejected_too_long', 0)}"
        )

    if not all_samples:
        print("ERROR: zero accepted windows — aborting.", file=sys.stderr)
        return 2

    sample_fields = [
        "sample_id",
        "genome",
        "fold",
        "species",
        "chromosome",
        "gene_id",
        "gene_name",
        "gene_start",
        "gene_end",
        "window_start",
        "window_end",
        "strand",
        "sequence",
        "TPM",
        "window_length",
        "gene_length",
        "orientation",
        "flank_bp",
    ]
    write_tsv(out / "samples.tsv", all_samples, sample_fields)
    write_tsv(
        out / "labels.tsv",
        [{"sample_id": s["sample_id"], "TPM": s["TPM"]} for s in all_samples],
        ["sample_id", "TPM"],
    )
    write_tsv(
        out / "excluded_genes.tsv",
        all_excluded,
        ["genome", "fold", "gene_id", "length", "reason"],
    )
    write_tsv(
        out / "manifest.tsv",
        manifest_rows,
        [
            "genome",
            "fold",
            "species",
            "fasta",
            "genes",
            "tpm",
            "gtf",
            "accepted",
            "rejected",
            "source",
        ],
    )

    stats = {
        "accepted_total": sum(1 for s in all_samples if s.get("orientation") == "forward"),
        "accepted_with_rc": len(all_samples),
        "rejected_total": len(all_excluded),
        "n_genomes": len(audit.units),
        "mode": audit.mode,
        "window_size": window_size,
        "flank_bp": flank,
        "per_genome": per_genome,
    }
    (out / "statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tool": "adapt",
        "script": str(SCRIPT_DIR / "adapt.py"),
        "seed": seed,
        "input": args.input,
        "mode": audit.mode,
        "assumptions": cfg.get("assumptions_documented", []),
        "software": {
            "python": sys.version.split()[0],
            "pyfaidx": "present" if Fasta is not None else "missing",
        },
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if yaml is not None:
        (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    else:
        (out / "config.yaml").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    write_caduceus_ready(out, all_samples)
    write_qc_report(out / "qc_report.md", audit, stats, all_excluded)
    write_dataset_readme(out / "README.md", cfg)
    write_method_decisions(out / "METHOD_DECISIONS.md", cfg)

    print(f"Wrote {out} ({stats['accepted_total']} accepted, {stats['rejected_total']} rejected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
