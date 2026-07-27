#!/usr/bin/env python3
"""Materialize Caduceus GB .txt windows labeled high/low expression (TPM).

Fold layout (Caduceus GenomicBenchmark-compatible):
  {out}/{train|val|test}/{high|low}/*.txt

Uses data_splits/full fold dirs (genome.fna, genes.tsv, expression_tpm.csv).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

from pyfaidx import Fasta


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--manifest", type=Path, default=Path("data_splits/full/fold_manifest.tsv"))
    p.add_argument("--out", type=Path, default=Path("data/caduceus_gb/random_full"))
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--max-per-species", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def chrom_lookup(fasta: Fasta) -> dict[str, str]:
    keys = list(fasta.keys())
    lut = {k: k for k in keys}
    for k in keys:
        if "." in k:
            lut.setdefault(k.split(".")[0], k)
    return lut


def extract_window(seq: str, tss: int, tes: int, strand: str, seq_len: int) -> str | None:
    if tss <= 0 or tes <= 0:
        return None
    if strand == "-":
        win_end = max(tss, tes)
        win_start = max(0, win_end - seq_len)
    else:
        win_start = min(tss, tes) - 1
        win_end = win_start + seq_len
    if win_start < 0:
        win_start = 0
        win_end = seq_len
    chunk = seq[win_start:win_end]
    if len(chunk) < seq_len:
        return None
    chunk = chunk[:seq_len].upper()
    if strand == "-":
        comp = str.maketrans("ACGTN", "TGCAN")
        chunk = chunk.translate(comp)[::-1]
    chunk = re.sub(r"[^ACGTN]", "N", chunk)
    return chunk


def load_tpm(path: Path) -> dict[str, float]:
    with path.open() as fh:
        header = fh.readline().rstrip("\n").split(",")
        values = fh.readline().rstrip("\n").split(",")
    out = {}
    for g, v in zip(header, values):
        try:
            out[g] = float(v)
        except ValueError:
            continue
    return out


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out_root = root / args.out
    rng = random.Random(args.seed)
    rows = list(csv.DictReader((root / args.manifest).open(), delimiter="\t"))

    # reset out tree completely (avoid stale label dirs from prior runs)
    if out_root.exists():
        import shutil

        shutil.rmtree(out_root)

    summary = {
        "seed": args.seed,
        "seq_len": args.seq_len,
        "max_per_species": args.max_per_species,
        "labeling": "tpm_median_high_low",
        "folds": {},
    }

    for r in rows:
        fold = r["fold"]
        sample_dir = root / r["fold_dir"]
        fna = sample_dir / "genome.fna"
        genes_path = sample_dir / "genes.tsv"
        tpm_path = sample_dir / "expression_tpm.csv"
        fasta = Fasta(str(fna), as_raw=True, sequence_always_upper=True)
        lut = chrom_lookup(fasta)
        tpm = load_tpm(tpm_path)

        joined = []
        with genes_path.open() as fh:
            for g in csv.DictReader(fh, delimiter="\t"):
                name = g.get("gene_name") or ""
                gid = g.get("gene_id") or ""
                tpm_val = tpm.get(name)
                if tpm_val is None:
                    tpm_val = tpm.get(gid)
                if tpm_val is None:
                    continue
                try:
                    tss = int(g["TSS"])
                    tes = int(g["TES"])
                except (KeyError, ValueError):
                    continue
                chrom = g["chromosome"]
                if chrom not in lut and chrom.split(".")[0] not in lut:
                    continue
                joined.append((gid or name, name, tss, tes, g.get("strand", "+"), chrom, tpm_val))

        if len(joined) < 4:
            raise RuntimeError(f"Too few gene↔TPM joins for {r['sample_id']}: {len(joined)}")

        rng.shuffle(joined)
        selected = joined[: args.max_per_species]
        median = sorted(v for *_, v in selected)[len(selected) // 2]

        counts = {"high": 0, "low": 0}
        for gene_id, gene_name, tss, tes, strand, chrom, tpm_val in selected:
            key = lut.get(chrom) or lut.get(chrom.split(".")[0])
            seq = str(fasta[key])
            window = extract_window(seq, tss, tes, strand, args.seq_len)
            if window is None:
                continue
            label = "high" if tpm_val >= median else "low"
            dest = out_root / fold / label
            dest.mkdir(parents=True, exist_ok=True)
            safe = sanitize(f"{r['sample_id']}__{gene_id or gene_name}")[:100]
            (dest / f"{safe}.txt").write_text(window)
            counts[label] += 1

        fasta.close()
        summary["folds"][f"{fold}:{r['sample_id']}"] = {
            "species": r["species"],
            "median_tpm": median,
            "n_joined": len(joined),
            **counts,
        }
        print(f"{fold}/{r['sample_id']}: high={counts['high']} low={counts['low']} median={median:.4g}")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "materialize_summary.json").write_text(json.dumps(summary, indent=2))
    print("Wrote", out_root / "materialize_summary.json")


if __name__ == "__main__":
    main()
