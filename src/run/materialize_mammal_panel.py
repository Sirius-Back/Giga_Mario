#!/usr/bin/env python3
"""Materialize EquCab3.0 + goat FNA/GTF into raw/ and build matching TPM CSVs."""
from __future__ import annotations

import gzip
import hashlib
import shutil
from pathlib import Path

from src.geo_expr_to_tpm import fpkm_column_to_symbol_tpm, horse_counts_to_symbol_tpm
from src.summarize_geo import write_wide_tpm

ROOT = Path(__file__).resolve().parents[2]
DL = ROOT / "data" / "raw" / "downloads"
RAW = ROOT / "raw"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_md5(path: Path, checksums: Path) -> None:
    expected = None
    name = path.name
    for line in checksums.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-1].endswith(name):
            expected = parts[0]
            break
    if expected is None:
        raise ValueError(f"No md5 entry for {name} in {checksums}")
    got = _md5(path)
    if got != expected:
        raise ValueError(f"MD5 mismatch for {path}: got {got}, expected {expected}")


def _gunzip_to(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src, "rb") as zin, dst.open("wb") as zout:
        shutil.copyfileobj(zin, zout)
    if dst.stat().st_size == 0:
        raise ValueError(f"Decompressed empty file: {dst}")


def materialize() -> None:
    equ_fna_gz = DL / "EquCab3.0" / "GCF_002863925.1_EquCab3.0_genomic.fna.gz"
    equ_gtf_gz = DL / "EquCab3.0" / "GCF_002863925.1_EquCab3.0_genomic.gtf.gz"
    goat_fna_gz = DL / "ARS1.2" / "GCF_001704415.2_ARS1.2_genomic.fna.gz"
    goat_gtf_gz = DL / "ARS1.2" / "GCF_001704415.2_ARS1.2_genomic.gtf.gz"
    equ_md5 = DL / "EquCab3.0" / "md5checksums.txt"
    goat_md5 = DL / "ARS1.2" / "md5checksums.txt"

    for path in (equ_fna_gz, equ_gtf_gz, goat_fna_gz, goat_gtf_gz, equ_md5, goat_md5):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Download missing: {path}")

    _verify = True
    try:
        _verify_md5 = True
        # inline md5 check
        import hashlib

        def md5(path: Path) -> str:
            h = hashlib.md5()
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()

        for path, checksums in (
            (equ_fna_gz, equ_md5),
            (equ_gtf_gz, equ_md5),
            (goat_fna_gz, goat_md5),
            (goat_gtf_gz, goat_md5),
        ):
            expected = None
            for line in checksums.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[-1].endswith(path.name):
                    expected = parts[0]
                    break
            if expected is None:
                raise ValueError(f"No md5 for {path.name}")
            got = md5(path)
            if got != expected:
                raise ValueError(f"MD5 mismatch {path.name}: {got} != {expected}")
            print(f"MD5 OK {path.name}")
    except Exception:
        raise

    equ_fna = RAW_FNA = Path("raw/fna") / "GCF_002863925.1_EquCab3.0_genomic.fna"
    equ_gtf = Path("raw/gtf") / "GCF_002863925.1_EquCab3.0_genomic.gtf"
    goat_fna = Path("raw/fna") / "GCF_001704415.2_ARS1.2_genomic.fna"
    goat_gtf = Path("raw/gtf") / "GCF_001704415.2_ARS1.2_genomic.gtf"

    for src, dst in [
        (DL / "EquCab3.0" / "GCF_002863925.1_EquCab3.0_genomic.fna.gz", equ_fna),
        (DL / "EquCab3.0" / "GCF_002863925.1_EquCab3.0_genomic.gtf.gz", equ_gtf),
        (DL / "ARS1.2" / "GCF_001704415.2_ARS1.2_genomic.fna.gz", goat_fna),
        (DL / "ARS1.2" / "GCF_001704415.2_ARS1.2_genomic.gtf.gz", goat_gtf),
    ]:
        if dst.exists() and dst.stat().st_size > 0:
            print(f"skip existing {dst}")
            continue
        print(f"decompress {src.name} -> {dst}")
        with gzip.open(src, "rb") as zin, dst.open("wb") as zout:
            while True:
                chunk = zin.read(1 << 20)
                if not chunk:
                    break
                zout.write(chunk)
        if dst.stat().st_size == 0:
            raise ValueError(f"Empty decompress: {dst}")

    horse_out = Path("raw/tpm/SRX19584896.csv")
    goat_out = Path("raw/tpm/SRX6696967.csv")
    tpm_h = horse_counts_to_symbol_tpm(
        DL / "geo" / "GSM7084192_14count.txt.gz",
        DL / "EquCab3.0" / "Equus_caballus.EquCab3.0.112.gtf.gz",
        equ_gtf := Path("raw/gtf/GCF_002863925.1_EquCab3.0_genomic.gtf"),
    )
    write_wide_tpm(horse_out, tpm_h)
    print(f"horse TPM genes={len(tpm_h)} sum={sum(tpm_h.values()):.3f}")

    tpm_g = fpkm_column_to_symbol_tpm(
        DL / "geo" / "GSE135692_AllSamples.GeneExpression.FPKM.txt.gz",
        "Blank-1_FPKM",
        Path("raw/gtf/GCF_001704415.2_ARS1.2_genomic.gtf"),
    )
    write_wide_tpm(goat_out, tpm_g)
    print(f"goat TPM genes={len(tpm_g)} sum={sum(tpm_g.values()):.3f}")


if __name__ == "__main__":
    main_materialize = True
    # intentionally not calling main() CLI here; materialize entry below
    raise SystemExit("Use: python -m src.geo_expr_to_tpm horse|goat ...")
