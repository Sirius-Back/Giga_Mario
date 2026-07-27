#!/usr/bin/env python3
"""Acquire one TPM CSV per genomes_smoketest assembly from public GEO matrices.

Outputs wide CSVs (header=gene_id, one TPM row) matching project expression_tpm.csv
convention. Does not invent expression values; remaps same-species matrices by gene
name only when documented.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENOME_DIR = ROOT / "genomes_smoketest"
OUT_DIR = GENOME_DIR
RAW_DIR = ROOT / "data" / "raw" / "expression" / "smoketest"
META_DIR = ROOT / "data" / "metadata" / "smoketest_tpm"
LOG = ROOT / "data" / "logs" / "download.log"
MANIFEST = ROOT / "data" / "manifests" / "smoketest_tpm_download_manifest.tsv"
CHECKSUMS = ROOT / "data" / "checksums" / "smoketest_tpm_checksums.txt"
CACHE = ROOT / "data" / "raw" / "expression" / "smoketest" / "_sources"

UA = "Mozilla/5.0 (compatible; User14-smoketest-tpm/1.0)"
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{NOW}] {msg}"
    with LOG.open("a") as fh:
        fh.write(line + "\n")
    print(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log(f"SKIP download (exists): {dest}")
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    log(f"CMD: download {url} -> {dest}")
    with urllib.request.urlopen(req, timeout=180) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)
    return dest


def read_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw)
    return raw


def parse_gtf_genes(gtf: Path) -> dict[str, dict]:
    """gene_id -> {gene, length, start, end}"""
    out: dict[str, dict] = {}
    with gtf.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            attrs = dict(re.findall(r'(\w+) "([^"]*)"', parts[8]))
            gid = attrs.get("gene_id")
            if not gid:
                continue
            gn = attrs.get("gene") or attrs.get("gene_name") or ""
            start, end = int(parts[3]), int(parts[4])
            out[gid] = {
                "gene": gn,
                "length": end - start + 1,
                "start": start,
                "end": end,
            }
    return out


def name_to_unique_ids(genes: dict[str, dict]) -> dict[str, str]:
    """Map gene name -> gene_id only when the name is unique."""
    buckets: dict[str, list[str]] = {}
    for gid, info in genes.items():
        gn = info["gene"]
        if not gn:
            continue
        buckets.setdefault(gn, []).append(gid)
    return {gn: ids[0] for gn, ids in buckets.items() if len(ids) == 1}


def counts_to_tpm(counts: dict[str, float], lengths: dict[str, int]) -> dict[str, float]:
    rpk = {}
    for gid, c in counts.items():
        L = lengths.get(gid)
        if not L or L <= 0:
            continue
        rpk[gid] = c / (L / 1000.0)
    s = sum(rpk.values())
    if s <= 0:
        return {k: 0.0 for k in rpk}
    return {k: v / s * 1e6 for k, v in rpk.items()}


def rpkm_to_tpm(rpkm: dict[str, float]) -> dict[str, float]:
    s = sum(rpkm.values())
    if s <= 0:
        return {k: 0.0 for k in rpkm}
    return {k: v / s * 1e6 for k, v in rpkm.items()}


def renormalize_tpm(tpm: dict[str, float]) -> dict[str, float]:
    """Rescale a gene subset so values sum to 1e6 (TPM units after filtering)."""
    s = sum(tpm.values())
    if s <= 0:
        return tpm
    return {k: v / s * 1e6 for k, v in tpm.items()}


def write_wide_csv(path: Path, tpm: dict[str, float]) -> None:
    tpm = renormalize_tpm(tpm)
    keys = sorted(tpm.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        w.writerow([f"{tpm[k]:.6f}" for k in keys])


def remap_by_gene_name(
    source_tpm_by_id: dict[str, float],
    source_genes: dict[str, dict],
    target_genes: dict[str, dict],
) -> dict[str, float]:
    src_name = {}
    for gid, val in source_tpm_by_id.items():
        gn = source_genes.get(gid, {}).get("gene")
        if gn and gn not in src_name:
            src_name[gn] = val
    tgt = name_to_unique_ids(target_genes)
    out = {}
    for gn, gid in tgt.items():
        if gn in src_name:
            out[gid] = src_name[gn]
    return out


def load_k12_tpm(path: Path) -> dict[str, float]:
    text = read_bytes(path).decode()
    # may already be uncompressed csv
    if path.suffix == ".csv" and path.exists():
        text = path.read_text()
    # handle gz-decompressed content saved without extension
    rows = list(csv.DictReader(text.splitlines()))
    # Prefer WT BOP27_R1; fall back to first numeric column after gene
    sample = "BOP27_R1"
    if sample not in rows[0]:
        sample = [c for c in rows[0] if c != "gene"][0]
    out = {}
    for row in rows:
        gid = row["gene"].strip('"')
        out[gid] = float(row[sample])
    return out, sample


def load_o157_wt_counts(path: Path) -> tuple[dict[str, float], dict[str, int], str]:
    text = read_bytes(path).decode()
    lines = [L for L in text.splitlines() if not L.startswith("#")]
    header = lines[0].split("\t")
    wt_cols = [i for i, c in enumerate(header) if re.search(r"EHEC_WT\d", c)]
    if not wt_cols:
        raise RuntimeError("No EHEC_WT columns in O157 matrix")
    attr_i = header.index("Attributes")
    start_i = header.index("Start")
    end_i = header.index("End")
    feat_i = header.index("Feature")
    counts: dict[str, list[float]] = {}
    lengths: dict[str, int] = {}
    for line in lines[1:]:
        parts = line.split("\t")
        if parts[feat_i] not in ("gene", "CDS"):
            continue
        attrs = parts[attr_i]
        m = re.search(r"locus_tag=([^;]+)", attrs)
        if not m:
            continue
        lt = m.group(1)
        # Prefer gene features; CDS fills gaps
        if lt in counts and parts[feat_i] == "CDS":
            continue
        vals = [float(parts[i]) for i in wt_cols]
        start, end = int(parts[start_i]), int(parts[end_i])
        counts[lt] = vals
        lengths[lt] = end - start + 1
    mean_counts = {k: sum(v) / len(v) for k, v in counts.items()}
    return mean_counts, lengths, "+".join(header[i] for i in wt_cols)


def load_shigella_culture_rpkm(path: Path) -> tuple[dict[str, float], dict[str, str], str]:
    """Return rpkm keyed by GEO GeneID, plus GeneID->Name map."""
    text = read_bytes(path).decode() if path.suffix == ".gz" or path.read_bytes()[:2] == b"\x1f\x8b" else path.read_text()
    # file may already be plain tsv
    if path.exists() and not path.name.endswith(".gz"):
        text = path.read_text()
    rows = list(csv.DictReader(text.splitlines(), delimiter="\t"))
    cols = [
        "RPKM_control_ssonnei_culture1",
        "RPKM_control_ssonnei_culture2",
        "RPKM_control_ssonnei_culture3",
    ]
    rpkm = {}
    id_to_name = {}
    for row in rows:
        gid = row["GeneID"]
        name = row.get("Name") or ""
        vals = [float(row[c]) for c in cols]
        rpkm[gid] = sum(vals) / len(vals)
        id_to_name[gid] = name
    return rpkm, id_to_name, "+".join(cols)


def load_pa_basemean_tpm(path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(path.open()))
    lengths = {}
    means = {}
    for row in rows:
        lt = row["locus_tag"]
        start, end = int(row["start"]), int(row["end"])
        lengths[lt] = end - start + 1
        means[lt] = float(row["baseMean"])
    return counts_to_tpm(means, lengths)


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    sources = {
        "GSE164236_tpm": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE164nnn/GSE164236/suppl/GSE164236_deseq2-tpm.csv.gz",
            CACHE / "GSE164236_deseq2-tpm.csv.gz",
        ),
        "GSE311113_o157": (
            "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE311113&format=file&file=GSE311113%5Fdeseq%5Fcomp%5FEHECdel%5Fvs%5FEHECwt%5Fwith%5Fannotation%5Fand%5Fcountings%5Ffull%2Ecsv%2Egz",
            CACHE / "GSE311113_EHEC_counts.csv.gz",
        ),
        "GSE140544_shigella": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE140nnn/GSE140544/suppl/GSE140544_ssonnei_processed_data.tsv.gz",
            CACHE / "GSE140544_ssonnei_processed_data.tsv.gz",
        ),
        "GSE291027_pa": (
            "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE291027&format=file&file=GSE291027%5Fp3565%2Eintcounts%2Etxt%5Fall%5Fannotation%2Ecsv%2Egz",
            CACHE / "GSE291027_p3565_annotation.csv.gz",
        ),
    }

    # Prefer local probe cache if present
    local_alts = {
        "GSE164236_tpm": Path("/tmp/geo_probe/GSE164236_deseq2-tpm.csv"),
        "GSE311113_o157": Path("/tmp/geo_probe/GSE311113_suppl"),
        "GSE140544_shigella": Path("/tmp/geo_probe/GSE140544_ssonnei_processed_data.tsv"),
        "GSE291027_pa": Path("/tmp/geo_probe/GSE291027_ann.csv"),
    }

    acquired = {}
    for key, (url, dest) in sources.items():
        alt = local_alts[key]
        if alt.exists() and alt.stat().st_size > 0:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if alt.suffix == ".gz" or alt.read_bytes()[:2] == b"\x1f\x8b":
                shutil.copy2(alt, dest)
            else:
                # store uncompressed copy with clear name
                plain = dest.with_suffix("") if dest.suffix == ".gz" else dest
                if key == "GSE164236_tpm":
                    plain = CACHE / "GSE164236_deseq2-tpm.csv"
                elif key == "GSE311113_o157":
                    plain = CACHE / "GSE311113_EHEC_counts.csv"
                elif key == "GSE140544_shigella":
                    plain = CACHE / "GSE140544_ssonnei_processed_data.tsv"
                elif key == "GSE291027_pa":
                    plain = CACHE / "GSE291027_p3565_annotation.csv"
                shutil.copy2(alt, plain)
                acquired[key] = {"path": plain, "url": f"local-cache:{alt}", "status": "success"}
                log(f"REUSE local {alt} -> {plain}")
                continue
        try:
            download(url, dest)
            acquired[key] = {"path": dest, "url": url, "status": "success"}
        except Exception as exc:  # noqa: BLE001
            # try local uncompressed
            if alt.exists():
                plain = CACHE / alt.name
                shutil.copy2(alt, plain)
                acquired[key] = {"path": plain, "url": f"local-cache:{alt}", "status": "success"}
                log(f"FALLBACK local after error ({exc}): {plain}")
            else:
                acquired[key] = {"path": dest, "url": url, "status": f"failed:{exc}"}
                log(f"FAIL {key}: {exc}")

    # Resolve usable paths
    def resolve(key: str, *candidates: str) -> Path:
        p = Path(acquired[key]["path"])
        if p.exists():
            return p
        for c in candidates:
            cp = CACHE / c
            if cp.exists():
                return cp
        raise FileNotFoundError(key)

    k12_path = resolve("GSE164236_tpm", "GSE164236_deseq2-tpm.csv", "GSE164236_deseq2-tpm.csv.gz")
    o157_path = resolve("GSE311113_o157", "GSE311113_EHEC_counts.csv", "GSE311113_EHEC_counts.csv.gz")
    ss_path = resolve(
        "GSE140544_shigella",
        "GSE140544_ssonnei_processed_data.tsv",
        "GSE140544_ssonnei_processed_data.tsv.gz",
    )
    pa_path = resolve(
        "GSE291027_pa",
        "GSE291027_p3565_annotation.csv",
        "GSE291027_p3565_annotation.csv.gz",
    )

    # Load genomes
    assemblies = {
        "Escherichia_coli_K12_ASM584v2": GENOME_DIR / "Escherichia_coli_K12_ASM584v2.gtf",
        "Escherichia_coli_K12_ASM2564343v1": GENOME_DIR / "Escherichia_coli_K12_ASM2564343v1.gtf",
        "Escherichia_coli_O157H7_ASM886v2": GENOME_DIR / "Escherichia_coli_O157H7_ASM886v2.gtf",
        "Escherichia_coli_O157H7_ASM169551v1": GENOME_DIR / "Escherichia_coli_O157H7_ASM169551v1.gtf",
        "Shigella_sonnei_ASM295039v1": GENOME_DIR / "Shigella_sonnei_ASM295039v1.gtf",
        "Shigella_sonnei_ASM360642v1": GENOME_DIR / "Shigella_sonnei_ASM360642v1.gtf",
        "Pseudomonas_aeruginosa_ASM676v1": GENOME_DIR / "Pseudomonas_aeruginosa_ASM676v1.gtf",
        "Pseudomonas_aeruginosa_NCTC10332": GENOME_DIR / "Pseudomonas_aeruginosa_NCTC10332.gtf",
    }
    genes = {k: parse_gtf_genes(v) for k, v in assemblies.items()}

    results = {}

    # --- K12 reference native TPM ---
    if k12_path.suffix == ".gz" or k12_path.read_bytes()[:2] == b"\x1f\x8b":
        text = read_bytes(k12_path).decode()
        tmp = CACHE / "GSE164236_deseq2-tpm.csv"
        tmp.write_text(text)
        k12_path = tmp
    k12_tpm, k12_sample = load_k12_tpm(k12_path)
    # keep only IDs present in ASM584v2
    k12_ref = {g: k12_tpm[g] for g in k12_tpm if g in genes["Escherichia_coli_K12_ASM584v2"]}
    results["Escherichia_coli_K12_ASM584v2"] = {
        "tpm": k12_ref,
        "source": "GSE164236",
        "sample": k12_sample,
        "method": "native_TPM",
        "role": "reference",
    }

    # K12 partner remap by gene name
    rem = remap_by_gene_name(
        k12_ref, genes["Escherichia_coli_K12_ASM584v2"], genes["Escherichia_coli_K12_ASM2564343v1"]
    )
    results["Escherichia_coli_K12_ASM2564343v1"] = {
        "tpm": rem,
        "source": "GSE164236",
        "sample": k12_sample,
        "method": "gene_name_remap_from_ASM584v2",
        "role": "partner",
    }

    # --- O157 reference: WT counts -> TPM ---
    if o157_path.suffix != ".gz" and o157_path.read_bytes()[:2] != b"\x1f\x8b":
        # plain csv already
        pass
    mean_counts, lengths_mat, o157_sample = load_o157_wt_counts(o157_path)
    # prefer GTF lengths when available
    gtf_len = {g: info["length"] for g, info in genes["Escherichia_coli_O157H7_ASM886v2"].items()}
    lengths = {g: gtf_len.get(g, lengths_mat[g]) for g in mean_counts}
    o157_tpm = counts_to_tpm(mean_counts, lengths)
    o157_ref = {g: o157_tpm[g] for g in o157_tpm if g in genes["Escherichia_coli_O157H7_ASM886v2"]}
    results["Escherichia_coli_O157H7_ASM886v2"] = {
        "tpm": o157_ref,
        "source": "GSE311113",
        "sample": o157_sample,
        "method": "counts_to_TPM",
        "role": "reference",
    }

    rem = remap_by_gene_name(
        o157_ref,
        genes["Escherichia_coli_O157H7_ASM886v2"],
        genes["Escherichia_coli_O157H7_ASM169551v1"],
    )
    results["Escherichia_coli_O157H7_ASM169551v1"] = {
        "tpm": rem,
        "source": "GSE311113",
        "sample": o157_sample,
        "method": "gene_name_remap_from_ASM886v2",
        "role": "partner",
    }

    # --- Shigella: culture RPKM -> TPM, map by gene name to each assembly ---
    if ss_path.suffix == ".gz" or ss_path.read_bytes()[:2] == b"\x1f\x8b":
        text = read_bytes(ss_path).decode()
        tmp = CACHE / "GSE140544_ssonnei_processed_data.tsv"
        tmp.write_text(text)
        ss_path = tmp
    ss_rpkm, id_to_name, ss_sample = load_shigella_culture_rpkm(ss_path)
    # key by gene name (unique names only in GEO table)
    name_rpkm: dict[str, float] = {}
    for gid, val in ss_rpkm.items():
        name = id_to_name.get(gid, "")
        if name and name not in name_rpkm:
            name_rpkm[name] = val
    name_tpm = rpkm_to_tpm(name_rpkm)

    def shigella_to_assembly(asm: str, role: str) -> dict:
        tgt = name_to_unique_ids(genes[asm])
        out = {gid: name_tpm[gn] for gn, gid in tgt.items() if gn in name_tpm}
        return {
            "tpm": out,
            "source": "GSE140544",
            "sample": ss_sample,
            "method": "RPKM_to_TPM_gene_name_map",
            "role": role,
        }

    results["Shigella_sonnei_ASM295039v1"] = shigella_to_assembly(
        "Shigella_sonnei_ASM295039v1", "reference"
    )
    results["Shigella_sonnei_ASM360642v1"] = shigella_to_assembly(
        "Shigella_sonnei_ASM360642v1", "partner"
    )

    # --- PAO1: baseMean length-normalized to TPM units ---
    if pa_path.suffix == ".gz" or pa_path.read_bytes()[:2] == b"\x1f\x8b":
        text = read_bytes(pa_path).decode()
        tmp = CACHE / "GSE291027_p3565_annotation.csv"
        tmp.write_text(text)
        pa_path = tmp
    pa_tpm = load_pa_basemean_tpm(pa_path)
    pa_ref = {g: pa_tpm[g] for g in pa_tpm if g in genes["Pseudomonas_aeruginosa_ASM676v1"]}
    results["Pseudomonas_aeruginosa_ASM676v1"] = {
        "tpm": pa_ref,
        "source": "GSE291027",
        "sample": "baseMean_all_contrasts_table",
        "method": "baseMean_length_normalized_TPM_units",
        "role": "reference",
    }
    rem = remap_by_gene_name(
        pa_ref,
        genes["Pseudomonas_aeruginosa_ASM676v1"],
        genes["Pseudomonas_aeruginosa_NCTC10332"],
    )
    results["Pseudomonas_aeruginosa_NCTC10332"] = {
        "tpm": rem,
        "source": "GSE291027",
        "sample": "baseMean_all_contrasts_table",
        "method": "gene_name_remap_from_ASM676v1",
        "role": "partner",
    }

    # Write outputs + manifest rows
    manifest_rows = []
    summary = []
    for asm, info in results.items():
        tpm = info["tpm"]
        if len(tpm) < 100:
            raise RuntimeError(f"Too few TPM genes for {asm}: {len(tpm)}")
        out_csv = OUT_DIR / f"{asm}.csv"
        raw_csv = RAW_DIR / f"{asm}.csv"
        write_wide_csv(out_csv, tpm)
        write_wide_csv(raw_csv, tpm)
        digest = sha256_file(out_csv)
        n_genes = len(genes[asm])
        cov = len(tpm) / n_genes if n_genes else 0
        summary.append(
            {
                "assembly": asm,
                "role": info["role"],
                "source": info["source"],
                "sample": info["sample"],
                "method": info["method"],
                "n_tpm_genes": len(tpm),
                "n_gtf_genes": n_genes,
                "coverage": round(cov, 4),
                "csv": str(out_csv.relative_to(ROOT)),
                "sha256": digest,
            }
        )
        manifest_rows.append(
            {
                "accession": info["source"],
                "repository": "NCBI_GEO",
                "download_url": acquired[
                    {
                        "GSE164236": "GSE164236_tpm",
                        "GSE311113": "GSE311113_o157",
                        "GSE140544": "GSE140544_shigella",
                        "GSE291027": "GSE291027_pa",
                    }[info["source"]]
                ]["url"],
                "local_path": str(out_csv.relative_to(ROOT)),
                "filename_bytes": out_csv.stat().st_size,
                "sha256": digest,
                "download_status": "success",
                "validation_status": "pass" if len(tpm) >= 500 else "warn",
                "biological_status": "pass" if info["role"] == "reference" and cov >= 0.5 else "warn",
                "notes": f"{asm};{info['method']};sample={info['sample']};genes={len(tpm)}/{n_genes}",
                "download_date": DATE,
            }
        )
        log(f"WROTE {out_csv} genes={len(tpm)} method={info['method']}")

    # Manifest
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "accession",
        "repository",
        "download_url",
        "local_path",
        "filename_bytes",
        "sha256",
        "download_status",
        "validation_status",
        "biological_status",
        "notes",
        "download_date",
    ]
    with MANIFEST.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for row in manifest_rows:
            w.writerow(row)

    CHECKSUMS.parent.mkdir(parents=True, exist_ok=True)
    with CHECKSUMS.open("w") as fh:
        fh.write(f"# SHA256 checksums — generated {DATE}\n")
        for row in summary:
            fh.write(f"{row['sha256']}  {row['csv']}\n")

    meta_path = META_DIR / "tpm_selection.json"
    meta_path.write_text(json.dumps({"date": DATE, "assemblies": summary}, indent=2))
    log(f"WROTE summary {meta_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
