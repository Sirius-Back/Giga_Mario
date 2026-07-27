#!/usr/bin/env python3
"""Acquire 3 genome-aligned TPM CSVs per prokaryote panel species from GEO.

Outputs wide CSVs (header=gene_id from GTF, one TPM data row) under
prokaryotes/tpm/, mirroring raw/tpm convention. Counts/RPKM are converted to
TPM using GTF gene lengths when native TPM is unavailable. Does not invent
expression values.
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
PROK = ROOT / "prokaryotes"
FNA_DIR = PROK / "fna"
GTF_DIR = PROK / "gtf"
TPM_DIR = PROK / "tpm"
CACHE = ROOT / "data" / "raw" / "prokaryotes" / "expression" / "_sources"
META = ROOT / "data" / "metadata" / "prokaryotes"
LOG = ROOT / "data" / "logs" / "download.log"
MANIFEST = ROOT / "data" / "manifests" / "prokaryotes_download_manifest.tsv"
CHECKSUMS = ROOT / "data" / "checksums" / "prokaryotes_checksums.txt"
MAPPINGS = PROK / "expr_file_mappings.csv"
SUMMARY = META / "tpm_alignment_summary.json"

UA = "Mozilla/5.0 (compatible; User14-prokaryotes-tpm/1.0)"
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")

# genome stem (GCF_…_ASM…) -> acquisition config
SPECIES = {
    "GCF_000005845.2_ASM584v2": {
        "species": "Escherichia coli K-12 MG1655",
        "gse": "GSE164236",
        "file": "GSE164236_deseq2-tpm.csv.gz",
        "kind": "tpm_csv",
        "id_col": "gene",
        # GEO matrix has BOP27_R1/R2 only; third is parental-background mutant replicate
        "samples": ["BOP27_R1", "BOP27_R2", "delta_cydB_R1"],
        "sample_aliases": ["BOP27_R1", "BOP27_R2", "delta_cydB_R1"],
    },
    "GCF_000009045.1_ASM904v1": {
        "species": "Bacillus subtilis 168",
        "gse": "GSE327651",
        "file": "GSE327651_RawCounts.txt.gz",
        "kind": "counts_tsv",
        "id_col": 0,
        "samples": ["WT168_1", "WT168_2", "WT168_3"],
        "id_transform": "strip_gene_prefix",
    },
    "GCF_000006845.1_ASM684v1": {
        "species": "Neisseria gonorrhoeae FA1090",
        "gse": "GSE152295",
        "file": "GSE152295_NGON_processed.txt.gz",
        "kind": "tpm_atlas",
        "samples": [
            "1 - NGON_Ctrl_1 (GE) - TPM",
            "1 - NGON_As_1 (GE) - TPM",
            "1 - NGON_Li_1 (GE) - TPM",
        ],
        "sample_aliases": ["NGON_Ctrl_1", "NGON_As_1", "NGON_Li_1"],
    },
    "GCF_000021165.1_ASM2116v1": {
        "species": "Helicobacter pylori G27",
        "gse": "GSE152295",
        "file": "GSE152295_HPG27_processed.txt.gz",
        "kind": "tpm_atlas",
        "samples": [
            "1 - HP_G27_Ctrl_1 (GE) - TPM",
            "1 - HP_G27_As_1 (GE) - TPM",
            "1 - HP_G27_Li_1 (GE) - TPM",
        ],
        "sample_aliases": ["HP_G27_Ctrl_1", "HP_G27_As_1", "HP_G27_Li_1"],
    },
    "GCF_000008685.2_ASM868v2": {
        "species": "Borreliella burgdorferi B31",
        "gse": "GSE304281",
        "file": "GSE304281_Exp1TPM.csv.gz",
        "kind": "tpm_csv",
        "id_col": "gene_id",
        "samples": [
            "TPM_JSB167.Media.1",
            "TPM_JSB167.Media.2",
            "TPM_JSB167.Media.3",
        ],
        "sample_aliases": ["Media_1", "Media_2", "Media_3"],
    },
    "GCF_000020425.1_ASM2042v1": {
        "species": "Bifidobacterium longum subsp. infantis ATCC 15697",
        "gse": "GSE155078",
        "file": "GSE155078_UMA272_raw_counts.txt.gz",
        "kind": "counts_tsv",
        "id_col": 0,
        "samples": ["ctrl.1", "ctrl.2", "cys.1"],
        "sample_aliases": ["ctrl1", "ctrl2", "cys1"],
    },
    "GCF_000020025.1_ASM2002v1": {
        "species": "Nostoc punctiforme PCC 73102",
        "gse": "GSE275682",
        "file": "GSE275682_sansTRNA.csv.gz",
        "kind": "counts_csv",
        "id_col": "Locus",
        "samples": ["Dawn0-1", "Dawn0-2", "Dawn0-3"],
        "sample_aliases": ["Dawn0_1", "Dawn0_2", "Dawn0_3"],
    },
    "GCF_000027105.1_ASM2710v1": {
        "species": "Clostridioides difficile R20291",
        "gse": "GSE336602",
        "file": "GSE336602_All_samples_raw_counts.csv.gz",
        "kind": "counts_csv_no_header_id",
        "samples": ["A_500_1_htseq", "A_500_2_htseq", "A_500_3_htseq"],
        "sample_aliases": ["A500_1", "A500_2", "A500_3"],
    },
    "GCF_000018225.1_ASM1822v1": {
        "species": "Rickettsia rickettsii str. Sheila Smith",
        "gse": "GSE290741",
        "file": "GSE290741_genotype_RSEM.genes.expected_count.all_samples.txt.gz",
        "kind": "counts_tsv",
        "id_col": "gene_id",
        "samples": ["genotype_WT_rep1", "genotype_WT_rep2", "genotype_WT_rep3"],
        "sample_aliases": ["WT_rep1", "WT_rep2", "WT_rep3"],
    },
    "GCF_003019785.1_ASM301978v1": {
        "species": "Fusobacterium nucleatum subsp. nucleatum ATCC 23726",
        "gse": "GSE284320",
        "file": "GSE284320_raw_counts_FNN.csv.gz",
        "kind": "counts_csv",
        "id_col": 0,
        "samples": ["Fnn23_H2O_30m_A", "Fnn23_H2O_30m_B", "Fnn23_H2O_30m_C"],
        "sample_aliases": ["H2O_30m_A", "H2O_30m_B", "H2O_30m_C"],
        "id_transform": "strip_cds_prefix",
    },
    "GCF_000210835.1_ASM21083v1": {
        "species": "Bacteroides fragilis 638R",
        "gse": "GSE282133",
        "file": "GSE282133_Bfr_raw_counts.csv.gz",
        "kind": "counts_csv",
        "id_col": "Geneid",
        "samples": ["WT1", "WT2", "WT3"],
        "sample_aliases": ["WT1", "WT2", "WT3"],
    },
    "GCF_000008725.1_ASM872v1": {
        "species": "Chlamydia trachomatis D/UW-3/CX",
        "gse": "GSE317337",
        "file": "GSE317337_count_table.tsv.gz",
        "kind": "counts_tsv",
        "id_col": 0,
        "samples": ["AB1", "AB2", "AB3"],
        "sample_aliases": ["AB1", "AB2", "AB3"],
        "id_transform": "ctl_to_ct",
    },
}


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


def geo_url(gse: str, filename: str) -> str:
    num = int(gse[3:])
    prefix = f"GSE{num // 1000}nnn"
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse}/suppl/{filename}"


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log(f"SKIP download (exists): {dest}")
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    log(f"CMD: download {url} -> {dest}")
    with urllib.request.urlopen(req, timeout=300) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)
    return dest


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw).decode("utf-8", "replace")
    # strip UTF-8 BOM
    return raw.decode("utf-8", "replace").lstrip("\ufeff")


def parse_gtf_genes(gtf: Path) -> dict[str, dict]:
    """gene_id -> {locus_tag, old_tags, gene, length, aliases}"""
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
            start, end = int(parts[3]), int(parts[4])
            old = attrs.get("old_locus_tag", "")
            old_tags = [t.strip() for t in re.split(r"[,;]", old) if t.strip()]
            locus = attrs.get("locus_tag") or ""
            gene = attrs.get("gene") or attrs.get("gene_name") or ""
            aliases = {gid}
            if locus:
                aliases.add(locus)
            aliases.update(old_tags)
            if gene:
                aliases.add(gene)
            # common NGO0001 / NGO_0001 variants
            for t in list(aliases):
                aliases.add(t.replace("_", ""))
                if re.match(r"^[A-Za-z]+\d+$", t):
                    # NGO0001 already
                    pass
            out[gid] = {
                "locus_tag": locus,
                "old_tags": old_tags,
                "gene": gene,
                "length": end - start + 1,
                "aliases": aliases,
            }
    return out


def build_alias_index(genes: dict[str, dict]) -> dict[str, str]:
    """Map alias -> gene_id; drop ambiguous aliases."""
    buckets: dict[str, set[str]] = {}
    for gid, info in genes.items():
        for a in info["aliases"]:
            buckets.setdefault(a, set()).add(gid)
            buckets.setdefault(a.upper(), set()).add(gid)
            buckets.setdefault(a.lower(), set()).add(gid)
    return {a: next(iter(gids)) for a, gids in buckets.items() if len(gids) == 1}


def transform_id(raw: str, how: str | None) -> str:
    s = raw.strip().strip('"')
    if not how:
        return s
    if how == "strip_gene_prefix":
        return re.sub(r"^gene-", "", s)
    if how == "strip_cds_prefix":
        s = re.sub(r"^cds-", "", s, flags=re.I)
        # cds-C4N14_00010 -> C4N14_00010; also map to RS form later via alias
        return s
    if how == "ctl_to_ct":
        # CTL0001 -> CT_001
        m = re.match(r'^"?CTL0*(\d+)"?$', s, re.I)
        if m:
            return f"CT_{int(m.group(1)):03d}"
        return s.strip('"')
    return s


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


def renormalize_tpm(tpm: dict[str, float]) -> dict[str, float]:
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


def align_values(
    raw: dict[str, float],
    alias_index: dict[str, str],
    genes: dict[str, dict],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for kid, val in raw.items():
        candidates = [kid, kid.upper(), kid.lower(), kid.replace("_", "")]
        # C4N14_00010 <-> C4N14_RS00010 heuristics
        m = re.match(r"^([A-Za-z0-9]+)_(\d+)$", kid)
        if m:
            candidates.append(f"{m.group(1)}_RS{m.group(2)}")
        m = re.match(r"^([A-Za-z0-9]+)_RS0*(\d+)$", kid)
        if m:
            candidates.append(f"{m.group(1)}_{int(m.group(2)):05d}")
            candidates.append(f"{m.group(1)}_{m.group(2)}")
        # NGO0001 style from NGO_RS / old
        m = re.match(r"^([A-Za-z]+)0*(\d+)$", kid.replace("_", ""))
        gid = None
        for c in candidates:
            if c in alias_index:
                gid = alias_index[c]
                break
            if c.upper() in alias_index:
                gid = alias_index[c.upper()]
                break
        if gid is None:
            continue
        # if duplicate aliases map to same gene, keep max (avoid double-count split)
        out[gid] = max(out.get(gid, 0.0), float(val))
    return out


def load_matrix(path: Path, cfg: dict) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Return (sample_names, {sample: {raw_id: value}})."""
    text = read_text(path)
    kind = cfg["kind"]
    id_xf = cfg.get("id_transform")

    if kind in ("tpm_csv", "counts_csv"):
        # DictReader; handle unnamed first column
        rows = list(csv.DictReader(text.splitlines()))
        if not rows:
            raise RuntimeError(f"Empty matrix {path}")
        # normalize header keys (BOM)
        def fix_row(r):
            return {(k.lstrip("\ufeff") if k else k): v for k, v in r.items()}

        rows = [fix_row(r) for r in rows]
        headers = list(rows[0].keys())
        id_col = cfg.get("id_col")
        if id_col is None or id_col == 0:
            # first column — may be None / "" key from unnamed
            id_key = headers[0]
        else:
            id_key = id_col if id_col in headers else headers[0]
        samples = cfg["samples"]
        if samples == "AUTO_FIRST3_NONEMPTY":
            prefer = cfg.get("prefer_samples", [])
            numeric_cols = [h for h in headers if h != id_key]
            chosen = [c for c in prefer if c in numeric_cols]
            for c in numeric_cols:
                if c not in chosen:
                    chosen.append(c)
                if len(chosen) >= 3:
                    break
            samples = chosen[:3]
        for s in samples:
            if s not in headers:
                raise RuntimeError(f"{path.name}: missing sample {s}; have {headers[:12]}")
        out: dict[str, dict[str, float]] = {s: {} for s in samples}
        for row in rows:
            rid = transform_id(str(row[id_key] or ""), id_xf)
            if not rid or rid in ("", "NA"):
                continue
            for s in samples:
                try:
                    out[s][rid] = float(row[s])
                except (TypeError, ValueError):
                    continue
        aliases = cfg.get("sample_aliases") or samples
        return list(aliases), {a: out[s] for a, s in zip(aliases, samples)}

    if kind == "counts_csv_no_header_id":
        # first line = sample headers only (no gene-id column name);
        # each data row = gene_id, then values in header order
        lines = text.splitlines()
        header = next(csv.reader([lines[0]]))
        samples = cfg["samples"]
        idx = {s: header.index(s) + 1 for s in samples}  # +1 for leading gene_id
        out = {s: {} for s in samples}
        for line in lines[1:]:
            parts = next(csv.reader([line]))
            if not parts:
                continue
            rid = transform_id(parts[0], id_xf)
            for s in samples:
                try:
                    out[s][rid] = float(parts[idx[s]])
                except (ValueError, IndexError):
                    continue
        aliases = cfg.get("sample_aliases") or samples
        return list(aliases), {a: out[s] for a, s in zip(aliases, samples)}

    if kind in ("counts_tsv", "tpm_atlas"):
        # sniff delimiter
        delim = "\t" if "\t" in text.splitlines()[0] else ","
        rows = list(csv.DictReader(text.splitlines(), delimiter=delim))
        rows = [{(k.lstrip("\ufeff").strip('"') if k else k): v for k, v in r.items()} for r in rows]
        headers = list(rows[0].keys())
        id_col = cfg.get("id_col", 0)
        if id_col == 0 or id_col not in headers:
            id_key = headers[0]
        else:
            id_key = id_col
        # for atlas, Name column is first
        if kind == "tpm_atlas":
            id_key = "Name" if "Name" in headers else headers[0]
        samples = cfg["samples"]
        # allow fuzzy match of sample headers
        resolved = []
        for s in samples:
            if s in headers:
                resolved.append(s)
            else:
                hits = [h for h in headers if s in h]
                if not hits:
                    raise RuntimeError(f"{path.name}: missing sample {s}; headers={headers[:8]}")
                resolved.append(hits[0])
        out = {s: {} for s in resolved}
        for row in rows:
            rid = transform_id(str(row[id_key] or ""), id_xf)
            if not rid:
                continue
            for s in resolved:
                try:
                    out[s][rid] = float(str(row[s]).strip('"'))
                except (TypeError, ValueError):
                    continue
        aliases = cfg.get("sample_aliases") or [re.sub(r"\W+", "_", s)[:40] for s in resolved]
        return list(aliases), {a: out[s] for a, s in zip(aliases, resolved)}

    raise RuntimeError(f"Unknown kind {kind}")


def genome_accession(stem: str) -> str:
    m = re.match(r"(GCF_\d+\.\d+)", stem)
    return m.group(1) if m else stem


def main() -> None:
    TPM_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)

    # Verify genomes present
    missing = []
    for stem in SPECIES:
        if not (FNA_DIR / f"{stem}_genomic.fna").exists():
            missing.append(f"{stem}_genomic.fna")
        if not (GTF_DIR / f"{stem}_genomic.gtf").exists():
            missing.append(f"{stem}_genomic.gtf")
    if missing:
        raise SystemExit(f"Missing genome files: {missing}")

    manifest_rows = []
    summary = []
    mapping_rows = []

    for stem, cfg in SPECIES.items():
        gtf = GTF_DIR / f"{stem}_genomic.gtf"
        genes = parse_gtf_genes(gtf)
        alias_index = build_alias_index(genes)
        lengths = {gid: info["length"] for gid, info in genes.items()}
        url = geo_url(cfg["gse"], cfg["file"])
        src = CACHE / cfg["file"]
        download(url, src)
        aliases, matrices = load_matrix(src, cfg)
        if len(aliases) != 3:
            raise RuntimeError(f"{stem}: expected 3 samples, got {aliases}")

        species_slug = re.sub(r"[^A-Za-z0-9]+", "_", cfg["species"]).strip("_")
        per_sample_info = []
        for alias in aliases:
            raw = matrices[alias]
            aligned = align_values(raw, alias_index, genes)
            kind = cfg["kind"]
            if kind in ("tpm_csv", "tpm_atlas") or "tpm" in kind:
                tpm = aligned
                method = "native_TPM_aligned"
            else:
                tpm = counts_to_tpm(aligned, lengths)
                method = "counts_to_TPM_GTF_lengths"
            if len(tpm) < 100:
                raise RuntimeError(
                    f"{stem}/{alias}: only {len(tpm)} genes aligned (raw keys={len(raw)})"
                )
            out_name = f"{cfg['gse']}_{alias}.csv"
            out_path = TPM_DIR / out_name
            write_wide_csv(out_path, tpm)
            digest = sha256_file(out_path)
            cov = len(tpm) / len(genes) if genes else 0
            per_sample_info.append(
                {
                    "sample": alias,
                    "csv": str(out_path.relative_to(ROOT)),
                    "n_tpm_genes": len(tpm),
                    "n_gtf_genes": len(genes),
                    "coverage": round(cov, 4),
                    "method": method,
                    "sha256": digest,
                }
            )
            mapping_rows.append(
                {
                    "id": f"{cfg['gse']}_{alias}",
                    "tpm": f"prokaryotes/tpm/{out_name}",
                    "assay": "RNA-seq",
                    "genome": genome_accession(stem),
                    "genome_stem": stem,
                    "species": cfg["species"],
                    "source": "NCBI GEO",
                    "gse": cfg["gse"],
                    "method": method,
                }
            )
            manifest_rows.append(
                {
                    "accession": f"{cfg['gse']}:{alias}",
                    "repository": "NCBI_GEO",
                    "download_url": url,
                    "local_path": str(out_path.relative_to(ROOT)),
                    "filename_bytes": out_path.stat().st_size,
                    "sha256": digest,
                    "download_status": "success",
                    "validation_status": "pass" if len(tpm) >= 200 else "warn",
                    "biological_status": "pass" if cov >= 0.3 else "warn",
                    "notes": f"{cfg['species']};{stem};{method};genes={len(tpm)}/{len(genes)}",
                    "download_date": DATE,
                }
            )
            log(
                f"WROTE {out_path.name} genes={len(tpm)}/{len(genes)} "
                f"cov={cov:.2%} method={method}"
            )

        summary.append(
            {
                "genome_stem": stem,
                "species": cfg["species"],
                "gse": cfg["gse"],
                "source_file": cfg["file"],
                "samples": per_sample_info,
            }
        )

        # also register genome files in manifest
        for kind, path in (
            ("fna", FNA_DIR / f"{stem}_genomic.fna"),
            ("gtf", GTF_DIR / f"{stem}_genomic.gtf"),
        ):
            digest = sha256_file(path)
            manifest_rows.append(
                {
                    "accession": genome_accession(stem),
                    "repository": "NCBI_RefSeq",
                    "download_url": "ncbi-datasets-cli",
                    "local_path": str(path.relative_to(ROOT)),
                    "filename_bytes": path.stat().st_size,
                    "sha256": digest,
                    "download_status": "success",
                    "validation_status": "pass",
                    "biological_status": "pass",
                    "notes": f"{cfg['species']};{kind}",
                    "download_date": DATE,
                }
            )

    # write mappings
    with MAPPINGS.open("w", newline="") as fh:
        fields = [
            "id",
            "tpm",
            "assay",
            "genome",
            "genome_stem",
            "species",
            "source",
            "gse",
            "method",
        ]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in mapping_rows:
            w.writerow(row)

    # manifest
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    mfields = [
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
        w = csv.DictWriter(fh, fieldnames=mfields, delimiter="\t")
        w.writeheader()
        for row in manifest_rows:
            w.writerow(row)

    CHECKSUMS.parent.mkdir(parents=True, exist_ok=True)
    with CHECKSUMS.open("w") as fh:
        for row in manifest_rows:
            fh.write(f"{row['sha256']}  {row['local_path']}\n")

    SUMMARY.write_text(json.dumps({"date": DATE, "species": summary}, indent=2))
    log(f"DONE species={len(SPECIES)} tpm_csv={len(mapping_rows)}")


if __name__ == "__main__":
    main()
