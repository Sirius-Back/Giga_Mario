# Repository Reference — Get Data

Official sources only. Verify commands against current tool documentation before execution.

## Accession type detection

| Pattern | Type | Primary repository |
|---------|------|-------------------|
| `SRR\d+`, `SRX\d+`, `SRS\d+`, `SAMN\d+`, `PRJNA\d+` | SRA / BioProject (NCBI) | NCBI SRA / Datasets |
| `ERR\d+`, `ERX\d+`, `ERS\d+`, `PRJEB\d+` | ENA | EBI ENA |
| `DRR\d+`, `DRX\d+`, `DRS\d+` | DDBJ | DDBJ / via ENA mirror |
| `GSE\d+`, `GSM\d+`, `GPL\d+` | GEO | NCBI GEO |
| `E-\w+-\d+` | ArrayExpress | EBI ArrayExpress |
| `MGYS\d+`, `MGYA\d+`, `MGYG\d+` | MGnify | EBI MGnify API |
| `PXD\d+` | PRIDE | ProteomeXchange |
| `zenodo\.\d+`, `\d+\.\d+/zenodo\.\d+` | Zenodo | Zenodo REST API |
| `10\.\d+/` (DOI) | DOI | Resolve via doi.org → repository |
| ENCODE accessions | ENCODE | ENCODE REST API |
| TCGA barcodes | GDC | GDC Data Transfer Tool |
| GitHub repo/release URL | GitHub | Official releases API |

Unsupported or malformed → report in Step 1; do not guess repository.

## Preferred acquisition methods

### NCBI SRA (SRR, SRX, PRJNA)

**Size estimation:**
```bash
datasets summary sra accession SRR1234567 --report ids,bytes,bases,files
```

**Download (preferred — NCBI Datasets CLI):**
```bash
datasets download sra accession SRR1234567 \
  --filename data/raw/SRR1234567.zip --decompress
```

**Alternative (SRA Toolkit):**
```bash
prefetch SRR1234567 --output-directory data/raw/
fasterq-dump data/raw/SRR1234567 --outdir data/raw/ --threads "${SLURM_CPUS_PER_TASK:-16}"
```

Record `prefetch --version` and `datasets --version`.

### ENA (ERR, ERX, PRJEB)

**Filereport API (metadata + FTP links):**
```bash
curl -s "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=ERR1234567&result=read_run&fields=run_accession,fastq_ftp,fastq_bytes,fastq_md5"
```

**Download:** use `fastq_ftp` HTTPS links from filereport — official ENA endpoints.

```bash
# Example after parsing filereport
curl -L -C - -o "data/raw/${FILENAME}" "ftp.sra.ebi.ac.uk/vol1/fastq/..."
```

Prefer HTTPS/FTP from filereport over third-party mirrors.

### GEO (GSE, GSM)

**Soft metadata:**
```bash
curl -L -o data/metadata/GSE12345_family.soft.gz \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE12nnn/GSE12345/soft/GSE12345_family.soft.gz"
```

**Supplementary / processed files:** parse SOFT or use NCBI FTP paths from GEO documentation.

**Raw SRA linked to GEO:** resolve GSM → SRR via metadata, then use SRA/ENA methods.

### MGnify

**API:**
```bash
curl -s "https://www.ebi.ac.uk/metagenomics/api/v1/analyses/MGYA1234567890/files"
```

Download files from URLs returned by official API only.

### Zenodo

**API:**
```bash
curl -s "https://zenodo.org/api/records/1234567"
```

Download from `files[].links.self` in JSON response. Record DOI and record ID.

### Figshare

Resolve article ID via Figshare API v2; download from official file URLs in response.

### Dryad

Resolve DOI → datadryad.org download links from dataset landing page or API.

### ENCODE

**REST API:**
```bash
curl -s "https://www.encodeproject.org/search/?type=File&accession=ENCFF123ABC"
```

Download from `href` in official ENCODE portal response.

### TCGA / GDC

Controlled access — **stop** and document GDC token requirement.

Open-access GDC data:
```bash
gdc-client download -m data/manifests/gdc_manifest.txt -d data/raw/
```

User must supply valid GDC manifest and credentials for controlled data.

### GitHub Releases

```bash
curl -L -o data/raw/asset.tar.gz \
  "https://github.com/org/repo/releases/download/v1.0.0/asset.tar.gz"
```

Verify release tag and asset checksum if published.

## SLURM template (large downloads)

```bash
#!/bin/bash
#SBATCH --job-name=get_data_sra
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=data/logs/get_data_%j.out
#SBATCH --error=data/logs/get_data_%j.err

set -euo pipefail
# Log all commands
exec > >(tee -a data/logs/download.log) 2>&1

datasets download sra accession SRR1234567 \
  --filename data/raw/SRR1234567.zip --decompress
```

Adjust memory/time from Step 3 estimates.

## Technical validation commands

| Format | Check |
|--------|-------|
| FASTQ | `zcat file.fq.gz \| head -4`; count reads if feasible |
| FASTA | `seqkit stats` or `grep -c '^>'` |
| BAM | `samtools quickcheck -v file.bam` |
| VCF | `bcftools view -h file.vcf` |
| gzip | `gzip -t file.gz` |
| tar | `tar tf file.tar \| head` |

## Checksum verification

```bash
sha256sum file > data/checksums/file.sha256
sha256sum -c data/checksums/file.sha256
```

Compare against repository-provided MD5/SHA when available from ENA filereport or Zenodo API.

## Biological validation checks

| Check | Method |
|-------|--------|
| Sample count | Compare manifest rows vs unique accessions requested |
| Paired-end | Match `_1`/`_2`, `R1`/`R2`, or `.fq.gz` pairs per sample |
| Split lanes | Enumerate `L001`, `L002` completeness per sample |
| Metadata match | Join local filenames to ENA/NCBI sample attributes |
| Suspicious size | Flag FASTQ < 1 MB (adjust threshold per study type) |
| Duplicates | Same MD5 or same accession downloaded twice |

## When to stop (missing-data-policy)

- Accession not found in official API
- No official download URL retrievable
- Controlled access without user credentials
- Ambiguous DOI resolving to multiple datasets — ask user to specify

Report: what was searched, why download cannot proceed, what user must provide.
