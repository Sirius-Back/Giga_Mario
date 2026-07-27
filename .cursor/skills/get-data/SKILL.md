---
name: get-data
description: >-
  Acquire scientific datasets from public repositories (SRA, ENA, GEO, Zenodo,
  etc.) with validation, manifests, and reproducible logging. Use when the user
  asks to download accessions, fetch public data, acquire datasets, or populate
  data/raw from repository identifiers or URLs.
disable-model-invocation: true
---

# Get Data

## Purpose

Acquire scientific datasets from public repositories in a fully reproducible, verifiable and publication-ready manner.

Follow project rules: **reproducibility**, **validation-first**, **slurm-execution-policy**, **missing-data-policy**, **scientific-integrity**.

## Supported inputs

- Individual accession numbers (e.g. SRR, ERR, DRR, PRJNA, PRJEB, GSE, GSM, BioProject, BioSample, SRA, ENA, DOI, Zenodo, Figshare, Dryad, MGnify, ArrayExpress, PRIDE, ProteomeXchange, ENCODE, TCGA, GTEx, etc.)
- Lists of identifiers stored in TXT, CSV, TSV, XLSX or other tabular formats.
- URLs pointing to datasets or repository pages.
- Existing metadata files containing dataset identifiers.
- Optional destination directory.

## Workflow

Copy and track progress:

```
Data acquisition:
- [ ] Step 1: Parse and validate input
- [ ] Step 2: Discover official data source
- [ ] Step 3: Estimate download requirements
- [ ] Step 4: Organize project structure
- [ ] Step 5: Download data
- [ ] Step 6: Verify downloaded data
- [ ] Step 7: Perform biological validation
- [ ] Step 8: Generate machine-readable manifests
- [ ] Step 9: Produce acquisition_report.md
```

### Step 1: Parse and validate input

- Automatically detect accession types.
- Read identifiers from supplied files.
- Remove duplicates.
- Validate identifier syntax.
- Report malformed or unsupported identifiers.

See accession patterns in [reference.md](reference.md).

### Step 2: Discover the official data source

- Search official documentation when necessary.
- Prefer official APIs.
- Prefer official FTP or HTTPS endpoints.
- Avoid unofficial mirrors whenever official repositories exist.
- Select the most reliable acquisition method available.

Preferred acquisition methods include (when applicable):

- ENA FTP/HTTPS
- NCBI Datasets CLI
- prefetch + fasterq-dump
- GEO FTP
- EBI APIs
- MGnify API
- Zenodo REST API
- Figshare API
- Dryad downloads
- ENCODE REST API
- GDC Data Transfer Tool
- Official GitHub Releases

Document chosen method and endpoint in the download plan before execution. See [reference.md](reference.md) for repository-specific commands.

### Step 3: Estimate download requirements

Before downloading:

- Estimate total download size whenever possible.
- Estimate required disk space after decompression.
- Estimate temporary storage requirements.
- Estimate expected download time if possible.
- Warn about unusually large datasets.
- Produce a download plan before execution.

Use official APIs or `ncbi-datasets summary`, ENA filereport, etc. when available. Mark estimates as **Verified** (from API) or **Approximate** (heuristic).

**Stop and present the download plan for user confirmation** when total size exceeds a reasonable threshold (default: >50 GB) unless user pre-approved.

### Step 4: Organize project structure

Create (when appropriate):

```
data/
    raw/
    metadata/
    manifests/
    checksums/
    logs/
```

Store downloaded files using original filenames whenever possible. Respect user-supplied destination directory.

### Step 5: Download data

- Prefer resumable downloads.
- Retry transient failures.
- Parallelize downloads where appropriate.
- Preserve repository metadata.
- Preserve timestamps whenever supported.
- Log every executed command to `data/logs/download.log`.

Large downloads should automatically generate **SLURM jobs** instead of interactive commands unless the user explicitly requests local execution. Per **slurm-execution-policy**: specify CPUs (16 or 32, even), memory, wall time, output log, error log.

### Step 6: Verify downloaded data

Perform all available validation:

- checksum verification
- archive integrity
- successful decompression
- expected number of files
- expected directory structure
- expected file sizes
- non-empty files
- readable FASTQ/FASTA/BAM/VCF/etc.
- metadata consistency

Never assume downloads succeeded without validation.

### Step 7: Perform biological validation

After successful technical validation, verify that the downloaded data are biologically consistent with the requested study.

Whenever applicable:

- Verify that the number of downloaded samples matches the requested accessions.
- Verify that sample identifiers match repository metadata.
- Verify sequencing layout (single-end vs paired-end).
- Ensure every paired-end read has both R1 and R2 files.
- Verify that lane or split files are complete when multiple files per sample are expected.
- Confirm that expected file types are present (FASTQ, FASTA, BAM, CRAM, VCF, GTF, metadata, etc.).
- Compare downloaded metadata against repository metadata whenever available.
- Detect duplicated, missing or unexpected samples.
- Report empty or suspiciously small sequencing files.
- Report inconsistencies between metadata and downloaded files.
- Summarize all detected biological inconsistencies in the final report.

Never assume that technically valid downloads are biologically complete.

### Step 8: Generate machine-readable manifests

Produce in `data/manifests/`:

- `download_manifest.tsv`
- `checksums.txt` (in `data/checksums/`)
- `download.log` (in `data/logs/`)

Manifest columns — see [manifest-template.md](manifest-template.md).

### Step 9: Produce acquisition_report.md

Include:

- requested datasets
- repositories used
- acquisition methods
- APIs or FTP endpoints
- software versions
- executed commands
- download date
- total download size
- expected file count
- actual file count
- successful downloads
- failed downloads
- validation summary
- detected problems
- recommendations for preprocessing

Save to `data/manifests/acquisition_report.md` or user path.

## Rules

- Never use unofficial repositories when official ones exist.
- Never silently ignore download failures.
- Never fabricate download links.
- Never assume validation passed.
- Preserve original filenames whenever possible.
- Prefer reproducible command-line tools over browser downloads.
- Record every external data source.
- Record every executed command.
- Record software versions whenever possible.
- If official documentation or download methods cannot be found, stop execution and report the issue instead of guessing.
- If authentication or controlled-access datasets are encountered, explain the required authorization process instead of attempting unsupported workarounds.

## Controlled-access datasets

When dbGaP, controlled GDC, or EGA accessions are detected:

1. Stop automated download
2. Document required authorization (dbGaP approval, GDC token, EGA account)
3. Provide official access URLs and typical workflow
4. Do not attempt credential bypass

## Deliverables

| Output | Path |
|--------|------|
| Raw data | `data/raw/` |
| Repository metadata | `data/metadata/` |
| Manifest | `data/manifests/download_manifest.tsv` |
| Checksums | `data/checksums/checksums.txt` |
| Command log | `data/logs/download.log` |
| Report | `data/manifests/acquisition_report.md` |
| SLURM script (if large) | `src/sbatch/get-data_<jobid>.sbatch` |

## Artifact registration

Instead of creating standalone reports in arbitrary locations, require every generated artifact to be registered inside `artifact-registry.md` (prefer `docs/artifact-registry.md`).

Each registry entry must contain:

- artifact
- producer skill
- generation date
- purpose
- status
- downstream consumers

Every generated report, graph, manifest or checkpoint must be registered immediately after it is written.

Update existing rows when regenerating the same path; mark replaced paths `superseded`.

Format: [artifact-registry-template.md](../_shared/artifact-registry-template.md). Project rule: `artifact-registry` (alwaysApply).

## Additional resources

- Repository-specific commands and APIs: [reference.md](reference.md)
- Manifest and report templates: [manifest-template.md](manifest-template.md)
