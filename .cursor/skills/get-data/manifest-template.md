# Manifest and Report Templates

## download_manifest.tsv

Tab-separated; one row per file acquired.

```tsv
accession	repository	download_url	local_path	filename_bytes	sha256	download_status	validation_status	biological_status	notes	download_date
SRR1234567	NCBI_SRA	https://...	data/raw/SRR1234567_1.fastq.gz	1234567890	abc123...	success	pass	pass		2026-07-15
SRR1234567	NCBI_SRA	https://...	data/raw/SRR1234567_2.fastq.gz	1234567890	def456...	success	pass	pass	paired R2	2026-07-15
SRR9999999	NCBI_SRA	—	—	0	—	failed	—	—	prefetch timeout	2026-07-15
```

### Column definitions

| Column | Values |
|--------|--------|
| `download_status` | `success`, `failed`, `skipped`, `partial` |
| `validation_status` | `pass`, `fail`, `not_run` |
| `biological_status` | `pass`, `fail`, `warn`, `not_run` |

## checksums.txt

```
# SHA256 checksums — generated YYYY-MM-DD
# tool: sha256sum (GNU coreutils 9.1)

abc123def456...  data/raw/SRR1234567_1.fastq.gz
def456abc789...  data/raw/SRR1234567_2.fastq.gz
```

## download.log

Append-only command log:

```
[2026-07-15T08:30:00Z] datasets --version
[2026-07-15T08:30:01Z] CMD: datasets summary sra accession SRR1234567 ...
[2026-07-15T08:30:05Z] ESTIMATE: 4.2 GB compressed, ~8 GB decompressed
[2026-07-15T08:30:10Z] CMD: sbatch src/sbatch/get_data_sra.sbatch
[2026-07-15T10:15:00Z] CMD: sha256sum data/raw/SRR1234567_1.fastq.gz
[2026-07-15T10:15:01Z] VALIDATE: samtools quickcheck — N/A (FASTQ)
[2026-07-15T10:15:02Z] BIO: paired-end check PASS — R1 and R2 present
```

## Download plan (pre-execution)

Present to user before large or multi-accession downloads:

```markdown
# Download Plan

**Date:** YYYY-MM-DD
**Requested identifiers:** SRR1, SRR2, … (N total)

## Summary
| Metric | Value | Confidence |
|--------|-------|------------|
| Accessions | N | Verified |
| Total download (compressed) | ~X GB | Approximate / Verified |
| After decompression | ~Y GB | Approximate |
| Temp space needed | ~Z GB | Approximate |
| Est. time (100 Mbps) | ~H hours | Approximate |

## Per-accession plan
| Accession | Repository | Method | Est. size |
|-----------|------------|--------|-----------|
| SRR1234567 | NCBI | datasets CLI | 4.2 GB |

## Execution mode
- [ ] Local
- [x] SLURM (`src/sbatch/get_data.sbatch`)

## Warnings
- Total > 50 GB — confirm disk space on /mnt/tank/...

Proceed? (User confirmation required for large jobs)
```

## acquisition_report.md

```markdown
# Data Acquisition Report

**Date:** YYYY-MM-DD
**Operator:** [agent / user]

## Requested datasets
- SRR1234567, SRR2345678 (from `samples/accessions.tsv`)

## Repositories used
| Repository | Accessions | Method |
|------------|------------|--------|
| NCBI SRA | 2 | NCBI Datasets CLI v16.x |

## Acquisition details
- **APIs / endpoints:** datasets.ncbi.nlm.nih.gov; ENA filereport (if used)
- **Software versions:** datasets 16.20.0; sra-tools 3.0.10; curl 8.5.0
- **SLURM job:** 12345 (if applicable)

## Executed commands
​```
[dataset commands from download.log]
​```

## Download summary
| Metric | Expected | Actual |
|--------|----------|--------|
| Accessions requested | 2 | 2 |
| Files expected | 4 | 4 |
| Files downloaded | 4 | 4 |
| Total size | ~8 GB | 7.9 GB |
| Successful | 2 | 2 |
| Failed | 0 | 0 |

## Technical validation summary
| Check | Result |
|-------|--------|
| Checksums | 4/4 pass |
| Archive integrity | pass |
| Non-empty files | pass |
| Format readable | 4/4 FASTQ pass |

## Biological validation summary
| Check | Result | Notes |
|-------|--------|-------|
| Sample count vs accessions | pass | 2/2 |
| Paired-end completeness | pass | R1+R2 for each |
| Metadata consistency | pass | ENA filereport match |
| Suspiciously small files | none | |

## Detected problems
| Severity | Issue | Accession | Action |
|----------|-------|-----------|--------|
| — | none | — | — |

## Failed downloads
| Accession | Error | Recommendation |
|-----------|-------|----------------|
| — | — | — |

## Recommendations for preprocessing
1. Run `@dataset-auditor` on `data/metadata/` and FASTQ manifest
2. [Study-specific QC steps if evident from metadata]

## Reproducibility notes
- Manifest: `data/manifests/download_manifest.tsv`
- Checksums: `data/checksums/checksums.txt`
- Full log: `data/logs/download.log`
```

## Input file parsing

For TSV/CSV/XLSX accession lists:

- Auto-detect ID column (headers: `accession`, `sample_id`, `srr`, `run`, etc.)
- If multiple columns, prefer explicit accession column
- Report rows with invalid IDs without stopping entire batch (unless all invalid)

```markdown
## Input validation report
- Total rows: 100
- Valid accessions: 98
- Duplicates removed: 3
- Malformed: 2 (row 14: "SR123"; row 87: empty)
```
