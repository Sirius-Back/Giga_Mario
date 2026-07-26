# Data Preparation Report Templates

## data-preparation-report.md

```markdown
# Data Preparation Report

**Date:** YYYY-MM-DD
**Mode:** Full execution | Dry run
**Overall status:** Data-ready | Not data-ready | Partial

## Executive summary
[3–5 sentences: what was requested, what was done, current readiness]

## Requested datasets
| Identifier | Source requested | Required |
|------------|------------------|----------|
| SRR1234567 | user input | yes |

---

## Phase 0 — Project state detection

### Existing artifacts found
| Artifact | Path | Status |
|----------|------|--------|
| Download manifest | data/manifests/download_manifest.tsv | present |
| Acquisition report | data/manifests/acquisition_report.md | present — 2026-07-10 |
| Dataset audit | docs/dataset-audit.md | present — Ready with warnings |

### Per-dataset state
| ID | Present | Technical validation | Biological validation | Decision |
|----|---------|---------------------|----------------------|----------|
| SRR1234567 | yes | pass | pass | skip download |
| SRR2345678 | no | — | — | acquire |

### Skipped actions (Phase 0)
| Action | Reason |
|--------|--------|
| @get-data for SRR1234567 | Already validated in manifest (2026-07-10) |

### Detected issues in existing data
| Issue | Severity | Details |
|-------|----------|---------|
| Duplicate FASTQ | warn | SRR1234567 listed twice in raw/ |

---

## Phase 1 — Data acquisition (@get-data)

**Invoked:** Yes | No (skipped)

### Summary
| Metric | Value |
|--------|-------|
| Datasets requested for download | N |
| Downloaded | N |
| Skipped (already valid) | N |
| Failed | N |

### Repositories and methods
| Accession | Repository | Method | Status |
|-----------|------------|--------|--------|
| SRR2345678 | NCBI SRA | datasets CLI | success |

### Failures (download vs validation)
| Accession | Failure type | Error | Blocks pipeline |
|-----------|--------------|-------|-----------------|
| — | — | — | — |

**Subordinate report:** [data/manifests/acquisition_report.md](data/manifests/acquisition_report.md)

---

## Phase 2 — Dataset audit (@dataset-auditor)

**Invoked:** Yes | No (skipped — reason)

### Summary
**Audit status:** Ready | Ready with warnings | Not ready

| Check area | Result |
|------------|--------|
| Metadata completeness | pass |
| Paired-end consistency | pass |
| Sequencing depth | warn — 3 low-depth samples |
| Batch confounding | warn |

**Subordinate report:** [docs/dataset-audit.md](docs/dataset-audit.md)

### Critical issues
| Issue | Impact | Recommendation |
|-------|--------|----------------|
| — | — | — |

---

## Phase 3 — Scientific project audit (@project-auditor)

**Invoked:** Yes | No (skipped — reason)

### Summary
**Publication / analysis readiness:** [rating from project audit]

| Dimension | Rating |
|-----------|--------|
| Reproducibility | Good |
| Data organization | Fair |

**Subordinate report:** [docs/project-audit.md](docs/project-audit.md)

### Blocking issues (P0)
| Issue | Evidence |
|-------|----------|
| — | — |

---

## Consolidated findings

### Detected issues (all phases)
| # | Phase | Severity | Issue | Status |
|---|-------|----------|-------|--------|
| 1 | 0 | warn | Obsolete duplicate in raw/ | open |

### Unresolved problems
- [List items blocking data-ready status]

### Recommended next steps
1. [Actionable step — e.g., run preprocessing QC]
2. [@generate-todo — update task T-2.1]

---

## Success criteria checklist

| Criterion | Met | Evidence |
|-----------|-----|----------|
| All datasets acquired or available | ✓ | manifest |
| Technical validation passed | ✓ | checksums.txt |
| Biological validation passed | ✓ | acquisition_report.md |
| Dataset audit successful | ✓ | docs/dataset-audit.md |
| No critical project audit blockers | ✗ | P0: missing environment.yml |
| Reports and manifests generated | ✓ | paths listed above |
| Ready for downstream analyses | ✗ | resolve P0 first |

**Data-ready:** No — resolve P0 project audit items and low-depth samples.

---

## Referenced reports (preserved, not replaced)

| Report | Path |
|--------|------|
| Acquisition | data/manifests/acquisition_report.md |
| Download manifest | data/manifests/download_manifest.tsv |
| Dataset audit | docs/dataset-audit.md |
| Project audit | docs/project-audit.md |
| Download log | data/logs/download.log |
```

## data-preparation-plan.md (Dry Run)

```markdown
# Data Preparation Plan (Dry Run)

**Date:** YYYY-MM-DD
**No files modified**

## Datasets to acquire
| ID | Repository | Method | Est. size | Already present |
|----|------------|--------|-----------|-----------------|
| SRR2345678 | NCBI | datasets CLI | ~4 GB | no |

## Storage estimates
| Metric | Estimate | Confidence |
|--------|----------|------------|
| Download (compressed) | X GB | Verified/Approximate |
| After decompression | Y GB | Approximate |
| Temp space | Z GB | Approximate |
| Est. time (100 Mbps) | H h | Approximate |

## Required software
- NCBI datasets CLI, curl, sha256sum

## Required repositories / endpoints
- datasets.ncbi.nlm.nih.gov

## Expected outputs
- data/raw/*.fastq.gz
- data/manifests/download_manifest.tsv
- data/manifests/acquisition_report.md

## Missing prerequisites
| Item | Blocks | Action |
|------|--------|--------|
| — | — | — |

## Phases that would run
| Phase | Would invoke | Reason |
|-------|--------------|--------|
| 0 | — | State detection |
| 1 | @get-data | 1 accession missing |
| 2 | @dataset-auditor | post-download |
| 3 | @project-auditor | readiness check |
| 4 | consolidated report | always |

## Skipped downloads
| ID | Reason |
|----|--------|
| SRR1234567 | Already validated — manifest pass |
```

## Phase skip log (internal)

Track for every execution:

```markdown
| Phase | Skipped | Reason |
|-------|---------|--------|
| 1 | yes | All accessions validated in manifest |
| 2 | no | Re-audit after new download |
| 3 | no | Required for data-ready check |
```
