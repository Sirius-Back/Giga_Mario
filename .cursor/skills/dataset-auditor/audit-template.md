# Dataset Audit Templates

## Audit Report Template

```markdown
# Dataset Audit Report

**Date:** [YYYY-MM-DD]
**Auditor:** [agent / user]
**Overall status:** Ready | Ready with warnings | Not ready

## 1. Data inventory
| File | Path | Rows × Cols | Notes |
|------|------|-------------|-------|
| Sample metadata | ... | ... | ... |
| Count/abundance table | ... | ... | ... |
| FASTQ manifest | ... | ... | ... |

## 2. Metadata completeness
| Column | Required | Missing (n) | Missing (%) | Issues |
|--------|----------|-------------|-------------|--------|
| sample_id | Yes | 0 | 0% | OK |
| group | Yes | 2 | 1.2% | 2 samples excluded |

## 3. Sample identifiers
- **Unique IDs:** [Yes/No — list duplicates if any]
- **Naming conventions:** [consistent / issues found]

## 4. Class balance
| Group | n | % |
|-------|---|---|
| Control | 45 | 45% |
| Treated | 55 | 55% |

## 5. Duplicates and missing values
- **Duplicate rows:** [count]
- **Duplicate sample IDs:** [list or count]
- **Missing value pattern:** [summary]

## 6. Outliers
| Variable | Method | Flagged samples (n) | Action recommended |
|----------|--------|---------------------|-------------------|
| read_count | IQR 1.5× | 3 | Manual review |

## 7. Metadata ↔ file consistency
| Check | Pass/Fail | Details |
|-------|-----------|---------|
| All metadata IDs have FASTQ | Fail | 2 IDs missing files: S12, S47 |
| Paired-end complete | Pass | — |
| Matrix samples match metadata | Pass | 100/100 |

## 8. Sequencing depth (if applicable)
| Statistic | Value |
|-----------|-------|
| Median reads/sample | 1.24 M |
| Min / Max | 12k / 4.1 M |
| Samples below threshold (<100k) | 3 |

*Source: [file or tool output path]*

## 9. Batch effects and confounding
| Batch | Control | Treated | Total |
|-------|---------|---------|-------|
| Run1 | 40 | 5 | 45 |
| Run2 | 5 | 50 | 55 |

**Confounding note:** Batch strongly associated with group — batch correction alone may be insufficient; consider design limitation in analysis plan.

## 10. Recommendations
| Priority | Issue | Recommendation |
|----------|-------|----------------|
| Critical | Missing FASTQ for S12, S47 | Exclude or obtain files before pipeline |
| High | 3 low-depth samples | Exclude or re-sequence |
| Optional | group column name `cond` | Rename to match pipeline config |

## 11. Analysis readiness summary
[One paragraph: can analysis proceed, with what exclusions/caveats]
```

## Issue Table (compact)

```markdown
| ID | Check | Status | Evidence | Recommendation |
|----|-------|--------|----------|----------------|
| D01 | Unique sample_id | PASS | 100 unique | — |
| D02 | ID-file match | FAIL | 2 orphan IDs | Fix manifest |
| D03 | Class balance | WARN | 1:9 ratio in subset | Note power limit |
```

## Checklist by Data Type

### Tabular / metadata
- [ ] Primary key unique
- [ ] No unexpected NA in required fields
- [ ] Categorical levels as expected
- [ ] Date/numeric fields parse

### Count / abundance matrix
- [ ] Sample IDs align with metadata
- [ ] Features × samples orientation documented
- [ ] Zero-inflation summary
- [ ] Library size range plausible

### Sequencing
- [ ] FASTQ paths valid
- [ ] Paired files matched
- [ ] Depth stats available or QC recommended
- [ ] Sample sheet matches filesystem

### Study design
- [ ] n per group for planned comparisons
- [ ] Batch × treatment crosstab reviewed
- [ ] Exclusion criteria applied consistently

## Readiness Decision Guide

| Finding | Typical status |
|---------|----------------|
| Duplicate sample IDs | Not ready |
| >primary group missing | Not ready |
| Minor missing covariates | Ready with warnings |
| Batch confounded with group | Ready with warnings — document limitation |
| Orphan files (no metadata) | Warn — exclude or add metadata |

## Anti-Patterns

| Avoid | Prefer |
|-------|--------|
| Inventing read counts without QC files | "Depth not verified; run FastQC/MultiQC" |
| Silent sample dropping | Explicit exclusion list with reasons |
| "Batch corrected" without design check | Report confounding; recommend valid approach |
| Passing audit with ID mismatches | Fail until joins are clean |
