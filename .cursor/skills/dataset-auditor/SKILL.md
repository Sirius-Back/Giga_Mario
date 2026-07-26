---
name: dataset-auditor
description: >-
  Inspect datasets for metadata completeness, identifiers, balance, duplicates,
  missing values, outliers, batch effects, and file consistency before analysis.
  Use when the user asks for a dataset audit, QC report, data readiness check,
  or metadata validation prior to analysis.
disable-model-invocation: true
---

# Dataset Auditor

Inspect datasets before analysis. Verify metadata completeness, sample identifiers, class balance, duplicates, missing values, outliers, batch effects, consistency between metadata and files, sequencing depth where applicable, and overall dataset readiness. Produce a structured audit report with recommendations.

Follow project rules: **validation-first**, **scientific-integrity**, **missing-data-policy**, **statistical-analysis**, **reproducibility**.

## Workflow

Copy and track progress:

```
Dataset audit:
- [ ] Step 1: Locate data artifacts
- [ ] Step 2: Validate metadata structure
- [ ] Step 3: Cross-check metadata vs files
- [ ] Step 4: Run quality checks
- [ ] Step 5: Assess analysis readiness
- [ ] Step 6: Write audit report and recommendations
```

### Step 1: Locate data artifacts

Find and inventory:

| Artifact | Typical locations |
|----------|-------------------|
| Sample metadata | `metadata/`, `*.tsv`, `*.csv`, `samplesheet` |
| Feature/count tables | `data/`, `counts/`, `abundance*.tsv` |
| Sequence files | `fastq/`, `*.fq.gz`, sample sheets with paths |
| Phenotype / labels | metadata columns, separate phenotype files |
| Batch / technical covariates | metadata columns, run sheets |
| Config references | `config/*.yaml`, workflow sample sheets |

Record file paths, row/column counts, and delimiters. Read files — do not assume schema from filenames.

### Step 2: Validate metadata structure

Check:

- **Required columns** present for planned analysis (sample ID, group, batch, etc.)
- **Sample identifiers** unique, consistent naming (no whitespace/case traps)
- **Data types** — numeric vs categorical fields parse correctly
- **Missing values** — count and pattern (MCAR vs structural)
- **Class balance** — group sizes; flag severe imbalance
- **Duplicates** — duplicate sample IDs or duplicated rows
- **Cardinality** — unexpected levels in categorical fields

Flag columns with >expected missingness or ambiguous encoding.

### Step 3: Cross-check metadata vs files

Verify consistency:

| Check | Action |
|-------|--------|
| Metadata samples ↔ file list | Every metadata ID has matching FASTQ/table row; no orphan files |
| Paired-end pairing | R1/R2 pairs complete |
| Count matrix orientation | Sample IDs match metadata (columns vs rows documented) |
| Subset filters | Included samples exist in all modalities |
| Path validity | Referenced paths exist and are non-empty |

Report **exact mismatches** with IDs listed (or counts if list is huge).

### Step 4: Run quality checks

Apply checks appropriate to data type. Execute scripts/commands when files exist.

**All tabular data**
- Missing value heatmap summary (counts per column)
- Outlier flags (IQR or domain rules — state method)
- Constant/near-zero variance features (if feature matrix)

**Classification / cohort studies**
- Class balance table with proportions
- Minimum n per group vs planned test (warn if underpowered)

**Sequencing (when FASTQ or depth stats available)**
- Reads per sample (median, range)
- Failed/empty libraries (zero or very low depth)
- GC content extremes if QC files exist
- Batch vs sequencing run confounding with group (crosstab)

**Batch effects (when batch + phenotype available)**
- Crosstab batch × group
- Note if batch is fully confounded with biology — blocks some analyses
- Recommend batch-aware methods or correction only when design supports it; do not overclaim correction fixes confounding

Use verified numbers only. If QC tools have not been run, state that and recommend running them — do not invent depth statistics.

### Step 5: Assess analysis readiness

Assign overall status:

| Status | Meaning |
|--------|---------|
| **Ready** | All critical checks pass |
| **Ready with warnings** | Proceed with documented caveats |
| **Not ready** | Critical failures block analysis |

Critical failures include: missing primary metadata, ID mismatches, empty files, no samples in a required group, duplicated IDs breaking joins.

### Step 6: Write audit report and recommendations

Use the template in [audit-template.md](audit-template.md).

Recommendations must be **actionable** (fix IDs, drop samples, merge batches, rerun QC, add columns). Prioritize: Critical / High / Optional.

Do not recommend specific statistical tests without noting design constraints surfaced in the audit.

## Deliverables

Default output:

1. **Dataset Audit Report** — structured findings
2. **Issue table** — check, status, evidence, recommendation
3. **Sample exclusion list** (if applicable) — IDs and reasons

Save to user-requested path, or suggest `docs/dataset-audit.md`.

Optional: runnable audit script stub only if user requests implementation.

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

- Report template and checklists: [audit-template.md](audit-template.md)
