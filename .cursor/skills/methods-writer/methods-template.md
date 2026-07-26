# Methods Output Templates

## Methods Section Template

Adapt headings to the project. Remove empty subsections.

```markdown
# Methods

## Study design and samples
[Design, cohort, inclusion/exclusion — only if documented in metadata/README]

## Data generation and acquisition
[Sequencing platform, library prep, public accession numbers — verified only]

## Preprocessing and quality control
[Adapter trimming, filtering thresholds, host removal, QC tools and cutoffs]

## [Primary analysis name, e.g. metagenomic assembly and annotation]
[Step-by-step workflow in execution order; key parameters]

## [Secondary analyses]
[Taxonomic profiling, functional annotation, etc.]

## Statistical analysis
[Tests, assumptions, multiple-testing correction, effect sizes, software]

## Visualization
[Only if plotting code defines non-default analytical choices worth reporting]

## Computational infrastructure
[Workflow runner, SLURM resources, containers — if verified and relevant]

## Reproducibility
[Environment file, workflow entry command, random seeds, version control reference]

## Software and references
| Software | Version | Source / citation |
|----------|---------|-------------------|
| ...      | ...     | ...               |
```

### Subsection prose pattern

Each subsection should follow:

1. **What was done** (past tense)
2. **With what tool** (name + version)
3. **Key parameters** (thresholds, references, databases)
4. **Criteria** (filtering, inclusion rules)

**Example (verified content only):**

> Quality-filtered paired-end reads were assembled with MEGAHIT v1.2.9 using default k-mer steps and a minimum contig length of 500 bp. Assemblies were submitted to [downstream step] as described below.

## Missing Information Report

Deliver separately from Methods. Never merge unverified content into Methods prose.

```markdown
# Missing Information Report

The following items could not be verified from project artifacts. They are **excluded** from the Methods section until confirmed.

## [Category, e.g. Software versions]

### [Item name]
- **Missing:** [Specific value or description]
- **Searched:** [Files, commands, docs inspected]
- **Blocks:** [Which Methods sentence/subsection cannot be written]
- **Required from user:** [Exact file, command output, or decision]

## Summary
| Priority | Count | Action |
|----------|-------|--------|
| Critical | N     | Must resolve before submission |
| Optional | N     | Improves completeness |
```

## Evidence Table (optional appendix)

For author review before submission:

```markdown
# Methods Evidence Table

| Methods claim | Source | Location | Confidence |
|---------------|--------|----------|------------|
| MEGAHIT v1.2.9 | conda env | environment.yml | Verified |
| FDR < 0.05 | script default | scripts/de.R:42 | Verified |
| Kraken2 database | not found | — | Unknown |
```

## Confidence Labels

| Label | Use in Methods? |
|-------|-----------------|
| Verified | Yes — state as fact |
| Inferred | Only with explicit caveat, or move to Missing Information Report |
| Unknown | No — Missing Information Report only |
