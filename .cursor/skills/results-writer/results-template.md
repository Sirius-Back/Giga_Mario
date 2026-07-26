# Results Output Templates

## Results Section Template

Adapt headings to the project. Remove empty subsections. Order follows the analysis narrative.

```markdown
# Results

## [Overview or cohort summary — if supported by tables]
[Sample counts, groups, data availability — observational facts only]

## [Quality control and data processing outcomes]
[Read counts, filtering rates, assembly metrics — cite Fig./Table]

## [Primary analysis results]
[Main endpoint or comparison; statistics with effect sizes]

## [Secondary or exploratory analyses]
[Clearly labeled exploratory if not pre-specified]

## [Additional analyses]
[Subgroup, sensitivity, or supplementary findings with verified outputs]
```

## Observation Prose Patterns

### Comparison with statistics

> [Metric] was [higher/lower] in [group A] than [group B] (median [value], IQR [range]; Wilcoxon rank-sum, q = [value], n = [nA] vs [nB]) (**Fig. 1a**, **Table 1**).

### Count or prevalence

> [Feature] was detected in [k] of [n] samples ([percent]%) (**Fig. 2b**).

### QC or workflow metric

> After quality filtering, a median of [value] paired-end reads per sample remained (range [min]–[max]) (**Fig. 1**, **Table 1**).

### Non-significant result (report fully)

> No significant difference in [metric] was observed between [groups] (median difference [value], 95% CI [low, high]; q = [value]) (**Fig. 3c**).

## Phrasing: Allowed vs Forbidden

| Allowed (Results) | Forbidden (belongs in Discussion) |
|-------------------|-----------------------------------|
| "X increased 2.1-fold in treated samples" | "X drives community restructuring" |
| "Species Y was enriched in group A (q < 0.05)" | "suggesting that species Y causes..." |
| "Assembly N50 was 12.4 kb (median)" | "indicating high-quality genomes" |
| "The model achieved an AUC of 0.82" | "demonstrating strong predictive power for disease" |

## Figure and Table References

- First mention: `(Fig. 1a)` or `(Table 1)` — bold labels optional per journal style
- Multiple panels: `(Fig. 2a–c)`
- Supplementary: `(Extended Data Fig. 1)`, `(Supplementary Table 2)`
- Do not reference figures/tables not present in the project unless user provides labels

## Missing Information Report

```markdown
# Missing Information Report

The following items could not be verified from project outputs. They are **excluded** from the Results section.

## [Category, e.g. Effect sizes for Fig. 2]

### [Item name]
- **Missing:** [Specific statistic, n, or label]
- **Searched:** [Files and outputs inspected]
- **Blocks:** [Which Results sentence cannot be written]
- **Required from user:** [Exact file, table export, or statistic]

## Summary
| Priority | Count | Action |
|----------|-------|--------|
| Critical | N     | Required for primary Results claims |
| Optional | N     | Improves completeness |
```

## Evidence Table (optional appendix)

```markdown
# Results Evidence Table

| Results claim | Source | Location | Confidence |
|---------------|--------|----------|------------|
| Median reads 1.2M | QC table | tables/qc_summary.tsv | Verified |
| q = 0.003 for species X | DE output | results/de.tsv:142 | Verified |
| Fold-change for Fig. 1b | not tabulated | — | Unknown |
```

## Statistical Reporting Checklist

Include when available in outputs:

- [ ] Test name matches output file
- [ ] Sample size per group
- [ ] Effect size (fold-change, difference, OR, R², etc.)
- [ ] Uncertainty (95% CI, IQR, s.d., s.e.)
- [ ] Adjusted q-value if multiple testing was corrected
- [ ] Both significant and non-significant primary outcomes reported
