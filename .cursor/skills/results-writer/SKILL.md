---
name: results-writer
description: >-
  Generate publication-quality Results from figures, tables, statistical outputs,
  and workflow results. Use when the user asks for a Results section, manuscript
  results, or narrative summary of findings from project outputs.
disable-model-invocation: true
---

# Results Writer

Generate a publication-quality Results section from figures, tables, statistical outputs and workflow results. Describe only supported observations without interpretation. Organize results into logical subsections, reference figures and tables, report statistics appropriately, and avoid speculative conclusions.

Follow project rules: **scientific-integrity**, **nature-writing-style**, **statistical-analysis**, **missing-data-policy**.

## Workflow

Copy and track progress:

```
Results generation:
- [ ] Step 1: Inventory result artifacts
- [ ] Step 2: Extract verified observations
- [ ] Step 3: Map figures and tables
- [ ] Step 4: Draft Results section
- [ ] Step 5: Write Missing Information Report
- [ ] Step 6: Self-check (observations only, no speculation)
```

### Step 1: Inventory result artifacts

Search for outputs that support Results prose:

| Source type | Typical locations | What to extract |
|-------------|-------------------|-----------------|
| Figures | `figures/`, `plots/`, `*.pdf`, `*.svg`, notebook outputs | Panel labels, axes, trends, sample sizes shown |
| Tables | `tables/`, `*.tsv`, `*.csv`, `*.xlsx`, LaTeX/HTML tables | Counts, summaries, test statistics |
| Statistics | `results/`, `stats/`, `*.json`, model logs, R/Python outputs | Test names, effect sizes, CIs, q-values |
| Workflow outputs | `results/`, pipeline summaries, QC reports | Pass/fail counts, yield metrics, assembly stats |
| Notebooks | `*.ipynb`, `*.qmd` | Printed outputs, summary cells (not code assumptions) |
| Logs | `logs/`, SLURM `.out` files | Completed step metrics when explicitly logged |

Read actual files. Do not infer values from figure filenames alone.

### Step 2: Extract verified observations

For each observation, record:

- **Observation** — factual statement for Results prose
- **Source** — file path, figure panel, table row/column, or output line
- **Confidence** — Verified / Inferred / Unknown

Rules:

- **Verified** — number or pattern appears explicitly in artifact
- **Inferred** — visually estimated or deduced; do not use unless user accepts approximation; prefer Missing Information Report
- **Unknown** — exclude from Results; list in Missing Information Report

Report statistics per **statistical-analysis** rule:

- Effect sizes and confidence intervals when available — not p-values alone
- Adjusted q-values when multiple testing correction was applied
- Sample sizes (n) alongside group comparisons
- Exact test names only when documented in outputs

### Step 3: Map figures and tables

Build a reference map before writing:

| Label | File / path | Main message (observational) |
|-------|-------------|------------------------------|
| Fig. 1a | figures/qc_yield.pdf | ... |
| Table 1 | tables/sample_summary.tsv | ... |

Use consistent labels (`Fig. 1`, `Extended Data Fig. 1`, `Table 1`) matching user or journal convention. If labels are not assigned, propose a logical order and state the mapping.

Reference each figure/table at first mention in the relevant subsection.

### Step 4: Draft Results section

Use the template in [results-template.md](results-template.md).

Writing rules (Nature-style Results):

- **Past tense** for completed analyses
- **Observations only** — no biological mechanism, no "suggesting that", no "indicating a role for"
- Order subsections **logically** (QC → primary endpoint → secondary/exploratory)
- Lead with magnitude and direction, then uncertainty (CI, IQR, s.d.) and test statistics
- One main idea per paragraph; avoid repeating Methods detail
- Use `(Fig. 1a)`, `(Table 1)` parenthetical references
- Distinguish **primary** vs **exploratory** results when the project documents this

**Forbidden in Results:**

- Interpretation, hypothesis framing, causal language
- Speculative conclusions or generalizations beyond the data
- Values not traceable to an artifact
- Cherry-picked subsets not defined in outputs

### Step 5: Write Missing Information Report

When statistics, figure labels, sample sizes, or key numbers cannot be verified, deliver a separate report. Use the template in [results-template.md](results-template.md#missing-information-report).

Do not substitute plausible numbers. Stop short of writing unsupported sentences.

### Step 6: Self-check

Before delivering:

- [ ] Every number in Results has a listed source
- [ ] No interpretive or causal language
- [ ] Figures and tables referenced correctly
- [ ] Statistics reported with appropriate detail (not p alone)
- [ ] Missing Information Report covers all Unknown items

## Deliverables

Default output (unless user specifies otherwise):

1. **Results** — manuscript-ready prose
2. **Missing Information Report** — gaps and required user input
3. **Figure/table reference map** (optional appendix)

Save to user-requested path, or suggest `docs/results.md` and `docs/results-missing.md`.

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

- Output templates and phrasing patterns: [results-template.md](results-template.md)
