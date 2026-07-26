---
name: methods-writer
description: >-
  Generate publication-quality Materials and Methods from an existing project by
  inspecting code, notebooks, workflows, configs, and software versions. Use when
  the user asks for a Methods section, Materials and Methods, protocol write-up,
  or manuscript methods from project artifacts.
disable-model-invocation: true
---

# Methods Writer

Generate publication-quality Materials and Methods directly from an existing project. Inspect code, notebooks, workflow definitions, scripts, configuration files, software versions and documentation. Produce a complete Methods section suitable for a Nature-style manuscript. Include software, parameters, databases, preprocessing, quality control, statistical analyses and workflow description. Never invent missing methods; instead report missing information separately.

Follow project rules: **scientific-integrity**, **nature-writing-style**, **reproducibility**, **statistical-analysis**, **missing-data-policy**.

## Workflow

Copy and track progress:

```
Methods generation:
- [ ] Step 1: Map project artifacts
- [ ] Step 2: Extract evidence-backed method facts
- [ ] Step 3: Read method-decision.md (if present)
- [ ] Step 4: Draft Methods section
- [ ] Step 5: Write Missing Information Report
- [ ] Step 6: Self-check (no invented content)
```

### Step 1: Map project artifacts

Search the repository systematically:

| Source type | Typical locations | What to extract |
|-------------|-------------------|-----------------|
| Workflows | `Snakefile`, `*.smk`, `nextflow.config`, `*.nf`, WDL/CWL | Step order, inputs/outputs, resource hints |
| Scripts | `scripts/`, `bin/`, `*.sh`, `*.py`, `*.R` | Commands, flags, thresholds, file paths |
| Notebooks | `*.ipynb`, `*.qmd` | Analysis steps, parameters, plots (not figures text) |
| Config | `config/`, `*.yaml`, `*.yml`, `*.toml`, `*.json` | Parameters, reference paths, sample metadata rules |
| Environments | `environment.yml`, `conda-lock.yml`, `Dockerfile`, `requirements.txt` | Package names and pinned versions |
| SLURM | `*.sbatch`, `#SBATCH` headers | Compute context if relevant to reproducibility |
| Docs | `README*`, `docs/`, comments in entry points | Stated protocols, database versions |
| Decisions | `method-decision.md` | Locked choices and rationale |

Prefer **executed** artifacts (saved commands, logged versions, config values) over comments or stale README text.

### Step 2: Extract evidence-backed facts

For each methodological claim, record:

- **Claim** — what will appear in Methods prose
- **Source** — file path and line/command/config key
- **Confidence** — Verified / Inferred / Unknown

Rules:

- **Verified** — value appears explicitly in repo artifacts
- **Inferred** — deduced from code flow but not documented; mark Tentative in prose or omit until confirmed
- **Unknown** — move to Missing Information Report; **do not** write into Methods

Extract at minimum:

- Software name, version, and citation DOI/URL when available in env files or `--version` output
- Reference databases (name, version/build date, download source)
- Preprocessing and QC steps with thresholds
- Statistical tests, multiple-testing correction, effect-size reporting
- Random seeds, thread counts, and key parameters
- Input/output file types and sample inclusion criteria

Run commands when needed and permitted (`tool --version`, `conda list`, reading lockfiles). Do not guess versions.

### Step 3: Read method-decision.md

If `method-decision.md` exists, treat **Locked** entries as authoritative. Use **Tentative** entries with cautious wording or flag for user confirmation. **Open** entries belong in Missing Information Report.

### Step 4: Draft Methods section

Use the template in [methods-template.md](methods-template.md).

Writing rules (Nature-style):

- Past tense, concise, precise; no hype or unsupported claims
- Subsections in **logical workflow order** (sample → QC → analysis → statistics)
- One paragraph per major step; bundle minor defaults without hiding critical parameters
- Name software on first mention with version; cite in References subsection or inline per journal convention
- Report parameters that affect results (cutoffs, k-mer sizes, FDR method, model hyperparameters)
- Describe reproducibility: environment files, seeds, workflow entry point when verified

Omit subsections with no verified evidence. Do not pad with generic boilerplate.

### Step 5: Write Missing Information Report

Always deliver this **separate** document/section when anything required is unverified. Use the template in [methods-template.md](methods-template.md#missing-information-report).

For each gap: what is missing, where you looked, why Methods cannot state it safely, what the user must provide.

### Step 6: Self-check

Before delivering:

- [ ] Every parameter/version in Methods has a listed source
- [ ] No invented citations, DOIs, or database builds
- [ ] Statistical methods match project code/notebooks
- [ ] Missing Information Report covers all Unknown items
- [ ] Tentative items are labeled or excluded

## Deliverables

Default output (unless user specifies otherwise):

1. **Methods** — manuscript-ready prose
2. **Missing Information Report** — gaps and required user input
3. **Evidence table** (optional appendix) — claim → source mapping for author review

Save to user-requested path, or suggest `docs/methods.md` and `docs/methods-missing.md`.

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

- Output templates and subsection guidance: [methods-template.md](methods-template.md)
