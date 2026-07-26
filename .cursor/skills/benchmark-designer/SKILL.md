---
name: benchmark-designer
description: >-
  Design scientifically rigorous benchmarking experiments with datasets, baselines,
  metrics, statistics, ablations, and reproducibility requirements. Use when the
  user asks for benchmark plans, method comparison studies, evaluation protocols,
  or ablation/robustness study design.
disable-model-invocation: true
---

# Benchmark Designer

Design scientifically rigorous benchmarking experiments. Define datasets, baselines, evaluation metrics, statistical tests, ablation studies, robustness analyses, sensitivity analyses, computational resource measurements, reproducibility requirements, and expected outputs according to current best practices.

Follow project rules: **scientific-integrity**, **statistical-analysis**, **reproducibility**, **validation-first**, **slurm-execution-policy**, **method-decision-tracking**, **missing-data-policy**.

## Workflow

Copy and track progress:

```
Benchmark design:
- [ ] Step 1: Define research question and claims
- [ ] Step 2: Specify datasets and splits
- [ ] Step 3: Select baselines and proposed method
- [ ] Step 4: Define metrics and statistical comparison plan
- [ ] Step 5: Plan ablations, robustness, and sensitivity analyses
- [ ] Step 6: Specify compute, reproducibility, and outputs
- [ ] Step 7: Record decisions in method-decision.md
- [ ] Step 8: Report gaps
```

### Step 1: Define research question and claims

Clarify before designing:

- **Primary claim** — what performance difference or property is being demonstrated
- **Scope** — task, data modality, generalization target
- **Success criteria** — minimum effect size or metric threshold (pre-specified)
- **Non-goals** — what the benchmark will not claim

One primary endpoint; secondary/exploratory analyses labeled explicitly.

### Step 2: Specify datasets and splits

Document for each dataset:

| Field | Requirement |
|-------|-------------|
| Name / source | Public accession or project path |
| Version / download date | Pinned reference |
| Sample size | n per split |
| Split strategy | Train/val/test, cross-validation, holdout — fixed and documented |
| Inclusion/exclusion | QC rules, filtering thresholds |
| Leakage prevention | No overlap between splits; group-aware splits when needed |

Prefer public benchmarks with established splits when available. If custom data, justify and document split seed.

### Step 3: Select baselines and proposed method

- **Baselines:** strong, widely used, fairly tuned methods — not strawmen
- **Proposed method:** exact variant under test
- **Hyperparameters:** search budget equal across methods when feasible; document tuning protocol
- **Implementation:** software versions, containers, random seeds

Log choices in `method-decision.md` (Alternatives, Justification, Status).

### Step 4: Define metrics and statistical comparison plan

Per **statistical-analysis** rule:

| Element | Specification |
|---------|---------------|
| Primary metric | One pre-registered endpoint |
| Secondary metrics | Labeled exploratory unless pre-specified |
| Effect size | Define before running (Δ AUC, relative improvement, etc.) |
| Uncertainty | 95% CI, bootstrap, or CV variance |
| Tests | Paired vs unpaired; parametric vs non-parametric per design |
| Multiple testing | Correction method when comparing across datasets/metrics |
| Reporting | Effect size + CI + adjusted q-values — not p alone |

Define **minimum practically meaningful difference** where possible.

### Step 5: Plan ablations, robustness, and sensitivity analyses

Use the template in [benchmark-template.md](benchmark-template.md).

**Ablations** — isolate component contributions (remove/replace one factor at a time)

**Robustness** — performance under perturbations:
- Subsampled data, noise injection, parameter jitter
- Alternative preprocessing or reference databases
- Different random seeds or cross-validation folds

**Sensitivity** — key hyperparameters, thresholds, subset definitions

Label each as **pre-specified** or **exploratory**. Exploratory analyses cannot support primary claims.

### Step 6: Specify compute, reproducibility, and outputs

**Computational measurement** (per **slurm-execution-policy** for HPC):

- Wall time, CPU/GPU hours, peak memory
- Hardware/software environment
- Fair comparison conditions (same hardware, thread counts)

**Reproducibility** (per **reproducibility** rule):

- Fixed seeds, pinned dependencies, config files
- Entry command or workflow target
- Expected output directory layout

**Expected outputs:**

- Metric tables (CSV/TSV), summary JSON, plots
- Per-run logs, SLURM logs, version manifest
- Leaderboard or comparison table schema

### Step 7: Record decisions in method-decision.md

Append benchmark design decisions (datasets, baselines, metrics, tests) with Status **Locked** if user-approved, else **Tentative**.

### Step 8: Report gaps

If datasets, baselines, or metrics cannot be verified or accessed, produce a **Benchmark Gap Report** — do not assume availability.

## Deliverables

Default output:

1. **Benchmark Protocol** — full design document
2. **Experiment matrix** — runs × methods × datasets × seeds
3. **Statistical analysis plan**
4. **Output schema** — files and tables to generate
5. **Benchmark Gap Report** (if needed)

Save to user-requested path, or suggest `docs/benchmark-protocol.md` and `docs/benchmark-gaps.md`.

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

- Templates and experiment matrices: [benchmark-template.md](benchmark-template.md)
