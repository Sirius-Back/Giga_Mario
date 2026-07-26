# Benchmark Design Templates

## Benchmark Protocol Template

```markdown
# Benchmark Protocol: [Study name]

## 1. Objective
- **Research question:**
- **Primary claim:**
- **Success criteria:**

## 2. Datasets
| Dataset | Source | Version | n (train/val/test) | Split | Notes |
|---------|--------|---------|---------------------|-------|-------|
| ...     | ...    | ...     | ...                 | ...   | ...   |

## 3. Methods under comparison
| Method | Role | Implementation | Tuning budget | Version |
|--------|------|----------------|---------------|---------|
| Proposed | Test | ... | ... | ... |
| Baseline A | Strong baseline | ... | Equal | ... |

## 4. Metrics
| Metric | Type (primary/secondary) | Definition | Higher/lower better |
|--------|--------------------------|------------|---------------------|
| ...    | Primary                  | ...        | ...                 |

## 5. Statistical comparison plan
- **Primary comparison:** [method A vs B on metric X]
- **Design:** [paired/unpaired; blocked by dataset]
- **Test:** [e.g., Wilcoxon signed-rank across folds]
- **Effect size:** [definition]
- **Multiple testing correction:** [BH-FDR across k comparisons]
- **Reporting:** point estimate, 95% CI, q-value

## 6. Ablation studies
| ID | Change from full model | Hypothesis | Metrics |
|----|------------------------|------------|---------|
| A1 | Remove module X | ... | Primary + secondary |

## 7. Robustness analyses
| ID | Perturbation | Levels | Purpose |
|----|--------------|--------|---------|
| R1 | Subsample 50% reads | 3 seeds | Stability to data loss |

## 8. Sensitivity analyses
| Parameter | Range | Default | Label |
|-----------|-------|---------|-------|
| k-mer size | 21, 31, 41 | 31 | Exploratory |

## 9. Computational benchmarking
- **Measured:** wall time, peak RAM, CPU-hours
- **Environment:** [hardware, SLURM resources]
- **Fairness:** same threads, same input subsets

## 10. Reproducibility
- **Seeds:** [list]
- **Environment file:** `environment.yml`
- **Entry point:** `snakemake benchmark_all -c 32`
- **Config:** `config/benchmark.yaml`

## 11. Expected outputs
| Output | Path | Description |
|--------|------|-------------|
| Summary table | `results/benchmark/metrics.tsv` | All methods × datasets |
| Stats report | `results/benchmark/stats.json` | Tests and CIs |
| Figures | `figures/benchmark/` | Comparison plots |
```

## Experiment Matrix Template

```markdown
# Experiment Matrix

Rows = unique runs. Include all seeds and ablations.

| run_id | dataset | method | variant | seed | split | status |
|--------|---------|--------|---------|------|-------|--------|
| 001 | DS1 | baseline_a | full | 42 | fold0 | planned |
| 002 | DS1 | proposed | full | 42 | fold0 | planned |
| 003 | DS1 | proposed | ablation_A1 | 42 | fold0 | planned |
```

## Statistical Analysis Plan (SAP) Snippet

```markdown
## Pre-specified primary analysis
Compare [proposed] vs [baseline] on [primary metric] across [n datasets]
using [test]. Report median difference with 95% CI and BH-adjusted q-values.

## Pre-specified secondary analyses
[List or "none"]

## Exploratory analyses (not for primary claims)
[List sensitivity/robustness runs]
```

## Best-Practice Checklist

### Design
- [ ] Primary endpoint named before experiments
- [ ] Baselines are strong and fairly tuned
- [ ] Splits prevent leakage (group/sample level)
- [ ] Sample sizes justified or power discussed

### Statistics
- [ ] Appropriate test for design and distribution
- [ ] Multiple testing correction defined
- [ ] Effect sizes and CIs planned
- [ ] Non-significant primary outcomes will be reported

### Rigor
- [ ] Ablations isolate single factors
- [ ] Robustness covers realistic perturbations
- [ ] Seeds and versions pinned
- [ ] Compute measured under fair conditions

### Integrity
- [ ] No post-hoc metric switching
- [ ] Exploratory analyses labeled
- [ ] Gaps documented when data/methods unavailable

## Benchmark Gap Report Template

```markdown
# Benchmark Gap Report

## [Item, e.g. Baseline B implementation]
- **Missing:** [What is unavailable]
- **Searched:** [Project paths, docs]
- **Impact:** [Which comparison or claim is blocked]
- **Required from user:** [Data, software, decision]
```

## Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Strawman baseline | Include well-tuned standard tool |
| Data leakage | Group-aware CV; document split logic |
| Metric shopping | Pre-register primary metric |
| Unfair compute | Match threads/hardware; report cost |
| Seed of one | Multiple seeds; report variance |
| p-only reporting | Add effect size and CI |
