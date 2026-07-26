# Task Execution Report Template

## docs/execution/\<task-id\>.md

Per-task report for `@prompt-orchestrator`. **Never** write `docs/execution-report.md` from this skill — that file is owned exclusively by `@do`.

Register every written report in `artifact-registry.md` / `docs/artifact-registry.md`.

```markdown
# Task Execution Report

**Date:** YYYY-MM-DD
**Report path:** docs/execution/<task-id>.md
**Assigned task ID:** [e.g. T-1.2]
**Assigned task:** [Single-task summary — not a project wave]
**Overall status:** Success | Partial | Failed
**Scope:** This report covers **one** assigned task only (`@prompt-orchestrator` is not a project scheduler; `@do` owns multi-task order and `docs/execution-report.md`)

## Executive summary
[3–5 sentences: what was planned for this task, what completed, what remains for this task]

---

## Phase 1 — Assigned-task understanding

| Field | Value |
|-------|-------|
| Objective | |
| Requested outputs | |
| Required inputs | |
| Deliverables | |
| Computational complexity | Low / Medium / High |
| Required datasets | |
| Risks identified | |
| Within-task deps only | [no project TODO graph walk] |

**Ambiguities resolved:** [context used or "none"]
**User clarifications requested:** [none / list]
---

## Phase 1b — Governance

| Field | Value |
|-------|-------|
| Governance | rules \| skills \| both |
| Applied rules | [names from `.cursor/rules/`] |
| Skills required? | yes / no |
| Selection principle | minimum necessary |

---

## Phase 2 — Execution plan

| Step | Skill / action | Depends on | Parallel? | Expected output |
|------|----------------|------------|-----------|-----------------|
| 1 | [discovered skill or "rules-only"] | — | no | … |

**Reused existing outputs:** [paths — skipped work]

**Validation strategy:** [checks planned]

---

## Phase 0/3 — Discovered registry and delegation

### Discovered skills (complete at run time)
| Name | Path | Responsibility (from metadata) | Selected? |
|------|------|--------------------------------|-----------|
| … | `.cursor/skills/…` | … | yes / no |

### Delegated skills (minimum necessary)
| Skill | Invoked | Reason | Deliverable | Status |
|-------|---------|--------|-------------|--------|
| … | yes | … | … | complete |

*(Empty table if governance = rules only.)*

---

## Phase 4 — Execution environments

| Step | Environment | Details |
|------|-------------|---------|
| Download | SLURM | sbatch scripts/sbatch/get_data.sbatch |
| Audit | Local | read-only inspection |
| Snakemake QC | Snakemake | `snakemake qc -c 32` |

### Software used
| Tool | Version | Source |
|------|---------|--------|
| datasets | 16.20.0 | conda env |

---

## Phase 5 — Monitoring summary

| Task | Started | Completed | Notes |
|------|---------|-----------|-------|
| SLURM job 12345 | yes | yes | exit 0 |

**Log paths:** data/logs/download.log, logs/get_data_12345.out

---

## Phase 6 — Recovery actions

| Failure | Cause | Recovery attempted | Outcome |
|---------|-------|-------------------|---------|
| prefetch timeout | network | retry with datasets CLI | recovered |
| OOM on assembly | 32G insufficient | resubmit 64G sbatch | recovered |

**Unrecoverable failures:**

| Failure | Cause | Action required |
|---------|-------|-----------------|
| — | — | — |

---

## Phase 7 — Validation

| Check | Result | Evidence |
|-------|--------|----------|
| Expected outputs exist | pass | data/raw/*.fastq.gz |
| Non-empty files | pass | |
| Reports valid | pass | acquisition_report.md |
| Critical errors | none | |

**todo.md updated:** yes — @synchronize-todo
**method-decision.md updated:** yes — Kraken2 pipeline logged

---

## Phase 8 — Outcomes

### Completed tasks
- [x] Data acquisition for SRR2345678
- [x] Dataset audit — Ready with warnings

### Generated outputs
| Output | Path |
|--------|------|
| Raw FASTQ | data/raw/ |
| Data prep report | docs/data-preparation-report.md |
| Workflow diagram | docs/workflow-architecture.md |

### Remaining issues
| Priority | Issue | Recommendation |
|----------|-------|----------------|
| P1 | 3 low-depth samples | Exclude or re-sequence per dataset-audit |

### Recommendations
1. Run Snakemake preprocessing per todo T-3.1
2. ...

---

## Parallel execution log

| Subagent / job | Task | Started | Merged at step |
|----------------|------|---------|----------------|
| explore-1 | visualize-architecture | T+0 | Step 4 |
| explore-2 | verify-methods | T+0 | Step 4 |

---

## Execution decisions recorded

| Decision | Rationale | Logged in |
|----------|-----------|-----------|
| SLURM 32 CPU download | slurm-execution-policy | method-decision.md |
| Skip re-download SRR123 | validated manifest | data-preparation-report.md |

**Report path:** docs/execution/<task-id>.md
**Artifact registry:** registered in docs/artifact-registry.md
```

## Status definitions

| Overall status | When |
|----------------|------|
| **Success** | All planned steps complete; validation pass |
| **Partial** | Some steps complete; non-blocking issues remain |
| **Failed** | Mandatory step failed; recovery exhausted |

## Minimal report (small tasks)

For single-skill requests, use abbreviated sections: Executive summary, Delegated skills, Validation, Outcomes, Generated outputs.
