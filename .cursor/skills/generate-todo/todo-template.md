# Todo.md Templates

## File Header

```markdown
# Project Todo

**Project:** [Name]
**Last updated:** YYYY-MM-DD
**Primary milestone:** [Description] — [date or TBD]

## Summary
[2–3 sentences: goal, current phase, next critical action]

## Milestones
| ID | Milestone | Target | Status | Depends on |
|----|-----------|--------|--------|------------|
| M1 | Environment ready | — | TODO | — |
| M2 | Dataset audit passed | — | TODO | M1, T-2.1 |
| M3 | Core analysis complete | — | TODO | M2 |
```

## Task Entry Format

Leaf tasks use checkboxes; metadata on sub-lines for parsing and tracking.

```markdown
## Phase 2 — Data acquisition & audit (→ M2)

### T-2 Data audit
- [ ] **T-2.1** Run dataset audit on raw metadata and FASTQ manifest
  - Priority: P0 | Complexity: S | Status: READY | Depends on: T-1.2
  - Outputs: `docs/dataset-audit.md`
  - Done when: Audit status is Ready or Ready with warnings documented; exclusion list finalized
  - Skill: `@dataset-auditor`

- [ ] **T-2.2** Resolve missing FASTQ paths for flagged samples
  - Priority: P0 | Complexity: M | Status: BLOCKED | Depends on: T-2.1
  - Blocked by: Samples S12, S47 missing from filesystem
  - Outputs: Updated `metadata/samplesheet.tsv`
  - Done when: Metadata ↔ file cross-check passes with zero orphan IDs
```

## Status Values

Use **only** these task execution statuses (rule: `task-status`):

| Status | Meaning |
|--------|---------|
| `TODO` | Not started; not yet ready to run |
| `READY` | Dependencies satisfied; eligible for execution |
| `RUNNING` | Currently executing |
| `BLOCKED` | Waiting on dependency, input, or external blocker |
| `FAILED` | Execution failed; not automatically recoverable in-place |
| `RECOVERABLE` | Failed or inconsistent; repair/retry expected |
| `COMPLETED` | Done criteria met (`[x]` checkbox) |
| `SKIPPED` | Out of scope, obsolete, or intentionally not run |

Do **not** introduce additional execution states.

Legacy mapping when importing old todos: `pending`→`TODO`, `in_progress`→`RUNNING`, `completed`→`COMPLETED`, `blocked`→`BLOCKED`, `cancelled`→`SKIPPED`.

## Complexity Guide

| Label | Typical scope |
|-------|----------------|
| S | Hours; single script or doc |
| M | 1–2 days; small pipeline stage |
| L | Several days; multi-step analysis |
| XL | Week+; major subsystem or benchmark suite |

Relative to project — document assumptions if calendar estimates requested.

## Priority Guide

| Label | Use |
|-------|-----|
| P0 | Blocks milestone or other P0 tasks |
| P1 | Core project deliverable |
| P2 | Enhancement, polish, optional robustness |

## Full Example (abbreviated)

```markdown
# Project Todo

**Project:** Metagenomic cohort study
**Last updated:** 2026-07-16
**Primary milestone:** M4 — Manuscript-ready figures

## Summary
16S cohort analysis from FASTQ to differential abundance. Phase 1 complete; data audit in progress.

## Milestones
| ID | Milestone | Status |
|----|-----------|--------|
| M1 | Conda env + config | COMPLETED |
| M2 | QC + audit passed | RUNNING |
| M3 | Taxonomic profiling done | TODO |
| M4 | Figures + draft Results | TODO |

---

## Phase 1 — Setup (→ M1)

### T-1 Environment
- [x] **T-1.1** Create `environment.yml` with pinned tools
  - Priority: P0 | Complexity: S | Status: COMPLETED
  - Outputs: `environment.yml`
  - Done when: `conda env create` succeeds; `snakemake --version` runs

---

## Phase 2 — Data audit (→ M2)

### T-2 Audit
- [ ] **T-2.1** Dataset audit
  - Priority: P0 | Complexity: S | Status: RUNNING | Depends on: T-1.1
  - Outputs: `docs/dataset-audit.md`
  - Done when: Readiness ≠ Not ready OR exclusions documented and approved

---

## Critical path
T-1.1 → T-2.1 → T-2.2 → T-3.1 (profiling) → T-5.1 (figures) → M4

## Planning Gaps
- **Sequencing depth threshold:** Not specified — using 100k reads (confirm with user)
- **Reference database:** Kraken2 index path not in config — blocks T-3.1
```

## Planning Gaps Template

```markdown
## Planning Gaps

| Gap | Blocks | Required from user |
|-----|--------|-------------------|
| Reference index path | T-3.1 | Path or download instructions |
| Primary endpoint for stats | T-4.2 | Column name + comparison pairs |
```

## Update Rules (living document)

1. Bump **Last updated** on every edit
2. Preserve task **IDs** permanently
3. Toggle `[ ]` → `[x]` only when Status is **COMPLETED**
4. Add new tasks; never renumber COMPLETED IDs
5. Record blockers inline with `Blocked by:` when Status is **BLOCKED**

## Mapping Inputs to Phases

| Input type | Maps to |
|------------|---------|
| Architecture data-flow diagram | Phase order + task dependencies |
| Spec acceptance criteria | Completion criteria on leaf tasks |
| User deadline | Milestone target dates |
| Existing code/results | Mark related tasks COMPLETED or SKIPPED |
