# Prepare Report Template

## prepare-report.md

```markdown
# Prepare Report

**Date:** YYYY-MM-DD
**Mode:** Planning only (no analyses executed; **no new tasks created**)
**Upstream:** `@prepare-prompt` (tasks) → `@verify-todo` (graph) → `@prepare`
**Readiness:** Ready for `@do` | Blocked | Partial — missing tasks → use `@prepare-prompt`

## Project summary
[2–4 sentences: objectives, current status, next milestone]

## TODO sync summary
| Metric | Count |
|--------|-------|
| Tasks in todo.md | N |
| Subtask files in ./todo/ | N |
| Linked (1:1) | N |
| Orphaned todo.md entries | N |
| Orphaned ./todo/ files | N |

### Orphaned TODO entries (in todo.md, no file)
| Task ID | Title |
|---------|-------|

### Orphaned TODO files (in ./todo/, no todo.md reference)
| Path | Suggested task ID |
|------|-------------------|

### Inconsistencies (todo.md vs repository)
| Issue | Evidence | Recommendation |
|-------|----------|----------------|
| T-3.1 marked TODO; outputs exist | `results/...` | Sync status before @do |

---

## Execution graph summary

**Source of truth:** [@verify-todo](../verify-todo/SKILL.md) artifacts — do not rebuild.

| Artifact | Path |
|----------|------|
| Graph | `docs/dependency-graph.md` (or `dependency-graph.md`) |
| Mermaid | `docs/dependency-graph.mmd` (or `dependency-graph.mmd`) |
| Validation | `docs/dependency-report.md` — Overall: Valid |

Embed or link Mermaid from `dependency-graph.mmd` only; do not invent a second graph.

### Planned execution order (from verified graph + Phase 3 skills)
| Wave | Task IDs | Mode | Skills |
|------|----------|------|--------|
| 1 | T-1.1 | sequential | — |
| 2 | T-2.1 | sequential | `@dataset-auditor` |
| 3 | T-3.1, T-3.2 | **parallel** | `@verify-methods`, `@visualize-architecture` |
| 4 | checkpoint | — | validate outputs |

### Parallel execution opportunities
| Wave | Tasks | Why safe to parallelize |
|------|-------|-------------------------|
| 3 | T-3.1, T-3.2 | From dependency-graph.md wave 3 |

### Blocking tasks (from verified graph)
| Blocker | Blocks | Reason |
|---------|--------|--------|
| T-2.2 | T-3.1 | From graph / project state |

### Optional tasks
| Task ID | Notes |
|---------|-------|

### Checkpoints
| After wave | Validation |
|------------|------------|
| 2 | dataset-audit status ≠ Not ready |

---

## Assigned skills
| Task ID | Skills | Rules |
|---------|--------|-------|
| T-2.1 | `@dataset-auditor` | validation-first, missing-data-policy |

## Applied rules (project-wide)
- scientific-integrity, reproducibility, validation-first, …

## Expected outputs
| Task ID | Output path |
|---------|-------------|
| T-2.1 | `docs/dataset-audit.md` |

## Unresolved blockers
| ID | Blocker | Blocks `@do`? | Required from user |
|----|---------|---------------|--------------------|
| B1 | No ./todo/ for T-4.1 | yes | Create subtask doc or drop task |

## Recommended next action
1. Resolve blockers listed above (if any)
2. Invoke `@do` to execute Wave 1 onward
3. Use `@monitor` for long-running SLURM waves

---

## Readiness checklist
| Check | Pass |
|-------|------|
| Every TODO has implementation plan / subtask doc | ✓/✗ |
| Every subtask has required skills | ✓/✗ |
| Every dependency satisfied or Status BLOCKED | ✓/✗ |
| No execution cycles | ✓/✗ |
| No missing references | ✓/✗ |
```

## Subtask document skeleton (./todo/)

```markdown
# T-2.1 — Dataset audit

## Description
[User / existing prose — preserve]

## Done when
Audit status is Ready or Ready with warnings; exclusion list finalized.

## Execution metadata

| Field | Value |
|-------|-------|
| Task ID | T-2.1 |
| Skills | `@dataset-auditor` |
| Rules | validation-first, missing-data-policy |
| Inputs | `data/metadata/`, `data/raw/`, `data/manifests/` |
| Outputs | `docs/dataset-audit.md` |
| Depends on | T-1.1 |
| Parallel with | — |
| Environment | local |
| Status | TODO |
```

## Cycle detection

Owned exclusively by `@verify-todo`. Prepare must not re-detect cycles; if `dependency-report.md` is not Valid, prepare must already have stopped at Phase 0.
