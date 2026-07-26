# Dependency Report Template

## dependency-report.md

```markdown
# Dependency Report

**Date:** YYYY-MM-DD
**Skill:** verify-todo
**Overall:** Blocking | Recoverable (repair pending) | Valid
**Exit status:** failure | success

Task execution Status values (for tasks in the graph) must be only:
`TODO` | `READY` | `RUNNING` | `BLOCKED` | `FAILED` | `RECOVERABLE` | `COMPLETED` | `SKIPPED`.
**Graph artifacts:** [dependency-graph.md](dependency-graph.md) · [dependency-graph.mmd](dependency-graph.mmd)

## Summary
| Class | Count |
|-------|-------|
| Blocking | N |
| Recoverable | N |
| Informational | N |

---

## Blocking issues

### B-001: [Title]
- **Type:** circular dependency | missing dependency | deleted task | impossible order | unavailable inputs | mutually exclusive | unresolved chain | invalid graph structure | …
- **Affected tasks:** T-A, T-B
- **Explanation:**
- **Graph location:** edge `T-A → T-B` in dependency-graph.md / node in dependency-graph.mmd
- **Suggested fixes:**

## Recoverable issues

### R-001: [Title]
- **Affected tasks:**
- **Evidence:**
- **Suggested repair (@debug):**

## Informational issues

### I-001: [Title]
- **Notes:**

---

## Orphans
| Kind | ID / Path | Marked independent? |
|------|-----------|---------------------|
| Orphan task | T-8.0 | no |

## Missing / ambiguous references
| Declared by | Reference | Issue |
|-------------|-----------|-------|
| T-3.1 | T-9.9 | missing / points to >1 match |

## Input / output chain gaps
| Task | Required input | Expected producer | Status |
|------|----------------|-------------------|--------|

## Mutually exclusive / impossible outputs
| Task A | Task B | Conflict |
|--------|--------|----------|

## Unreachable tasks
| Task ID | Details |
|---------|---------|

## Project-state conflicts
| Dependency | Conflict | Details |
|------------|----------|---------|

---

## Why execution cannot proceed
[Only if Blocking — clear paragraph]

## Recommended corrective actions
1. …
2. …

## Parent propagation
**Invoked by:** prepare | prepare-prompt | do | prompt-orchestrator | other | user
**Propagate failure:** yes (Blocking) | no (Success)
**Additional skills invoked:** none (Blocking) | debug → prompt-orchestrator (Recoverable)

---

## Validation after repair (if applicable)
| Check | Result |
|-------|--------|
| Graph regenerated | yes / no |
| Re-validation | pass / fail |
| Remaining Blocking | N |
```

## Classification guide

| Class | Examples |
|-------|----------|
| **Blocking** | Cycles; missing ID; deleted/SKIPPED upstream; impossible order; exclusive outputs; unavailable inputs; invalid/disconnected structure where connectedness is required; unreachable required tasks |
| **Recoverable** | Soft orphan (link missing but ID clear); fixable metadata typo; redundant optional edge; formatting of Depends on |
| **Informational** | Explicitly independent orphans; unused optional tasks; stylistic ID naming |

## Success stub (no Blocking)

```markdown
# Dependency Report

**Overall:** Valid
**Exit status:** success

## Success criteria
- dependency graph successfully constructed;
- dependency graph validated;
- execution order determined;
- project is ready for execution.

**Artifacts:** dependency-graph.md, dependency-graph.mmd

## Execution order
| Wave | Tasks | Parallel |
|------|-------|----------|
| 1 | T-1.1 | no |
| 2 | T-2.1, T-2.2 | yes |

## Informational (non-blocking)
- …
```
