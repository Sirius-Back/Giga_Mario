# Do-Fast Execution Report Template

Final (and optionally mid-run cumulative) report for `@do-fast`.

## docs/execution-report.md

```markdown
# Execution Report

**Date:** YYYY-MM-DD
**Skill:** do-fast
**Overall status:** Success | Partial | Failed (Blocking) | Interrupted (checkpoint)

## Executive summary
[3–6 sentences: what ran, what completed, what blocked]

## COMPLETED tasks
| Task ID | Title | Cycle | Outputs |
|---------|-------|-------|---------|
| T-2.1 | … | 1 | docs/dataset-audit.md |

## SKIPPED tasks
| Task ID | Reason |
|---------|--------|
| T-8.0 | Status: SKIPPED |

## FAILED / RECOVERABLE tasks
| Task ID | Failed stage | Evidence | Retries |
|---------|--------------|----------|---------|
| T-3.1 | code-review | Critical CR-001 | 3 |

## Dependency graph summary
- **Artifacts:** dependency-graph.md, dependency-graph.mmd
- **Last verify-todo:** Valid | Blocking
- **Terminal reason (if stopped):** Blocking issue B-00N / all reachable complete

## Project audit summary
| When | Path | Critical P0 |
|------|------|-------------|
| Phase 0 | docs/project-audit.md | N |
| Post T-2.1 | … | 0 |

## Code review summary
| Task | Result | Report |
|------|--------|--------|
| T-2.1 | Pass | docs/code-review.md |

## Monitoring summary
| Task | Duration | Recoveries | Outcome |
|------|----------|------------|---------|
| T-3.1 | 2h default | 1 OOM | success |

## Recovered failures
| Task | Failure | Recovery | Outcome |
|------|---------|----------|---------|

## Unresolved blockers
| ID | Description | Blocks |
|----|-------------|--------|

## Generated outputs
| Path | Producing task |
|------|----------------|

## Checkpoint information
| Field | Value |
|-------|-------|
| Last checkpoint | docs/do-fast-checkpoint.md |
| Timestamp | … |
| Last completed wave | 2 |
| Resume command | `@do-fast` (resumes from checkpoint) |

## Recommendations
1. …
2. …
```

## docs/do-fast-checkpoint.md

```markdown
# Do-Fast Checkpoint

**Timestamp:** YYYY-MM-DDTHH:MM:SS
**Cycle:** N
**verify-todo:** success
**Completed tasks:** T-1.1, T-2.1
**Failed tasks:** —
**Next executable (from graph):** T-2.2, T-3.1
**Artifact paths:**
- todo.md
- todo/
- docs/dependency-graph.md
- docs/dependency-graph.mmd
- docs/method-decision.md (if any)
- docs/execution-report.md
- docs/do-fast-progress.md
```
