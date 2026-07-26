# Do Progress Report Template

Human-readable summary after each synchronization cycle. Prefer prose + short tables.

## docs/do-progress.md

```markdown
# Execution Progress

**Updated:** YYYY-MM-DD HH:MM
**Cycle:** N
**Phase:** Main loop — post-sync
**Dependency graph:** Valid (see dependency-graph.md) | Blocking — stopped

## At a glance
We finished **[list]**. Still running: **none** (sync barrier). Next up: **[tasks]**.

## COMPLETED this cycle
| Task | Notes |
|------|-------|
| T-2.1 | Dataset audit Ready with warnings |

## Still to run
| Task | Why waiting |
|------|-------------|
| T-3.1 | Waiting on T-2.2 |

## BLOCKED
| Task | Blocker |
|------|---------|
| T-2.2 | Missing FASTQ S12 |

## FAILED / RECOVERABLE (this cycle)
| Task | Stage | Action |
|------|-------|--------|
| — | — | — |

## Parallelism
- Resource estimates (this batch): [CPU / mem / runtime per task]
- Cap applied: [e.g. 64 CPU, 256G available → launched 2 of 5 READY]
- Launched in parallel: T-3.1 ‖ T-3.2
- Deferred (resource wait): T-3.3, T-3.4
- Policy: throughput over agent count; Sync: all finished before selecting next wave

## Recovered failures
- Job 12346 OOM → resubmitted 128G (monitor)

## Remaining work (estimate)
- ~K executable after blockers clear; ~M total incomplete

## Next actions
1. Resolve T-2.2 inputs
2. Continue `@do` cycle (1 project-auditor → 2 verify-todo → … → 5 sync)
```
