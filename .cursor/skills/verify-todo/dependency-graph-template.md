# Dependency Graph Templates

Artifacts produced by `@verify-todo` only. Other skills read these; they do not rewrite them.

## dependency-graph.md

```markdown
# Dependency Graph

**Date:** YYYY-MM-DD
**Source of truth:** `@verify-todo`
**Status:** Valid | Invalid (see dependency-report.md)
**Mermaid file:** [dependency-graph.mmd](dependency-graph.mmd)

## Tasks

Task **Status** must be one of: `TODO` | `READY` | `RUNNING` | `BLOCKED` | `FAILED` | `RECOVERABLE` | `COMPLETED` | `SKIPPED` (no other execution states).

| Task ID | Title | Status | Prerequisites | Dependents | Terminal | Independent |
|---------|-------|--------|---------------|------------|----------|-------------|
| T-1.1 | Env setup | COMPLETED | — | T-2.1 | no | no |
| T-2.1 | Dataset audit | READY | T-1.1 | T-2.2, T-3.1 | no | no |
| T-9.9 | Optional note | TODO | — | — | yes | yes (marked) |

## Edges (prerequisite → dependent)
| Prerequisite | Dependent | Declared in |
|--------------|-----------|-------------|
| T-1.1 | T-2.1 | todo.md, todo/T-2.1.md |

## Execution order (waves)
| Wave | Task IDs | Parallelizable |
|------|----------|----------------|
| 1 | T-1.1 | no |
| 2 | T-2.1 | no |
| 3 | T-2.2, T-3.1 | yes |

## Parallelizable branches
| Wave | Branch tasks | Notes |
|------|--------------|-------|
| 3 | T-2.2 ‖ T-3.1 | No edge between them; both need T-2.1 |

## Blocking tasks
| Task ID | Blocks |
|---------|--------|
| T-2.1 | T-2.2, T-3.1 |

## Terminal tasks
| Task ID | Notes |
|---------|-------|
| T-5.1 | No dependents |

## Orphan tasks
| Task ID | Reason | Marked independent? |
|---------|--------|---------------------|
| T-8.0 | No edges | no → Blocking or Recoverable |

## Missing references
| Declared by | Missing ID | Kind |
|-------------|------------|------|
| T-3.1 | T-9.9 | Depends on |

## Unreachable tasks
| Task ID | Why unreachable |
|---------|-----------------|

## Input → producer map
| Consumer | Required input | Producer task | Status |
|----------|----------------|---------------|--------|
| T-3.1 | `data/raw/*.fastq.gz` | T-2.0 | ok / missing |
```

## dependency-graph.mmd

Raw Mermaid for rendering (no markdown fence required in the `.mmd` file):

```text
flowchart TD
  T11["T-1.1 Env setup"]
  T21["T-2.1 Dataset audit"]
  T22["T-2.2 Resolve FASTQ"]
  T31["T-3.1 Profiling"]

  T11 --> T21
  T21 --> T22
  T21 --> T31

  classDef terminal fill:#e8f5e9
  classDef blocking fill:#fff3e0
  classDef orphan fill:#ffebee
  classDef missing stroke:#c62828,stroke-dasharray: 5 5

  class T22,T31 terminal
```

### Mermaid conventions

| Element | Representation |
|---------|----------------|
| Task node | `TID["T-x.y Title"]` |
| Prerequisite edge | `A --> B` means A must finish before B |
| Missing reference | dashed node + dashed edge |
| Parallel wave | same rank / subgraph `wave_N` optional |

```text
subgraph wave_3["Wave 3 (parallel)"]
  T22
  T31
end
```

Regenerate the entire `.mmd` file on every graph change — do not patch partially.
