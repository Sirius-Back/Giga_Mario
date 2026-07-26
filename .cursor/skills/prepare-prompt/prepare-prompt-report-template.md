# Prepare-Prompt Report Template

## prepare-prompt-report.md

```markdown
# Prepare-Prompt Report

**Date:** YYYY-MM-DD
**Mode:** Planning only (tasks not executed)
**User prompt (summary):** …
**verify-todo:** success | failure

## Created tasks
| Task ID | Path | Status | Skills |
|---------|------|--------|--------|
| T-2.1 | todo/T-2.1-dataset-audit.md | READY | dataset-auditor |

## Modified tasks
| Task ID | Change |
|---------|--------|
| T-1.1 | Added Dependent T-2.1 |

## Reused / not duplicated
| Existing ID | Relationship to prompt |
|-------------|------------------------|
| T-2.0 | Partial overlap — extended instead of new task |

## Detected dependencies
| Task | Prerequisites | Dependents |
|------|---------------|------------|
| T-2.1 | T-1.1 | T-2.2 |

## Required skills
| Skill | Why |
|-------|-----|
| dataset-auditor | Dataset QC deliverable |

## Required rules
| Rule | Why |
|------|-----|
| validation-first | Pre-flight checks |
| task-status | Status vocabulary |

## Suggested execution order
| Wave | Tasks | Notes |
|------|-------|-------|
| 1 | T-2.1 | After verify-todo Valid |
| 2 | … | |

**Next action:** `@prepare` (sync metadata, assign skills/rules, execution plan — does not create tasks) then `@do`.

## verify-todo
- Artifacts: dependency-graph.md, dependency-graph.mmd
- Blocking issues: none | see dependency-report.md
```
