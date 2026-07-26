# Code Review Report Template

```markdown
# Code Review Report

**Date:** YYYY-MM-DD
**Scope:** [files / full project / git diff range]
**Reviewer mode:** Read-only

## Expected behavior baseline
[From user prompt, spec, method-decision.md, or documented assumptions]

**method-decision.md:** present | missing
**Decisions in scope:** [list Decision titles checked]

## Summary
| Severity | Count |
|----------|-------|
| Critical | N |
| Major | N |
| Minor | N |
| Suggestion | N |

**Overall assessment:** [Pass with issues / Fail / Consistent with spec / Insufficient spec to judge]

---

## Findings

### CRITICAL

#### CR-001: [Short title]
- **Description:**
- **Affected files:** `path/to/file.py` (lines X–Y)
- **Evidence:** [quote, trace, missing output]
- **Expected behavior:**
- **Recommended fix:** [advisory — not applied]

### MAJOR

#### MJ-001: [Short title]
- **Description:**
- **Affected files:**
- **Evidence:**
- **Expected behavior:**
- **Recommended fix:** [advisory — must not violate method-decision.md]

#### MJ-00N: Contradiction with method-decision.md
- **Description:** Implementation uses [X]; recorded Decision is [Y]
- **Affected files:** `…`
- **Evidence:** method-decision.md § …; code at …
- **Expected behavior:** Match recorded Decision (or update Decision via @verify-methods with user approval first)
- **Recommended fix:** Align code/config to Decision — do **not** suggest adopting a conflicting method

### MINOR

#### MN-001: ...

### SUGGESTION

#### SG-001: ...

---

## Correctness vs specification

| Requirement | Status | Evidence |
|-------------|--------|----------|
| [Req from spec] | Met / Partial / Missing | |
| [Decision from method-decision.md] | Aligned / Contradicts (→ Major) | |

## Duplication and complexity
| Location A | Location B | Notes |
|------------|------------|-------|

## Reproducibility and robustness
| Check | Result |
|-------|--------|
| Pinned dependencies | pass/fail |
| Input validation | pass/fail |
| Error handling | pass/fail |

---

## Conclusion

[If no issues: Implementation appears consistent with the supplied specification. This review cannot guarantee the absence of undiscovered defects.]

[If issues: Prioritized list of what must be fixed before relying on this implementation.]

## Review limitations
- [Untested paths, missing spec, runtime not executed]
```

## Severity guide

| Level | Examples |
|-------|----------|
| Critical | Wrong scientific result, silent data corruption, secret in repo |
| Major | Spec feature missing; no validation on required inputs; breaks pipeline; **code contradicts method-decision.md Decision** |
| Minor | Edge case mishandled, unclear error message |
| Suggestion | Refactor opportunity (only if still consistent with method-decision.md) |

## Monitor failure review (scoped template)

When invoked from `@monitor`:

```markdown
## Failure context
- **Job ID:**
- **Failed command/script:**
- **Log excerpt:**

## Operational vs code issues
| Finding | Type | Monitor may auto-fix? |
|---------|------|----------------------|
| OOM | operational | yes — resubmit more mem |
| Wrong algorithm in script | code | no — report only |

## Recovery recommendation
- [ ] Operational retry sufficient
- [ ] Code fix required before retry
- [ ] Escalate to @prompt-orchestrator (architectural/methodological)
```
