# Method Verification Templates

## Extended method-decision.md Entry

Includes standard fields plus verification metadata.

```markdown
### Kraken2 taxonomic classification (2026-07-15)

- **Decision:** Kraken2 with PlusPF database, confidence threshold 0.1
- **Alternatives considered:** MetaPhlAn 4, Centrifuge
- **Justification:** [From repo docs or inferred from workflow role]
- **Expected impact:** k-mer based classification; sensitive to database completeness
- **Confidence:** High | Medium | Low
- **Status:** Locked | Tentative | Open

#### Verification (verify-methods)
- **Detection:** Confirmed | Inferred | Unknown
- **Evidence source:** `config/kraken.yaml:12`, `Snakefile` rule `kraken_classify`
- **Parameters detected:** `--confidence 0.1`, db=`refs/pluspf/`
- **SOTA assessment:** Current | Acceptable | Outdated | Uncertain
- **Newer SOTA available:** Yes | No | Uncertain
- **Literature / benchmarks:** [Verified citation only, or "Not searched"]
- **Issues:** [Missing validation, questionable params, or —]
- **Recommendations:** [Specific improvement or —]
- **Updated:** 2026-07-15 — initial detection from repo scan

##### SOTA alternative (only if Newer SOTA available = Yes)
- **Alternative:** [name + version / protocol]
- **Advantages:** [evidence-based gains vs current Decision]
- **Migration cost:** Low | Medium | High — [effort, revalidation, compute, breakage, result-risk]
- **Auto-replaced:** No (never automatically replace the existing method)
- **Adoption:** Requires explicit user approval
```

## Detection Labels

| Label | Definition | Use in Methods? |
|-------|------------|-----------------|
| **Confirmed** | Explicit in code, config, env, or doc | Yes |
| **Inferred** | Deduced; not explicitly documented | Caution; flag for confirmation |
| **Unknown** | No traceable evidence | No — gap report only |

## SOTA Assessment Guide

| Rating | When to use |
|--------|-------------|
| **Current** | Standard tool/approach in recent (<3–5 yr) domain benchmarks or reviews |
| **Acceptable** | Still valid; newer alternatives exist with modest gains |
| **Outdated** | Known limitations; field has moved (e.g., superseded aligner, invalid stats) |
| **Uncertain** | No literature search, niche domain, or conflicting guidance |

Always note **assessment basis**: search date, sources consulted, or "expert inference only — verify."

## SOTA alternative record (required when superior approach exists)

Whenever a decision is reconstructed, determine whether newer SOTA alternatives have appeared.

If a superior approach exists, document it with:

| Field | Content |
|-------|---------|
| Alternative | Name and version/era |
| Advantages | Why it is superior for this use case (evidence-based) |
| Migration cost | Low / Medium / High + brief justification |
| Auto-replaced | Always **No** |

Never automatically replace the existing method in code, configs, or the **Decision** field.

## Methods Verification Report Template

```markdown
# Methods Verification Report

**Date:** YYYY-MM-DD
**Scope:** Full repository scan
**method-decision.md:** Updated (N new, M revised entries)

## Summary
| Metric | Count |
|--------|-------|
| Decisions detected | N |
| Confirmed | N |
| Inferred | N |
| Unknown (gaps) | N |
| SOTA: Outdated / questionable | N |
| Newer SOTA alternatives documented | N |
| Methods auto-replaced | 0 (must remain 0) |

## Decision inventory
| ID | Decision | Detection | Evidence | SOTA | Status |
|----|----------|-----------|----------|------|--------|
| D1 | Kraken2 + PlusPF | Confirmed | config/kraken.yaml | Acceptable | Tentative |
| D2 | Wilcoxon DE | Confirmed | scripts/de.R | Acceptable | Tentative |

## Priority findings

### Newer SOTA alternatives (not applied)
| Current decision | Alternative | Advantages | Migration cost | Auto-replaced |
|------------------|-------------|------------|----------------|---------------|
| ... | ... | ... | Low/Med/High | **No** |

### Outdated or suboptimal methods
| Decision | Issue | Suggested alternative | Evidence needed |
|----------|-------|----------------------|-----------------|
| ... | ... | ... | ... |

### Missing validation
| Gap | Risk | Recommendation |
|-----|------|----------------|
| No assumption checks before ANOVA | Invalid p-values | Add Shapiro/Levene or use non-parametric |

### Questionable parameters
| Parameter | Current | Concern | Recommendation |
|-----------|---------|---------|----------------|
| confidence=0.1 | Kraken2 | Low stringency | Compare 0.2–0.5; report sensitivity |

### Methodological improvements
| Area | Current | Proposed | Expected benefit |
|------|---------|----------|------------------|
| Compositional analysis | raw proportions | ALDEx2/ANCOM-BC | Valid inference on relative abundances |

## Confirmed vs inferred requiring user review
| Decision | Detection | Action |
|----------|-----------|--------|
| Thread count = 32 | Inferred from SLURM | Confirm in config or document |

## Literature review log
| Topic | Sources consulted | Date | Result |
|-------|-------------------|------|--------|
| Metagenomic classifiers | [Title, Year] or "Web search — no DOI verified" | YYYY-MM-DD | Kraken2 still benchmark-competitive |

*Never list invented DOIs.*

## Items not documented in repository
| Expected decision | Searched | Required from user |
|-------------------|----------|-------------------|
| Host read removal tool | Snakefile, configs | Confirm if not performed |
```

## Issue Categories

### Missing validation
- No `@dataset-auditor` or QC gate before analysis
- Statistical assumptions untested
- No train/test split for ML
- No multiple-testing correction
- No ablation or sensitivity analysis for critical choices

### Questionable parameters
- Default tool settings on non-standard data
- Hard thresholds without justification in repo
- Version pins missing for reproducibility-critical tools

### Outdated methods (examples — verify per domain)
- Use domain-appropriate checks; do not apply generic labels without evidence
- Flag only when literature or benchmark source supports "superseded" claim

## Merge Rules for method-decision.md

1. **New decision** → append new `###` section
2. **Existing + new evidence** → add `#### Verification` block or `Updated:` line
3. **Conflict** → set Status Open; list both evidence sources
4. **User manual notes** in file → preserve verbatim
5. **Never delete** prior entries; mark superseded entries `Status: cancelled` with reason

## Anti-Patterns

| Avoid | Prefer |
|-------|--------|
| Invent methods not in repo | Unknown + gap report |
| Fabricated DOIs for SOTA claims | Verified citation or Uncertain |
| Treating defaults as Confirmed | Inferred + recommend documenting |
| Generic "use deep learning" advice | Specific, evidence-linked recommendation |
| Recreating method-decision.md | Merge and timestamp updates |
| Auto-switching pipeline to newer SOTA | Document alternative + cost; keep Decision until user approves |
