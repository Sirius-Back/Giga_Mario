---
name: verify-methods
description: >-
  Reconstruct methodological decisions from project artifacts, update
  method-decision.md, compare choices to current SOTA, and document newer
  alternatives with advantages and migration cost without auto-replacing methods.
  Use when the user asks to verify methods, audit methodology, review
  parameter choices, or assess whether analyses follow state of the art.
disable-model-invocation: true
---

# Verify Methods

Inspect the complete project and reconstruct every methodological decision from code, workflows, scripts, notebooks, configuration files and documentation. Update method-decision.md with all detected methods, parameters and software choices.

For every decision, compare the selected approach against current state-of-the-art practices reported in recent literature. Identify outdated methods, missing validation, questionable parameter choices and possible methodological improvements. Clearly distinguish confirmed decisions from inferred ones and never invent undocumented methods.

Follow project rules: **scientific-integrity**, **method-decision-tracking**, **statistical-analysis**, **validation-first**, **missing-data-policy**.

Pair with `@methods-writer` (Methods prose) and `@dataset-auditor` (input validation gaps).

## Workflow

Copy and track progress:

```
Method verification:
- [ ] Step 1: Map all methodological artifacts
- [ ] Step 2: Extract decisions with evidence
- [ ] Step 3: Classify confirmed vs inferred
- [ ] Step 4: Compare against current best practices + scan for newer SOTA alternatives
- [ ] Step 5: Update method-decision.md (document alternatives; never auto-replace)
- [ ] Step 6: Write Methods Verification Report
```

### Step 1: Map all methodological artifacts

Scan comprehensively (same sources as `@methods-writer`):

| Source | Extract |
|--------|---------|
| Workflows | Tool chain, rule order, default params |
| Scripts / notebooks | Commands, flags, thresholds, test choices |
| Config files | Reference DBs, cutoffs, model hyperparameters |
| Environment files | Software names and pinned versions |
| Documentation | Stated protocols, README claims |
| Existing `method-decision.md` | Prior entries — merge, do not discard |

Build a **decision inventory** — one row per distinct methodological choice (software, test, threshold, database, QC rule, workflow design).

### Step 2: Extract decisions with evidence

For each decision record:

| Field | Content |
|-------|---------|
| **Decision** | Exact choice (tool + version + key params) |
| **Evidence source** | File path, config key, line, or command |
| **Parameters** | Values explicitly set in repo |
| **Detection** | Confirmed / Inferred / Unknown |

**Confirmed** — explicit in code, config, or documented command  
**Inferred** — deduced from defaults or pipeline structure; not explicitly set  
**Unknown** — cannot determine; exclude from Methods claims; list in report gaps

Never document a method not traceable to at least Inferred evidence with cited source.

### Step 3: Classify confirmed vs inferred

Rules:

- Prefer **Confirmed** only with direct artifact citation
- **Inferred** entries must state inference chain (e.g., "Snakemake rule calls `megahit` with no `-t`; thread count inferred from SLURM header")
- Do not upgrade Inferred → Confirmed without new evidence
- User-locked choices in chat/docs → **Status: Locked**

### Step 4: Compare against current best practices

For each **Confirmed** or **Inferred** decision, assess:

| Assessment | Meaning |
|------------|---------|
| **Current** | Aligns with widely accepted recent practice for this task |
| **Acceptable** | Valid but not latest; document trade-offs |
| **Outdated** | Superseded approaches with known limitations |
| **Uncertain** | Insufficient literature access or domain ambiguity |

Literature review protocol:

1. Search recent reviews, benchmark papers, and tool documentation **when the user requests public search or when assessing SOTA**
2. Cite only **verified** titles, years, or DOIs from search results — never invent references
3. If search is not performed or inconclusive, mark SOTA as **Uncertain** and state what search is needed

#### Newer state-of-the-art alternatives (required)

Whenever a methodological decision is reconstructed, **automatically determine** whether newer state-of-the-art alternatives have appeared (relative to the reconstructed choice and its pinned version/date).

For each decision, answer explicitly:

- Has a newer SOTA alternative appeared? **Yes / No / Uncertain**
- If **Yes**, complete the **SOTA alternative record** (below)
- If **Uncertain**, state what search or evidence is missing

If a superior approach exists:

- **document it** (name, version/era, key references if verified);
- **explain advantages** over the current project method (accuracy, bias, scalability, interpretability, maintenance — evidence-based only);
- **estimate migration cost** (effort, revalidation, compute, API/format breaks, risk to published results);
- **never automatically replace** the existing method.

The reconstructed decision remains the recorded **Decision**. Alternatives belong under verification / recommendations only. Changing the live pipeline requires explicit user approval (then Status may become Locked to the new choice).

Flag specifically:

- **Missing validation** — no QC, no held-out test, no assumption checks
- **Questionable parameters** — defaults inappropriate for data type; thresholds undocumented
- **Methodological improvements** — concrete alternatives with expected benefit and migration cost (hypothesis, not promise)

Cross-check **statistical-analysis** rule: multiple testing, effect sizes, test appropriateness.

### Step 5: Update method-decision.md

**Merge** into existing file — append new entries, update changed entries in place, preserve history.

Use extended entry format in [verify-template.md](verify-template.md). Include all fields from **method-decision-tracking** plus verification fields, including the SOTA alternative record when applicable.

Update rules:

- New decision → append with date heading
- Existing entry with new evidence → add `Updated:` line; do not delete prior text
- Contradiction found (code vs config) → **Status: Open**; document both sources
- Newer superior alternative found → document under verification; **do not change Decision** or rewrite pipeline code
- Never recreate the file from scratch

Set **Status: Locked** for user-specified methods; **Tentative** for detected-only choices.

### Step 6: Write Methods Verification Report

Deliver separate summary (see template):

1. Decision inventory table (Confirmed / Inferred counts)
2. Outdated or questionable choices (priority ranked)
3. **Newer SOTA alternatives** (documented advantages + migration cost; none auto-applied)
4. Missing validation gaps
5. Recommended improvements (actionable, not generic; require user approval to adopt)
6. Literature gaps — unverified SOTA claims
7. Items requiring user confirmation

## Deliverables

1. **Updated `method-decision.md`**
2. **Methods Verification Report** — `docs/methods-verification.md` (or user path)

## Artifact registration

Instead of creating standalone reports in arbitrary locations, require every generated artifact to be registered inside `artifact-registry.md` (prefer `docs/artifact-registry.md`).

Each registry entry must contain:

- artifact
- producer skill
- generation date
- purpose
- status
- downstream consumers

Every generated report, graph, manifest or checkpoint must be registered immediately after it is written.

Update existing rows when regenerating the same path; mark replaced paths `superseded`.

Format: [artifact-registry-template.md](../_shared/artifact-registry-template.md). Project rule: `artifact-registry` (alwaysApply).

## Additional resources

- Extended entry format and report template: [verify-template.md](verify-template.md)
