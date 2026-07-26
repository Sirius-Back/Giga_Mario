---
name: reviewer-response
description: >-
  Generate structured peer-reviewer responses with manuscript change locations
  and scientific justification. Use when the user asks for reviewer replies,
  rebuttal letters, point-by-point responses, or revision cover letters.
disable-model-invocation: true
---

# Reviewer Response

Generate detailed responses to peer reviewers. Match every reviewer comment with a structured response explaining manuscript modifications, scientific justification, and corresponding manuscript locations. Maintain a professional and constructive tone while avoiding unsupported arguments.

Follow project rules: **scientific-integrity**, **nature-writing-style**, **statistical-analysis**, **missing-data-policy**.

## Workflow

Copy and track progress:

```
Reviewer response:
- [ ] Step 1: Parse all reviewer comments
- [ ] Step 2: Read current manuscript and revision artifacts
- [ ] Step 3: Classify each comment
- [ ] Step 4: Draft point-by-point responses
- [ ] Step 5: Flag unsupported or missing changes
- [ ] Step 6: Self-check tone and evidence
```

### Step 1: Parse all reviewer comments

Accept input as:

- Editor decision letter, reviewer PDFs, copied comment text, or structured lists
- Number comments exactly as reviewers wrote them (Reviewer 1, Comment 2.3, etc.)

Preserve original wording in block quotes. Do not merge separate comments unless the user requests it.

### Step 2: Read current manuscript and revision artifacts

Before responding, inspect when available:

| Artifact | Purpose |
|----------|---------|
| Manuscript draft (`.docx`, `.tex`, `.md`) | Locate sections, line/paragraph references |
| Track-changes or diff | Verify what actually changed |
| `method-decision.md` | Justify methodological choices |
| New figures/tables, stats outputs | Support revised claims |
| `@methods-writer` / `@results-writer` outputs | Align wording with Methods/Results |

Do not claim a change was made unless verified in manuscript or diff.

### Step 3: Classify each comment

Assign one type per comment:

| Type | Response strategy |
|------|-------------------|
| **Accepted — change made** | Describe change + location + rationale |
| **Accepted — partial change** | Explain what was done; note limits honestly |
| **Already addressed** | Point to existing text/figure; quote briefly |
| **Declined with justification** | Polite disagreement + evidence/literature |
| **Requires author input** | Cannot draft final response; list what's missing |
| **Out of scope** | Acknowledge; suggest Discussion/limitation if appropriate |

Never classify as "change made" without verified manuscript location.

### Step 4: Draft point-by-point responses

Use the template in [response-template.md](response-template.md).

Every response must include:

1. **Gratitude / acknowledgment** — brief, professional
2. **Response summary** — what was done or why not
3. **Manuscript changes** — section, paragraph, figure/table, or line reference
4. **Scientific justification** — evidence-based; cite literature only when real DOIs/titles are provided or found
5. **Quoted revision text** (optional) — short excerpt of new/changed wording

Tone rules:

- Constructive, respectful, non-defensive
- No sarcasm, dismissiveness, or blame
- Avoid "we believe the reviewer misunderstood" → prefer "we have clarified …"
- Do not overclaim ("this fully resolves") when evidence is partial

### Step 5: Flag unsupported or missing changes

If a comment requires data, analyses, or text not available in the project, add an entry to **Author Action Required** (see template). Do not invent:

- New statistics or p-values
- References or DOIs
- Changes not present in the manuscript
- Promises of future work without user confirmation

### Step 6: Self-check

Before delivering:

- [ ] Every reviewer comment has exactly one response block
- [ ] All "change made" claims have manuscript locations
- [ ] No unsupported scientific arguments
- [ ] Tone is professional throughout
- [ ] Author Action Required lists unresolved items

## Deliverables

Default output (unless user specifies otherwise):

1. **Point-by-Point Response Letter** — full rebuttal
2. **Author Action Required** — comments needing user input or new analyses
3. **Change Log** (optional) — comment → file/section mapping for internal use

Save to user-requested path, or suggest `docs/reviewer-response.md` and `docs/reviewer-response-actions.md`.

## Coordination with other skills

- `@methods-writer` — verify Methods revisions match stated responses
- `@results-writer` — ensure Results changes support reviewer-requested clarifications
- `@figure-designer` — new or revised figure panels referenced in responses

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

- Response structure and phrasing: [response-template.md](response-template.md)
