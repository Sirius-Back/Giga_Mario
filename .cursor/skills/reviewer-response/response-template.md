# Reviewer Response Templates

## Point-by-Point Response Letter

```markdown
# Response to Reviewers

We thank the editor and reviewers for their constructive comments. We have revised the manuscript accordingly. Changes in the revised manuscript are highlighted in [yellow/bold/track changes — specify format].

---

## Reviewer 1

### Comment 1.1
> [Exact reviewer comment quoted here]

**Response:** We thank the reviewer for this suggestion. [Summary of action taken.]

**Changes in manuscript:** [Section name], [paragraph/line or Fig./Table reference]. [One-sentence description of edit.]

**Justification:** [Evidence-based rationale. No unsupported claims.]

[Optional excerpt:]
> "[Short quoted revised text from manuscript.]"

---

### Comment 1.2
> [Next comment]

**Response:** ...

---
```

## Response Block Fields

Use all applicable fields; omit only when truly N/A.

| Field | Content |
|-------|---------|
| **Response** | Direct answer; what was done or why |
| **Changes in manuscript** | Precise location(s) |
| **Justification** | Scientific or methodological reasoning |
| **Excerpt** | New wording (≤3 sentences) when helpful |

## Phrasing Patterns

### Change made

> We agree that [issue] warranted clarification. We have [added/revised/expanded] [content] in [location]. This now states that [factual summary].

### Already in manuscript

> We appreciate this point. This information was present in the original submission ([location]). We have [bolded/expanded/moved] the text to improve visibility.

### Partial agreement

> We agree that [valid point]. We have [specific change]. We did not [requested item] because [evidence-based limit]. We note this limitation in [Discussion, paragraph X].

### Respectful decline

> We thank the reviewer for this suggestion. We respectfully maintain [current approach] because [design/data/evidence reason]. We have added [clarifying sentence] in [location] to make this rationale explicit to readers.

### Requires new analysis

> We agree this analysis would strengthen the manuscript. [Describe planned analysis if user confirmed.] **Author action:** [specific computation or data needed before final response.]

## Author Action Required

Separate document for items the assistant cannot complete.

```markdown
# Author Action Required

Comments that need user input, new analyses, or verified manuscript edits before the response is final.

## Reviewer 2, Comment 3
- **Reviewer request:** [Summary]
- **Gap:** [What's missing — analysis, text, figure]
- **Searched:** [Manuscript files, results inspected]
- **Needed from author:** [Exact action]
- **Draft response status:** Hold — do not submit until resolved

## Summary
| Reviewer | Comment | Priority | Status |
|----------|---------|----------|--------|
| R2 | 3 | Critical | Awaiting analysis |
```

## Change Log (optional internal)

```markdown
# Revision Change Log

| Reviewer | Comment | Manuscript location | Change type |
|----------|---------|---------------------|-------------|
| R1 | 1.1 | Results, para 2 | Reworded |
| R1 | 1.2 | Fig. 2c | New panel |
| R2 | 3 | — | Pending |
```

## Tone Checklist

- [ ] Thank reviewer or acknowledge the point
- [ ] Answer the specific question asked
- [ ] Location cited for every claimed edit
- [ ] No invented statistics, citations, or changes
- [ ] No defensive or dismissive language
- [ ] Limitations acknowledged where appropriate
- [ ] Consistent terminology with revised manuscript

## Anti-Patterns

| Avoid | Prefer |
|-------|--------|
| "The reviewer is wrong." | "We have clarified in [location] that …" |
| "This is obvious." | "We have added an explicit statement …" |
| "We will address in a future study." (without user approval) | "We discuss this limitation in [location]." |
| Claiming p = 0.01 without source | "The updated analysis (Supplementary Table X) shows …" |
| Vague "throughout the manuscript" | "Results, paragraph 3; Methods, 'Statistical analysis'" |
