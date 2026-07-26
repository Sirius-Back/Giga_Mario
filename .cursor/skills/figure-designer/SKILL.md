---
name: figure-designer
description: >-
  Design publication-quality figure layouts for Nature-style manuscripts from
  project goals and available analyses. Use when the user asks for figure plans,
  panel layouts, figure sets, captions, or visualization design for a paper.
disable-model-invocation: true
---

# Figure Designer

Design publication-quality figure layouts for scientific manuscripts. Based on the project goals and available analyses, propose the complete figure set, panel organization, captions, expected visualizations, color consistency, and publication-ready formatting. Optimize figures for Nature-style journals.

Follow project rules: **publication-figures**, **nature-writing-style**, **scientific-integrity**, **statistical-analysis**, **missing-data-policy**.

## Workflow

Copy and track progress:

```
Figure design:
- [ ] Step 1: Clarify project goals and story
- [ ] Step 2: Inventory available analyses and outputs
- [ ] Step 3: Propose complete figure set
- [ ] Step 4: Design panel layouts per figure
- [ ] Step 5: Write captions and visual specs
- [ ] Step 6: Report gaps (analyses not yet available)
```

### Step 1: Clarify project goals and story

Read before designing:

- User request and target journal (default: Nature-style unless specified)
- `README`, `method-decision.md`, workflow outputs, notebooks
- Primary claims the figures must support (one main message per figure)

Identify the **narrative arc**: design/QC → primary result → mechanism or validation → supplementary depth.

### Step 2: Inventory available analyses and outputs

Map what exists vs what would need to be generated:

| Analysis / output | Location | Status | Suitable plot type |
|-------------------|----------|--------|--------------------|
| QC yield | tables/qc.tsv | Available | Box/violin |
| Beta diversity | results/beta/ | Missing | PCoA |

Only assign panels to **Available** or clearly label **Planned** panels. Do not present Planned analyses as completed.

### Step 3: Propose complete figure set

Deliver a numbered figure plan:

- **Main figures** (typically 4–6 for Nature-style; adjust to story)
- **Extended Data / Supplementary figures** for QC detail, sensitivity analyses, methods schematics
- **Tables** when tabular data communicates better than plots

Each figure gets:

- **Figure number and title**
- **One-sentence main message**
- **Results linkage** — which Results subsection references this figure
- **Priority** — Essential / Recommended / Optional

Nature-style conventions:

- Fig. 1 often: study design schematic + overview/QC or primary cohort summary
- Reserve the strongest quantitative result for an early main figure when possible
- Group related panels; avoid redundant panels showing the same conclusion

### Step 4: Design panel layouts

For each figure, specify a **panel grid** using the template in [figure-template.md](figure-template.md).

Include per panel:

- **Panel ID** (a, b, c, …)
- **Visualization type** (bar, box, heatmap, PCoA, schematic, etc.)
- **X/Y axes or equivalent** with units
- **Data source** (file path or table)
- **Statistics overlay** (error bars, CI, n, q-values — per statistical-analysis rule)
- **Size role** — hero panel vs supporting panel

Layout rules:

- Align panels to a consistent grid; shared legends where possible
- Hero panel ≥ supporting panels in visual weight
- Max ~6–8 panels per main figure unless journal allows composite complexity
- Schematics separate from data panels when clarity improves

### Step 5: Write captions and visual specs

**Captions** (Nature-style):

- Start with bold panel labels if multi-panel: **a**, Experimental design. …
- Factual, concise; describe what is shown — not interpretation (save for Discussion)
- Include n, test names, correction method when shown on panel
- Define all abbreviations once per figure

**Global visual spec** (project-wide):

- Output: PDF/SVG vector default; ≥300 DPI raster only when needed
- Palette: colorblind-safe (Okabe–Ito or viridis); document hex codes
- Typography: font family, axis label size (typically 8–10 pt final size)
- Line weights, marker shapes for grayscale readability
- File naming: `figures/fig1_abundance.pdf`

Provide a **style block** the project can reuse (see [figure-template.md](figure-template.md)).

### Step 6: Report gaps

If analyses required for proposed panels do not exist, deliver a **Figure Gap Report** separately. Do not imply data exists. List what must be computed and which panel it unblocks.

## Deliverables

Default output (unless user specifies otherwise):

1. **Figure Plan** — complete numbered set with rationale
2. **Panel Layout Specs** — grid, plot types, data sources per panel
3. **Captions** — draft manuscript-ready text
4. **Visual Style Guide** — palette, fonts, export settings
5. **Figure Gap Report** — missing analyses (if any)

Save to user-requested path, or suggest `docs/figure-plan.md` and `docs/figure-gaps.md`.

## Coordination with other skills

- `@results-writer` — align figure order with Results narrative
- `@methods-writer` — schematics and pipeline overview panels
- Implementation — after design approval, generate plots following **publication-figures** rule

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

- Templates and layout patterns: [figure-template.md](figure-template.md)
