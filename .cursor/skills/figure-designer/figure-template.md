# Figure Design Templates

## Figure Plan Template

```markdown
# Figure Plan

## Narrative summary
[One paragraph: what story the figure set tells]

## Main figures

### Fig. 1 — [Short title]
- **Main message:** [One sentence]
- **Results link:** [Subsection that references this figure]
- **Priority:** Essential
- **Panels:** schematic (a) | QC overview (b) | cohort summary (c)
- **Status:** [Ready / Partial / Planned]

### Fig. 2 — [Short title]
...

## Extended Data / Supplementary
### Extended Data Fig. 1 — [Title]
...

## Tables
### Table 1 — [Title]
...
```

## Panel Layout Spec Template

```markdown
### Fig. 1 layout

| Panel | Type | Data source | Axes / content | Stats overlay | Notes |
|-------|------|-------------|----------------|---------------|-------|
| a | Schematic | methods workflow | Study design | — | Illustrator or BioRender style |
| b | Box plot | tables/qc_summary.tsv | x: group; y: reads (M) | median, IQR; n per group | Shared y-axis with c |
| c | Violin | tables/qc_summary.tsv | x: group; y: GC (%) | — | Color by group |

**Grid:** 1 row × 3 columns (a wide schematic spanning top optional)
**Dimensions:** 183 mm full width (Nature double column) or 89 mm single column
**Export:** `figures/fig1_design_qc.pdf`
```

## Caption Template

Nature-style; observational, not interpretive.

```markdown
**Fig. 1 | [Brief title sentence case].**
**a**, [Experimental design description with n and groups]. **b**, [QC metric] across [groups] (n = [values]; box, median and IQR). **c**, [Second metric] ([units]). [Additional note on test or filtering if shown on panel].
```

Extended Data example:

```markdown
**Extended Data Fig. 1 | [Title].**
**a**, … **b**, … Statistical tests: [test name]; multiple testing correction: [method] where applicable.
```

## Visual Style Guide Template

```markdown
# Project Figure Style Guide

## Export
- Primary: PDF (vector), fallback SVG
- Raster: PNG ≥300 DPI only for dense heatmaps/microscopy
- `bbox_inches="tight"`; embed fonts

## Palette (colorblind-safe)
| Role | Hex | Use |
|------|-----|-----|
| Group A | #0072B2 | Control |
| Group B | #D55E00 | Treated |
| Group C | #009E73 | Third group |
| Neutral | #999999 | Background/reference |

Grayscale backup: distinct markers (circle, triangle, square) + line styles (solid, dashed).

## Typography
- Font: Arial or Helvetica (Nature-compatible)
- Axis labels: 9 pt; tick labels: 8 pt; panel labels: 10 pt bold (a, b, c)
- Legend: outside plot area when possible

## Layout
- Panel labels: top-left, bold lowercase letter
- Consistent bar width, point size, line width (1–1.5 pt)
- Shared color mapping across all figures for same groups/conditions

## File naming
`figures/fig{N}_{short_description}.{pdf|svg}`
```

## Recommended Plot Types by Analysis

| Analysis | Preferred plot | Avoid |
|----------|----------------|-------|
| Group comparison (continuous) | Box + points or violin | Pie charts |
| Composition | Stacked bar or treemap | 3D charts |
| Beta diversity | PCoA/NMDS with ellipses | Unlabeled axes |
| Differential abundance | Volcano or effect-size forest | Raw p without correction |
| Time series | Line with CI ribbon | Overplotting without alpha |
| Correlation | Scatter + regression CI | Dual y-axes without need |
| Heatmap | Clustered with colorbar | Rainbow colormap |

## Figure Gap Report Template

```markdown
# Figure Gap Report

Panels in the figure plan that cannot be produced yet.

## Fig. 2b — PCoA by treatment
- **Missing:** Beta diversity ordination coordinates
- **Required analysis:** Weighted UniFrac PCoA from `data/otu_table.tsv`
- **Blocks:** Fig. 2 panel b and Results paragraph 3
- **Suggested command/script:** [if known from project workflow]
```

## Nature-Style Checklist

Before finalizing design:

- [ ] One clear message per main figure
- [ ] Colorblind-safe palette documented
- [ ] Grayscale-readable encodings (shape/line type)
- [ ] All axes labeled with units
- [ ] n displayed or in caption
- [ ] Vector export specified
- [ ] No interpretive language in captions
- [ ] Extended Data holds QC/sensitivity overflow
- [ ] Planned panels clearly marked vs available data
