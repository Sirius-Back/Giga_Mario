# Metagenome analysis — pipeline order

Skills from input → final rendered article. Omit SKIPPED branches; never reorder Locked stages.

## Full order

| # | Skill | When | Role |
|---|-------|------|------|
| 1 | `@setup` | Always | `Article.Rmd` theme_main, TARGET/BATCH |
| 2 | `@data` | Missing/unvalidated raw data | get-data + auditors |
| 3 | `@fix-metadata` | Metadata missing or IDs misaligned | `metadata_fixed.csv` |
| 4a | `@metabarcoding-import` | 16S | Complete phyloseq |
| 4b | `@bracken-parse` → `@metagenomic-import` (+ `@taxonomy-tree`) | WGS taxonomy | Complete phyloseq |
| 5 | `@removebatch` | BATCH present and adjustment requested | MMUPHin-adjusted phyloseq |
| 6 | `@rarefaction-analysis` | After import (± batch) | Curves + even-depth objects |
| 7 | `@alpha-diversity` | After rarefaction (preferred) | Raincloud/box + KW/Wilcoxon |
| 8 | `@beta-diversity` | After rarefaction (preferred) | PCoA + PERMANOVA |
| 9 | `@ordination` | After rarefaction | sPLS-DA (+ optional NMDS) |
| 10 | `@isa` | After rarefaction (2–3 target levels) | Indicator species + grazing Fig. 3; drop NA targets; −log10(p) audit |
| 11 | `@upset` | After rarefaction | ComplexUpset; **target-only** sets; descending bars; count labels |
| 12 | `@network` | After rarefaction | Default **NetCoMi** (SparCC for speed; SpiecEasi intended); coexistence; chord; igraph viz |
| 13 | `@phyloseq2metacoder` | After diversity stack | Taxmap |
| 14 | `@heattree` | After Taxmap | Family heat trees |
| 15 | `@ancombc` | After phyloseq ready | Taxonomic DA (ANCOM-BC2, **multilevel**) |
| 16 | `@difftree-metacoder` | After ancombc | Differential heat trees |
| 17 | `@difftree-ggtree` | After ancombc | Default **cladobox** (circular highlight + side boxes); fruit/twosided optional |
| 17b | `@difftree-ggdiffclade` | After rarefaction (alt. to 17) | PacBio **MicrobiotaProcess** `ggdiffclade`+`ggdiffbox`; legend on right boxplot |
| 18 | `@metabolism` | WGS functional tables present | Gene tables + top-N pheatmap (no GO) |
| 19 | `@metabolism-de` | After metabolism | ANCOM-BC2 on product/KO/EC |
| 20 | `@go` | WGS GO abundances present | ANCOM-BC2 GO DEG + enricher |
| 21 | `@figure-designer` | Optional | Figure plan / captions |
| 22 | `@methods-writer` | After analyses | Methods **inside** `Article.Rmd` |
| 23 | `@results-writer` | After analyses + figures | Results **inside** `Article.Rmd` |
| 24 | Render `Article.Rmd` | Last | HTML/PDF final manuscript |

Orchestration / QA skills (not analysis stages): `@prepare`, `@prepare-prompt`, `@do`, `@code-review` (via `@do`), `@monitor`, `@debug`, `@verify-todo`, `@verify-methods`.

## Branch matrix

| Modality | Import | Diversity + taxonomic DA | Functional |
|----------|--------|--------------------------|------------|
| 16S only | 4a | 5–17 (include `@isa` `@upset` `@network` `@ancombc` `@difftree-*`) | SKIP 18–20 |
| WGS taxonomy only | 4b | 5–17 | SKIP 18–20 unless Bakta/GO tables appear |
| WGS + Bakta/GO | 4b | 5–17 | 18–20 |
| Dual 16S+WGS | 4a and 4b (parallel after metadata) | per object | functional on WGS only |

## @prepare wiring

When calling `@prepare` (after `@prepare-prompt` if `./todo/*.md` are missing), each row above that is not SKIPPED becomes a task with:

- **Depends on** = previous non-SKIPPED task in the same branch
- **Skills** = that skill only (plus `@prompt-orchestrator` via `@do`)
- **Outputs** = paths expected by the skill’s `SKILL.md`

Writers (22–23) depend on all COMPLETED analysis tasks in scope. Render (24) depends on 22–23.

Always include `@isa`, `@upset`, and `@network` after rarefaction when the target has usable levels (ISA needs 2–3 levels; skip ISA with reason if not).

## Dependency sketch

```mermaid
flowchart TD
  setup[@setup]
  data[@data optional]
  fix[@fix-metadata optional]
  import16[@metabarcoding-import]
  importWGS[@metagenomic-import]
  batch[@removebatch optional]
  rare[@rarefaction-analysis]
  alpha[@alpha-diversity]
  beta[@beta-diversity]
  ord[@ordination]
  isa[@isa]
  upset[@upset]
  network[@network]
  p2m[@phyloseq2metacoder]
  heat[@heattree]
  ancom[@ancombc]
  dtm[@difftree-metacoder]
  dtg[@difftree-ggtree]
  met[@metabolism]
  metde[@metabolism-de]
  go[@go]
  meth[@methods-writer]
  res[@results-writer]
  rend[Render Article.Rmd]

  setup --> data --> fix
  fix --> import16
  fix --> importWGS
  import16 --> batch
  importWGS --> batch
  batch --> rare --> alpha & beta & ord & isa & upset & network
  rare --> p2m --> heat
  rare --> ancom --> dtm & dtg
  importWGS --> met --> metde
  importWGS --> go
  alpha & beta & ord & isa & upset & network & heat & dtm & dtg & metde & go --> meth --> res --> rend
```
