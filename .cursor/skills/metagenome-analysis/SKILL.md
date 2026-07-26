---
name: metagenome-analysis
description: >-
  End-to-end metagenome/metabarcoding orchestration: analyze inputs, @prepare
  with all analysis skills in pipeline order, run @do subagents, write Methods
  and Results into Article.Rmd via methods-writer/results-writer, then render.
  Use when the user invokes /metagenome-analysis or asks for a full 16S/WGS
  analysis from data through manuscript Rmd.
disable-model-invocation: true
---

# Metagenome Analysis

## Purpose

Orchestrate a full metagenome or metabarcoding analysis from input data through a rendered `Article.Rmd`.

**Never duplicate** subordinate skill logic — read each skill’s `SKILL.md` and delegate.

Follow project rules: **validation-first**, **missing-data-policy**, **method-decision-tracking**, **scientific-integrity**, **task-status**, **artifact-registry**, **slurm-execution-policy**, **reproducibility**, **nature-writing-style**.

## Subordinate skills (fixed order)

Canonical list and branch rules: [pipeline.md](pipeline.md).

## Orchestration checklist

```
Metagenome analysis:
- [ ] Phase 0: Analyze input (modality, paths, TARGET/BATCH, gaps)
- [ ] Phase 1: @prepare — ALL required skills listed in pipeline order
- [ ] Phase 2: @do — execute waves via subagents
- [ ] Phase 3: Ensure Article.Rmd skeleton (@setup)
- [ ] Phase 4: @methods-writer → Methods inside Article.Rmd
- [ ] Phase 5: @results-writer → Results inside Article.Rmd
- [ ] Phase 6: Render Article.Rmd (final HTML/PDF)
- [ ] Phase 7: Register artifacts + report
```

---

## Phase 0 — Analyze input

Inspect the project and user paths. Build an **input state table**; do not guess missing files.

Detect:

| Signal | Implication |
|--------|-------------|
| `*.qza`, `feature-table.tsv`, ASV/OTU + taxonomy | **16S** → `@metabarcoding-import` |
| `*.bracken*`, `*.nt.G.bracken`, Kraken/Bracken reports | **WGS taxonomy** → `@metagenomic-import` (+ `@bracken-parse` / `@taxonomy-tree` as needed) |
| Bakta / metabolic long-wide tables, GFF3 gene products | **WGS functional** → `@metabolism` → `@metabolism-de`; GO → `@go` |
| Metadata missing / sample IDs misaligned | Insert `@fix-metadata` before import |
| BATCH column present and user wants adjustment | Include `@removebatch` |
| No public accessions yet / empty `data/raw` | Include `@data` (→ `@get-data` / auditors) only if needed |

Record:

- modality (`16S` | `WGS` | `both`)
- primary TARGET and optional BATCH
- existing phyloseq / Taxmap / DE outputs (skip completed steps)
- blockers (stop per missing-data-policy)

Write `docs/metagenome-analysis-input.md` (or project-root equivalent) with the state table. Register it.

---

## Phase 1 — @prepare

Invoke **`@prepare`** (`/prepare`). Do **not** invent a separate prepare-todo skill.

`@prepare` syncs task metadata, assigns skills/rules, consumes the `@verify-todo` graph, and writes `prepare-report.md`. It does **not** create new `./todo/*.md` files.

Before `@prepare`:

1. Ensure `todo.md` exists (run `@generate-todo` only if absent / full replan needed).
2. Ensure `./todo/*.md` specs exist for **every** required skill below in **pipeline order** (run `@prepare-prompt` if tasks are missing — that is `@prepare`’s documented upstream).
3. Pass / assign the ordered skill list so each task’s Skills field matches [pipeline.md](pipeline.md).

Then run `@prepare`. If `@verify-todo` is Blocking, **stop** — do not run `@do`.

### Required skill list (include all that apply)

List **all** necessary skills for the detected modality in this order (see [pipeline.md](pipeline.md) for full I/O):

1. `@setup` — Article.Rmd theme_main
2. `@data` — only if acquisition/audit needed
3. `@fix-metadata` — only if metadata broken
4. `@metabarcoding-import` **or** `@metagenomic-import` (mutually exclusive primary import; both if dual modality)
5. `@removebatch` — only if BATCH adjustment required
6. `@rarefaction-analysis`
7. `@alpha-diversity`
8. `@beta-diversity`
9. `@ordination`
10. `@isa` — indicator species (grazing Fig. 3); 2–3 target levels; drop NA targets; −log10(p) audit
11. `@upset` — ComplexUpset; target-only sets; descending intersections; count labels
12. `@network` — NetCoMi default (SparCC for speed); coexistence; chord; igraph (phylum/edge colors + tip labels)
13. `@phyloseq2metacoder`
14. `@heattree`
15. `@ancombc` — multilevel taxonomic DA
16. `@difftree-metacoder`
17. `@difftree-ggtree` — default **cladobox** (circular highlight + side boxes); fruit/twosided optional
17b. `@difftree-ggdiffclade` — PacBio MicrobiotaProcess `ggdiffclade`+`ggdiffbox` (legend on right); alt. to 17
18. `@metabolism` → `@metabolism-de` → `@go` — **WGS functional only**
19. `@figure-designer` — optional figure plan before prose
20. `@methods-writer` — Methods into Article.Rmd
21. `@results-writer` — Results into Article.Rmd
22. Render task — knit final Article.Rmd

Do **not** invent skills outside `.cursor/skills/`. Do **not** reorder Locked pipeline stages. Do **not** skip `@isa` / `@upset` / `@network` / `@ancombc` / `@difftree-*` on 16S unless levels or prior ancombc are missing (then SKIP with reason).

---

## Phase 2 — @do

Invoke `@do` so executable tasks run via subagents (`prompt-orchestrator` → `code-review` → monitor → auditor cycle).

- Resume from checkpoints if restarted
- Respect SLURM for heavy jobs
- Do not bypass verify-todo Blocking exits
- Continue until taxonomic (+ functional if planned) analysis tasks are COMPLETED or blocked

Manuscript writer + render tasks may run in `@do` waves **after** analysis outputs exist, or immediately after `@do` finishes analysis (Phases 4–6). Prefer including them in the todo graph so `@do` owns execution.

---

## Phase 3 — Article.Rmd skeleton

If not already done in a `@do` task: invoke `@setup` so `Article.Rmd` has canonical `theme_main` / TARGET / BATCH.

Ensure placeholders or sections exist for:

- Methods
- Results

Do not invent numerical results in the Rmd before writers run.

---

## Phase 4 — @methods-writer inside Rmd

Invoke `@methods-writer`.

- Source of truth: executed scripts, configs, `method-decision.md` (Locked), env versions
- **Write the Methods prose into `Article.Rmd`** (Methods section / chunk), not only a standalone `docs/methods.md` (standalone copy optional)
- Attach Missing Information Report as `docs/methods-missing.md` when gaps exist
- Never invent parameters or citations

---

## Phase 5 — @results-writer inside Rmd

Invoke `@results-writer` **after** analysis artifacts and figures exist.

- Observations only; Nature Results style
- **Write the Results prose into `Article.Rmd`** (Results section / chunk)
- Reference figures/tables with stable labels
- Missing Information Report → `docs/results-missing.md` when needed

---

## Phase 6 — Render final results

Knit the article:

```bash
Rscript -e 'rmarkdown::render("Article.Rmd", output_format = "html_document")'
```

Prefer PDF when the project already uses `pdf_document` / bookdown and TeX is available. For long knits, use an sbatch wrapper (CPUs/mem/time/logs per slurm-execution-policy).

Fail early if `Article.Rmd` or required figure paths are missing.

Outputs: rendered HTML and/or PDF beside `Article.Rmd` (or `docs/`). Register them.

---

## Phase 7 — Report

Write `docs/metagenome-analysis-report.md`:

- modality and skill path taken
- prepare / do outcome summary
- Article.Rmd + rendered paths
- blockers / SKIPPED steps with reasons

Register all new artifacts in `artifact-registry.md`.

---

## Execution rules

- Analyze before planning; `@prepare` before `@do`; writers after analysis; render last.
- `@prepare` must cover **all** necessary skills for the detected modality in pipeline order (create missing `./todo/*.md` via `@prepare-prompt` first if needed).
- Always run `@do` for analysis execution (subagents) — do not silently run the full pipeline only in the parent agent unless `@do` is impossible (then report why).
- Methods and Results must go **into** `Article.Rmd` via `@methods-writer` / `@results-writer`.
- Honor Locked entries in `method-decision.md`.
- Stop on missing required inputs (missing-data-policy).

## Additional resources

- Pipeline order and branch matrix: [pipeline.md](pipeline.md)
- Prepare skill: [../prepare/SKILL.md](../prepare/SKILL.md)
