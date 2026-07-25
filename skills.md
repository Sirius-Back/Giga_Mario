# Skills catalog

Project skills live in [`.cursor/skills/*/SKILL.md`](.cursor/skills/). **50** skills; all use `disable-model-invocation: true`. Shared templates: [`.cursor/skills/_shared/`](.cursor/skills/_shared/).

Discovery at runtime is owned by `@prompt-orchestrator` ([skills-map.md](.cursor/skills/prompt-orchestrator/skills-map.md)) — this file is a human summary, not a closed registry.

---

## Setup check (2026-07-26)

| Check | Result |
|-------|--------|
| Skill root | `.cursor/skills/` — present |
| Skill count | 50 directories with `SKILL.md` |
| Orchestrators | `@do`, `@do-fast`, `@data`, `@metagenome-analysis`, `@caduceus` (+ `@prepare` / `@split`) |
| Metagenome pipeline | `metagenome-analysis/pipeline.md` — present |
| Caduceus folds | `@split` + `splits/*.md` + `AGENTS.md` |
| Gaps | No prior `skills.md`; no `artifact-registry.md` until this write |

---

## Brief catalog

### Execution & planning

| Skill | Brief |
|-------|-------|
| `do` | Multi-step parent execution engine (auditor → verify-todo → READY waves → orchestrator → review → monitor → sync). |
| `do-fast` | One orchestrator; lean gates: full audit start+end, `@task-gate` mid-loop, code-review at end. |
| `prepare` | Sync todos, assign skills/rules, order from verify-todo graph; write prepare-report. |
| `prepare-prompt` | Create/update `./todo/*.md` from request or `todo.md`; validate via verify-todo. |
| `generate-todo` | Build hierarchical `todo.md` from architecture/specs (not `./todo/*.md`). |
| `edit-prompt` | Edit plan without executing: impact analysis, freeze completed, propagate, rollback. |
| `verify-todo` | Sole TODO dependency-graph builder/validator (`dependency-graph.md` / `.mmd`). |
| `synchronize-todo` | Reconcile `todo.md` with repo state (completed/partial/obsolete/missing). |
| `prompt-orchestrator` | Execute exactly one assigned TODO task (not a project scheduler). |
| `code-review` | Independent review vs TODO, prompt, acceptance criteria, method-decision. |
| `monitor` | Status-only supervision of long jobs (default 30m direct; 2h when inherited from do/do-fast). |
| `debug` | Safe repair of recoverable failures; escalate when unsafe. |
| `project-auditor` | Scientific project audit (reproducibility, docs, readiness); delegates methods to verify-methods. |

### Data

| Skill | Brief |
|-------|-------|
| `data` | Orchestrate data-ready: get-data → dataset-auditor → project-auditor. |
| `get-data` | Acquire public datasets (SRA/ENA/GEO/…) with manifests, checksums, logs. |
| `dataset-auditor` | Dataset QC: metadata, IDs, balance, duplicates, batch, file consistency. |
| `fix-metadata` | Align sample IDs; write `metadata_fixed.csv` when misaligned. |
| `mock-data` | Reproducible 16S/WGS/Bracken fixtures under `./test/`. |

### Metagenome / metabarcoding analysis

| Skill | Brief |
|-------|-------|
| `metagenome-analysis` | End-to-end: prepare → do → Methods/Results in `Article.Rmd` → render. |
| `setup` | Canonical `Article.Rmd` (`theme_main`, TARGET/BATCH, libraries). |
| `metabarcoding-import` | 16S → complete phyloseq. |
| `bracken-parse` | Fast Bracken/Kraken-report → wide counts + host cleanup. |
| `metagenomic-import` | WGS Bracken → complete phyloseq (+ host cleanup). |
| `taxonomy-tree` | Rebuild trees from names/taxids (rentrez → ape → ggtree). |
| `removebatch` | MMUPHin batch adjustment preserving TARGET. |
| `rarefaction-analysis` | Rarefaction curves + even-depth phyloseq. |
| `alpha-diversity` | Alpha metrics + box/raincloud by targets. |
| `beta-diversity` | PCoA (Aitchison / wUniFrac) + PERMANOVA. |
| `ordination` | sPLS-DA (default) or NMDS by targets. |
| `isa` | Indicator species + grazing Figure 3 panel set. |
| `upset` | ComplexUpset presence/absence by target. |
| `network` | NetCoMi / SparCC coexistence, chord, igraph viz. |
| `phyloseq2metacoder` | phyloseq → metacoder Taxmap. |
| `heattree` | Family (default) metacoder heat trees. |
| `ancombc` | ANCOM-BC2 multilevel taxonomic DA. |
| `difftree-metacoder` | Differential metacoder heat trees (uses prior ancombc). |
| `difftree-ggtree` | Differential ggtree; default cladobox layout. |
| `difftree-ggdiffclade` | MicrobiotaProcess `ggdiffclade` + `ggdiffbox` (alt. to ggtree). |
| `metabolism` | Bakta functional tables + top-N pheatmap (no GO). |
| `metabolism-de` | ANCOM-BC2 on product/KO/EC. |
| `go` | GO DEG + clusterProfiler enricher. |

### Caduceus / ML splits

| Skill | Brief |
|-------|-------|
| `caduceus` | Caduceus DNA LM facts (HF, Hydra, RC/Ph/PS); folds via `@split`; pipelines via `@do-fast`. |
| `split` | Build train/val/test/(zero-shot) from `splits/*.md`; execute via lean `@do-fast`. |
| `task-gate` | Brief post-task output/AC smoke check (not full audit or code-review). |
| `genome-fna-gtf-reformat` | Index paired `.fna`/`.gtf` genomes; optional distinct-species subsample; manifests for `@split`. |
| `benchmark-designer` | Design rigorous benchmarking experiments. |

### Manuscript & QA

| Skill | Brief |
|-------|-------|
| `methods-writer` | Materials & Methods from code/configs/versions (into Article.Rmd when orchestrated). |
| `results-writer` | Results prose from figures/tables/stats (observations only). |
| `figure-designer` | Nature-style figure plans, panels, captions. |
| `reviewer-response` | Point-by-point peer-reviewer rebuttals. |
| `verify-methods` | Reconstruct/update `method-decision.md`; SOTA comparison without auto-lock. |
| `visualize-architecture` | Mermaid data-flow from repo + method-decision. |

---

## Family: do & do-fast

Same inclusion set; difference is **invocation shape** (parent multi-step vs one-shot orchestrator subagent).

### Inclusion

```mermaid
flowchart TB
  subgraph engines["Engines"]
    do["@do"]
    dofast["@do-fast"]
  end
  subgraph included["Included cycle skills"]
    aud["@project-auditor"]
    vt["@verify-todo"]
    po["@prompt-orchestrator"]
    cr["@code-review"]
    mon["@monitor"]
    dbg["@debug"]
    sync["@synchronize-todo optional"]
  end
  prep["@prepare prerequisite"]
  prep -.-> do
  prep -.-> dofast
  do --> aud & vt & po & cr & mon & dbg
  do -.-> sync
  dofast --> aud & vt & po & cr & mon & dbg
  dofast -.-> sync
```

### Dependencies

```mermaid
flowchart TD
  prepare["@prepare"] --> engine{"@do or @do-fast"}
  engine --> aud1["@project-auditor"]
  aud1 --> vt["@verify-todo"]
  vt -->|Blocking| stop[Terminate]
  vt --> ready[READY tasks + resource gate]
  ready --> po["@prompt-orchestrator"]
  po --> cr["@code-review"]
  cr -->|Critical/Major| po
  cr --> mon{"computational?"}
  mon -->|yes| monitor["@monitor"]
  monitor -->|error Safe| dbg["@debug"]
  dbg --> po
  mon -->|no| aud2["@project-auditor"]
  monitor -->|ok| aud2
  aud2 -->|Critical| po
  aud2 -->|ok| done[Mark COMPLETED]
  done --> sync[Sync todo / graph / artifacts / checkpoint]
  sync --> aud1
```

---

## Family: data

### Inclusion

```mermaid
flowchart TB
  data["@data"]
  data --> gd["@get-data"]
  data --> da["@dataset-auditor"]
  data --> pa["@project-auditor"]
```

### Dependencies

```mermaid
flowchart TD
  p0[Phase 0: detect state] --> need{missing / invalid data?}
  need -->|yes| gd["@get-data"]
  need -->|no skip| da
  gd -->|fail required| stop[Stop]
  gd -->|ok| da["@dataset-auditor"]
  da -->|Not ready| stop2[Stop unless override]
  da -->|ok| pa["@project-auditor"]
  pa --> report["data-preparation-report.md"]
```

---

## Family: metagenome

Orchestrator: `@metagenome-analysis`. Canonical order: [pipeline.md](.cursor/skills/metagenome-analysis/pipeline.md).

### Inclusion

```mermaid
flowchart TB
  meta["@metagenome-analysis"]
  meta --> prep["@prepare / @prepare-prompt / @generate-todo"]
  meta --> do["@do"]
  meta --> writers["@methods-writer / @results-writer"]
  meta --> setup["@setup"]
  subgraph analysis["Analysis skills included by modality"]
    data["@data optional"]
    fix["@fix-metadata optional"]
    imp16["@metabarcoding-import"]
    brk["@bracken-parse"]
    impWGS["@metagenomic-import"]
    tax["@taxonomy-tree"]
    batch["@removebatch optional"]
    rare["@rarefaction-analysis"]
    div["@alpha-diversity @beta-diversity @ordination"]
    isa["@isa @upset @network"]
    p2m["@phyloseq2metacoder @heattree"]
    da["@ancombc @difftree-*"]
    func["@metabolism @metabolism-de @go WGS functional"]
    fig["@figure-designer optional"]
  end
  meta --> analysis
  do --> cr["@code-review"]
  do --> mon["@monitor"]
  do --> dbg["@debug"]
  do --> vt["@verify-todo"]
```

### Dependencies

```mermaid
flowchart TD
  setup["@setup"] --> data["@data optional"]
  data --> fix["@fix-metadata optional"]
  fix --> import16["@metabarcoding-import"]
  fix --> bracken["@bracken-parse"]
  bracken --> importWGS["@metagenomic-import"]
  importWGS --> tax["@taxonomy-tree as needed"]
  import16 --> batch["@removebatch optional"]
  importWGS --> batch
  batch --> rare["@rarefaction-analysis"]
  rare --> alpha["@alpha-diversity"]
  rare --> beta["@beta-diversity"]
  rare --> ord["@ordination"]
  rare --> isa["@isa"]
  rare --> upset["@upset"]
  rare --> network["@network"]
  rare --> p2m["@phyloseq2metacoder"]
  p2m --> heat["@heattree"]
  rare --> ancom["@ancombc"]
  ancom --> dtm["@difftree-metacoder"]
  ancom --> dtg["@difftree-ggtree"]
  rare -.-> dtc["@difftree-ggdiffclade alt"]
  importWGS --> met["@metabolism"]
  met --> metde["@metabolism-de"]
  importWGS --> go["@go"]
  alpha & beta & ord & isa & upset & network & heat & dtm & dtg & metde & go --> meth["@methods-writer"]
  meth --> res["@results-writer"]
  res --> rend[Render Article.Rmd]
```

Orchestration spine (outside analysis DAG): input analysis → `@prepare` → `@do` → writers/render → report.

---

## Family: caduceus

### Inclusion

```mermaid
flowchart TB
  cad["@caduceus"]
  cad --> facts[Caduceus facts: HF / Hydra / RC Ph-PS / env pins]
  cad --> split["@split required for folds"]
  cad --> dofast["@do-fast required for pipelines"]
  split --> splitsmd["splits/*.md + AGENTS.md"]
  dofast --> cycle["do-fast cycle skills"]
  cad --> mtrain["model-train.mdc process"]
```

### Dependencies

```mermaid
flowchart TD
  need[Caduceus task] --> folds{needs train/val/test/zero-shot?}
  folds -->|yes| dataok{data present?}
  dataok -->|no| data["@data / @get-data"]
  dataok -->|yes| split
  data --> split["@split via @do-fast"]
  folds -->|no one-shot HF| hf[Part 1 inference in-chat]
  split --> dofast["@do-fast + Caduceus overrides"]
  need -->|multi-step train/eval/VEP| dofast
  dofast --> mon["@monitor via do-fast"]
  dofast --> out[checkpoints / embeddings / reports]
```

Order: resolve data + split strategy → `@split` → `@do-fast` for Caduceus work beyond folds. Do not invent splits or bypass `@do-fast` for project pipelines.
