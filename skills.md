# Skills catalog

Project skills live in [`.cursor/skills/*/SKILL.md`](.cursor/skills/). Shared templates: [`.cursor/skills/_shared/`](.cursor/skills/_shared/).

Discovery at runtime is owned by `@prompt-orchestrator` ([skills-map.md](.cursor/skills/prompt-orchestrator/skills-map.md)) — this file is a human summary, not a closed registry.

**Rules:** agents write production-ready `./src` code, reuse existing modules, and run pytest when changing prior functions (see `.cursor/rules/agent-production-code.mdc`, `skills-write-and-exec-src.mdc`).

Legacy Caduceus/LegNet/adapt skills were moved to [`archive/skills/`](archive/skills/). Prefer the universal skills below.

---

## Universal DNA / FM pipeline (primary)

| Skill | Role | Writes / execs |
|-------|------|----------------|
| `preprocess` | GTF+FNA+TARGET → ID / MARKED / PARSED / PREDICT (+ optional fold) + `parse.md` | `src/run/preprocess_{which_data}.py` |
| `split-generate` | Author strategy code from `splits/*.md` (fold/strat helpers, `id_rule`) | `src/splits/<id>.py` (+ registry) |
| `split` | split-predict + materialize SPLIT (+ ZSV) | `src/run/run_id/{data}_{split}_{direct\|adversarial}.py` |
| `train` | Caduceus/LegNet train + viz + TB + optional ZSV eval | `src/run/run_id/{data}_{split}_{train}_{direct\|adversarial}.py` |
| `adversarial` | Adversarial panel + random re-split | `src/run/run_id/{data}_{split}_adversarial.py` |
| `pipeline` | Orchestrate split → train → optional adversarial → train | `src/run/run_id/pipeline.py` (`dry`\|`run`) |

Docs: [wiki/architecture.md](wiki/architecture.md), [wiki/split-generate.md](wiki/split-generate.md).

### Obligatory inputs

#### `/preprocess`

| Required | Role |
|----------|------|
| `which_data` | Runner name → `src/run/preprocess_{which_data}.py` |
| GTF path/folder | `id_gen` + `adapt` |
| FNA path/folder | `adapt` |
| TARGET folder | TPM/MPRA for `parse_target` (or after `get_mpra`) |
| `outdir` | Stage outputs + `parse.md` |

Optional: `--mappings`, ZSV/`prepare_fold`, `environment`/`window`, `to_type` caduceus\|legnet, `get_mpra` flags.

#### `/split-generate`

| Required | Role |
|----------|------|
| Split strategy id | e.g. `random` |
| `splits/<id>.md` | Caption (Description, Split, Implementations, References) |
| Algorithm intent | Assignment rules from MD + user locks |
| Reuse preference | Prefer `src/splits/` + `src/pipeline/` |
| Tests | pytest for novel strategy logic |

When needed: fold rules (`prepare_fold` / `generate_fold` + `id_rule`); stratification CSV (auto `generate_stratification` is still **Not implemented**).

#### `/pipeline`

| Required | Role |
|----------|------|
| `run_id` | `src/run/<run_id>/` |
| `mode` | `dry` \| `run` |
| `data`, `split`, `train`, `type` | Panel / strategy / model / task |
| Panel or SPLIT inputs | Explicit PARSED, PREDICT, ID.csv / fold / strat as needed by `/split` |
| `out-root` | Artifact root |

If adversarial: `outdir_new` + `train_adversarial` (model for adversarial train).

Also see skill files for `/split`, `/train`, `/adversarial` obligatory tables.

---

## Execution & planning

| Skill | Brief |
|-------|-------|
| `do` | Multi-step parent execution engine |
| `do-fast` | Lean bookend audits + end code-review |
| `prepare` | Sync todos, assign skills/rules |
| `prepare-prompt` | Create/update `./todo/*.md` |
| `generate-todo` | Hierarchical `todo.md` from specs |
| `edit-prompt` | Edit plan without executing |
| `verify-todo` | TODO dependency graph |
| `synchronize-todo` | Reconcile `todo.md` with repo |
| `prompt-orchestrator` | Execute one assigned TODO |
| `code-review` | Independent review vs spec / method-decision |
| `monitor` | Long-job status only |
| `debug` | Safe repair of recoverable failures |
| `project-auditor` | Project audit; methods → verify-methods |
| `task-gate` | Brief post-task smoke check |

## Data

| Skill | Brief |
|-------|-------|
| `data` | get-data → dataset-auditor → project-auditor |
| `get-data` | Public dataset acquisition |
| `dataset-auditor` | Dataset QC |
| `fix-metadata` | Align sample IDs |
| `mock-data` | Reproducible fixtures |
| `summarize_GEO` | Mean-merge GEO TPM CSVs |

## Metagenome / metabarcoding

| Skill | Brief |
|-------|-------|
| `metagenome-analysis` | End-to-end Article.Rmd pipeline |
| `setup` | Canonical Article.Rmd |
| `metabarcoding-import` / `metagenomic-import` / `bracken-parse` | Import paths |
| `taxonomy-tree`, `removebatch`, `rarefaction-analysis` | Prep |
| `alpha-diversity`, `beta-diversity`, `ordination`, `isa`, `upset`, `network` | Analyses |
| `phyloseq2metacoder`, `heattree`, `ancombc`, `difftree-*` | Taxonomy DA / trees |
| `metabolism`, `metabolism-de`, `go` | Function / GO |

## Manuscript & QA

| Skill | Brief |
|-------|-------|
| `methods-writer` / `results-writer` | Nature-style drafts |
| `figure-designer` | Publication figures |
| `verify-methods` | method-decision.md maintenance |
| `reviewer-response` | Reviewer reply drafts |
| `benchmark-designer` | Benchmark design |
| `visualize-architecture` | Architecture diagrams |

## Archived (legacy)

See [`archive/skills/`](archive/skills/): former `adapt`, `adapt-legacy`, `caduceus`, `caduceus-full`, `legnet`, `legnet-adapt`, `train-viz`, `prepare-target`, genome reformat helpers, `analyze-ready-data`. Use `/preprocess`, `/split`, `/train`, `/pipeline` instead.
