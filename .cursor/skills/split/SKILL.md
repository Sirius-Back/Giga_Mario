---
name: split
description: >-
  Split genomic REGIONS and linked prediction targets (TPM CSV by default) into
  train/val/test per splits/*.md. Runs @adapt before split when input is
  Caduceus-like; acquires/converts data otherwise. Use for /split or folds.
disable-model-invocation: true
---

# Split

## Purpose

Produce reproducible **train / validation / test** (and optional zero-shot) folds for a **named dataset** and **named split strategy** (`splits/*.md`), then execute through **`@do-fast`** (lean verification by default).

**Atomic split unit = genomic REGION** (interval / gene window / sequence sample) **plus** its **linked prediction** (default: continuous TPM from `*tpm*.csv` / `expression_tpm.csv`).

It MUST NOT treat a whole-assembly FASTA file as one ML sample unless the user **explicitly** locks species/sample-grain folding (legacy exception). Default and Caduceus paths split **regions**, not full FNA-per-genome bags.

do-fast in brief: one orchestrator subagent; **full `@project-auditor` only at start & end**; **brief `@task-gate` after each task**; **`@code-review` once at end**; `@verify-todo` each wave; monitor only for heavy jobs.

This skill owns **split planning + handoff to `@do-fast`**. It does not reimplement `@do-fast`.

Follow: **validation-first**, **missing-data-policy**, **reproducibility**, **method-decision-tracking**, **artifact-registry**, **slurm-execution-policy**, **task-status**. Model training extras: `AGENTS.md`, `.cursor/rules/model-train.mdc`.

## Required inputs

| Input | Meaning | Examples |
|-------|---------|----------|
| **данные** (data) | Dataset / path to split | `random/`, `data/raw/genomes`, `adapt/`, `data_splits/full`, Caduceus GB tree |
| **сплит** (split) | Strategy under `splits/` | `random`, `splits/random.md` |

If either is missing → **stop**. Ask for both; do not guess.

Optional: seed, ratios / fold counts, `OUT`, prediction path/glob (default TPM CSV), whether zero-shot, implementation name from `# Implementations`.

## Analyze split caption (mandatory)

Before any write, read `splits/<id>.md` and record:

| Section | Extract |
|---------|---------|
| `# Description` | What is randomly (or otherwise) assigned — **samples / intervals / sequences** |
| `# Split` | Roles of `train` / `validation` / `test` / `zero_shot` |
| `# Implementations` | `url`, `split_location`, `run`, `notes`, fallbacks |
| Input vs output | Expected on-disk **input** shape vs fold **output** shape for the chosen implementation |

**`splits/random.md` (reference):** samples (genomic intervals, sequences or examples) are randomly assigned to train / validation / test without chromosome/gene/species blocking. Caduceus GB: train/test from dataset; validation = random 90/10 from train via `dataset.train_val_split_seed`.

Map that caption onto **this project’s dual streams**:

1. **Genomic stream** — regions derived from FNA + GTF/GFF/BED/genes (or already-adapted sequences)
2. **Prediction stream** — TPM (default) or other label column, **row-aligned to the same region IDs**

## Dual-stream contract (LOCKED for default path)

| Stream | What is split | Default sources |
|--------|---------------|-----------------|
| Genomic regions | Intervals / windows / `.txt` sequences — **parts of genomes**, not whole FNA as one sample | FNA+GTF/genes → regions; or `adapt/samples.tsv` / `adapt/caduceus_ready/**/sequences` |
| Predictions | One target per region | `*tpm*.csv`, `expression_tpm.csv`, or `adapt/labels.tsv` |

**Linkage rule:** every genomic region has exactly one prediction row with the same `sample_id` / `region_id`.

- If the region has an associated gene with TPM → use that TPM
- If the region has **no gene** / no TPM join → prediction **`0`** (document in manifest; do not drop the region unless the split MD or user says so)
- Never leave genomic and prediction folds misaligned

## Input classification → routing

```
Classify DATA:
  A) Caduceus-like / adapt-compatible
  B) Already region-split (folds or GB train/test trees)
  C) Raw / other (assemblies, random/*, mixed)
```

### A — Caduceus-like input → **`@adapt` before `@split`**

Treat as Caduceus-like when DATA looks like what Caduceus / `@adapt` consume, e.g.:

- Per-genome `genome.fna` + `genes.tsv` / GTF + `expression_tpm.csv`
- `adapt/` already present (reuse; do not re-adapt unless forced)
- Sequence folders that match Caduceus / GB `.txt` trees **plus** a TPM or label table to build

**Action:**

1. If `adapt/samples.tsv` (or user `ADAPT_OUT`) is missing/incomplete → invoke **`@adapt`** (`/adapt`) on DATA
2. Then split **adapted region samples** (rows of `samples.tsv` / sequence files) into `train`/`val`/`test` per `splits/*.md` + seed
3. Split **predictions** by copying/joining `TPM` (or labels) with the **same** region IDs into each fold
4. Write Caduceus-ready fold trees under `OUT/` (see Outputs)

Do **not** skip `@adapt` on Caduceus-like raw genomes and then fold whole FNA files.

### B — Already split genomic data → **predict the split**

When DATA already has `train`/`val`/`test` (or GB `train`/`test`) **at region or sample grain**:

1. Verify genomic region files exist per fold
2. Build or verify prediction tables **per fold** so each region ID keeps its label
3. If predictions missing: join TPM CSV by gene/region; missing gene → **0**
4. Do **not** randomly re-split unless the user asks for a new split run
5. Run `@adapt` only if regions are still whole-genome assets that need windowing before Caduceus

### C — Raw / different input → acquire → convert → regionize → split

1. **Check necessary data** (genomes, annotations, TPM). List gaps; never invent.
2. **Missing → `@data` / `@get-data`** (`/data`) for acquisition; stop if still critical-missing.
3. **Convert** when layout ≠ split algorithm needs:
   - Prefer existing skills: `@genome-fna-gtf-reformat`, `@genome-tpm-caduceus-reformat`, `@adapt`
   - Else tools/scripts: GTF→BED, FASTA region extraction, interval join to TPM, etc.
   - Record every conversion in `method-decision.md`
4. Materialize **region table** + **prediction table** (linked)
5. If the target model is Caduceus and regions are not yet Caduceus windows → **`@adapt`** then split adapted samples
6. Run the split algorithm from `splits/*.md`

## Workflow

```
Split:
- [ ] 1. Parse DATA + SPLIT_MD; fail early if missing
- [ ] 2. Read AGENTS.md; read splits/*.md caption (Description / Split / Implementations)
- [ ] 3. Analyze input vs expected output; classify A / B / C
- [ ] 4. Ensure genomic + prediction sources (TPM default); @data if missing
- [ ] 5. Convert / regionize as needed; @adapt if Caduceus-like or windows required
- [ ] 6. Link regions ↔ predictions (no gene → TPM 0)
- [ ] 7. NEW run? archive old todos + reset do-fast checkpoint
- [ ] 8. Materialize chunky todo.md + ./todo/*.md
- [ ] 9. @verify-todo → @prepare if needed
- [ ] 10. @do-fast with Split execution contract (VERIFY_PROFILE=lean unless strict)
- [ ] 11. Surface Final Report + OUT paths
```

### New run vs resume

| Situation | Action |
|-----------|--------|
| Resume same OUT / same todo IDs | Keep `docs/do-fast-checkpoint.md` |
| New OUT or new task graph | Reset checkpoint; archive old `todo/T-*.md`; fresh `todo.md` |

### Todo shaping

| Prefer | Avoid |
|--------|--------|
| One task: caption analysis + input class A/B/C + pin implementation | Doc-only micro-tasks |
| One task: acquire/convert/adapt as required by class | Silent whole-FNA fold assignment for Caduceus |
| One task: region+TPM linked split + manifests + report | Genomic folds without prediction tables |

**Proven Caduceus region path:**

| Task | Scope |
|------|-------|
| T-1 | Read AGENTS + SPLIT_MD; classify input; lock seed/OUT/ratios |
| T-2 | Ensure data (`@data` if needed) + convert; **`@adapt`** if Caduceus-like |
| T-3 | Seeded region+TPM split → `OUT/{train,val,test}/` + manifests + `docs/split-report.md` |

### Split execution contract (runs inside `@do-fast`)

Pass **verbatim** into `@do-fast` `{{USER_OVERRIDES_OR_NONE}}` (plus `DATA=…`, `SPLIT_MD=…`, `OUT=…`, `SEED=…`, `PRED=TPM`):

```
* читает agents.md
* читает соответствующий splits/*md
* анализирует input vs output под caption сплита; классифицирует A/B/C
* скачивает код реализации при необходимости
* обеспечивает genomic + prediction (TPM по умолчанию): @data если нет; конвертеры/скилы если формат не подходит
* если input Caduceus-like — сначала @adapt, затем split регионов
* сплитует GENOMIC REGIONS (не целый FNA как один sample) по алгоритму splits/*.md
* сплитует PREDICTIONS согласованно: raw → из TPM; already-split → prediction следует genomic fold
* связка region↔prediction обязательна; нет гена → prediction=0
* сохраняет фолды train/, val/, test/ (+ zero-shot только если указано)
* репортит результаты
```

Also pass (unless user requested strict gates):

```
VERIFY_PROFILE=lean
```

| Step | Action |
|------|--------|
| agents.md | Pipeline, Main MDs, training-advance note |
| splits/*md | Description / Split / Implementations; honor fallbacks |
| анализ A/B/C | Caduceus-like → adapt-first; already-split → align preds; raw → data/convert |
| код | Clone/fetch implementation `url` if needed; record commit; do not re-clone if present |
| data | `@data`/`@get-data` when missing; never invent TPM/sequence |
| convert | Skills/tools (GTF→BED, region extract, `@genome-tpm-caduceus-reformat`, …) |
| `@adapt` | Required before region split when input is Caduceus-like / needs gene±flank windows |
| region split | Assign **region IDs** to folds per caption (e.g. random). Fixed seed |
| prediction split | Same IDs; TPM default; missing gene → **0** |
| save | `OUT/train|val|test/` with genomic + prediction artifacts + manifests |
| report | `docs/split-report.md` + artifact registry |

## Outputs

Under `OUT` (project-relative):

| Path | Content |
|------|---------|
| `train/` `val/` `test/` | Per-fold **regions** + **predictions** |
| `zero-shot/` | Only if MD/user requests |
| `fold_manifest.tsv` | `region_id`, fold, genome, chrom, start, end, strand, gene_id, TPM, sequence_path, pred_path, seed, split_id |
| `predictions/{fold}.tsv` or per-fold `labels.tsv` | Linked targets (TPM) |
| `docs/split-report.md` | Always |

### Caduceus-oriented fold layout (after adapt + region split)

Prefer:

```
OUT/{train,val,test}/
  sequences/<region_id>.txt    # DNA window
  labels.tsv                   # region_id, TPM
```

Optional: retain source genome hardlinks for audit, but **ML samples are regions**.

Legacy whole-sample FNA+GTF+TPM dirs are allowed **only** when the user explicitly locks species/sample-grain folding — record that Locked exception in `method-decision.md`.

### Report template (`docs/split-report.md`)

```markdown
# Split Report

**Date:** YYYY-MM-DD
**Data:** …
**Split:** splits/<id>.md (`id`)
**Input class:** A Caduceus-like | B already-split | C raw/other
**Implementation:** name + url @ commit
**Seed / ratios:** …
**Prediction:** TPM (default) | other: …

## Caption analysis
- Description unit: …
- Train/val/test/zero_shot: …
- Input → output mapping: …

## Folds written
| Fold | N regions | N preds | Path |
|------|-----------|---------|------|
| train | … | … | … |
| val | … | … | … |
| test | … | … | … |

## Genomic ↔ prediction linkage
- Join key: …
- Regions with TPM: …
- Regions with prediction=0 (no gene): …
- Orphans / mismatches: … (must be 0 or explained)

## Adaptation / conversion
- @adapt: yes/no + out path
- @data / converters / skills: …
- Exclusions: …

## Run
- Command(s): …
- Outcome: success | failed | partial
- Manifest: …

## Notes / blockers
…
```

## Invoking `@do-fast`

1. Todos cover the contract; prefer chunky T-1…T-3.
2. New run → reset checkpoint.
3. Launch `@do-fast` exactly as that skill requires (one orchestration prompt).
4. `USER_OVERRIDES` = verbatim Russian bullets + resolved paths + `VERIFY_PROFILE`.
5. Wait for exit; do not micromanage; do not duplicate do-fast internals.

## Lessons (finalize reference)

### Region + TPM (current default)

What to do:

- Read split caption first; unit = interval/sequence sample
- Caduceus-like genomes → **`@adapt` then random-split region rows**
- Link every region to TPM; **no gene → 0**
- Acquire with `@data` when TPM/FNA/GTF missing; convert when schemas differ

What to avoid:

- Folding whole mammalian FNA as one Caduceus training example by default
- Genomic folds without prediction tables
- Inventing TPM; silent adapt skip on Caduceus-like input
- Re-using an old EXIT A checkpoint for a new OUT

### Legacy species-grain panels (`data_splits/full`)

Still valid **only** as a Locked user exception (coherent genome+transcriptome bags). Prefer region-level splits going forward for Caduceus training.

## Rules

- Never invent split semantics — only `splits/*.md` + user inputs.
- Never invent data — acquire (`@data`) or stop.
- Split **regions + predictions** together; keep IDs aligned.
- Missing gene on a region → prediction **0** (documented).
- Caduceus-like input → **`@adapt` before split** (unless adapt outputs already valid).
- Fixed seeds for any randomness; relative project paths.
- Zero-shot only when explicitly indicated.
- CV / LOO / extras: implement in **both** `@split` and `@caduceus` when supported (`AGENTS.md`).

## Coordination

| Skill / doc | Role |
|-------------|------|
| `AGENTS.md` | Pipeline + split MD shape |
| `splits/*.md` | Strategy caption + implementations |
| `@do-fast` | Execution engine (required; lean default) |
| `@adapt` | Caduceus windows + TPM **before** region split when class A |
| `@data` / `@get-data` | Download when genomic or TPM assets missing |
| `@genome-fna-gtf-reformat` | Paired FNA/GTF manifests |
| `@genome-tpm-caduceus-reformat` | GCF + TPM coherent manifests (pre-regionize) |
| `@caduceus` | Consumes region folds + labels after split |
| `@prepare-prompt` / `@generate-todo` | Todo materialization |

## Additional resources

- Example strategy: `splits/random.md`
- Adapt: `~/.cursor/skills/adapt/SKILL.md`
- Data: `~/.cursor/skills/data/SKILL.md`
- do-fast: `~/.cursor/skills/do-fast/SKILL.md`
- Format notes: `docs/caduceus_format.md`
