# Refactoring plan: universal GigaMario tool

**Date:** 2026-07-28  
**Target architecture:** [wiki/architecture.md](wiki/architecture.md)  
**Evidence base:** `src/`, `.cursor/skills/` (Caduceus/LegNet ML path), `AGENTS.md`, `tasks.md`  
**Goal:** Package the repo as a tool anyone can run: adapt → parse_data → parse_target → split-predict → split → train — without Caduceus-only path assumptions.


> **Sync (2026-07-28):** Stage names updated — **parse** → **parse_data**, **prepare** → **parse_target**. PREDICT = `ID.ext` + `predict.csv`. Full function list in [wiki/architecture.md](wiki/architecture.md). Code changes still step-by-step; do not edit proven `src/` until the matching cutover.
>

---

## 0. Current architecture (as implemented)

```text
raw/{fna,gtf,tpm} + mapping.csv
        │
        ├─► src/preprocessing.py (@adapt)
        │     └─► data_ready/  (ready.fna, ready.csv, caduceus_ready/, QC tables)
        │           ready/ → symlink to data_ready/
        │
        ├─► src/legnet_preprocess.py (@legnet-adapt)   [parallel path]
        │     └─► legnet_ready/*.tsv (+ BED/FASTA)
        │
        └─► src/splits/{main,random,common}.py (@split)
              └─► splits/.../M1|M2/{train,val,test}/{sequences,labels.tsv}
                    │
                    ├─► src/caduceus.py (@caduceus) → runs/... + logs/
                    └─► src/legnet.py (@legnet) ← uses legnet_ready TSV, not project splits
                          │
                          └─► src/train_viz/ (@train-viz)
```

Orchestration: `src/runs/caduceus_full.py` = split → train M1/M2 → ZS → viz (**skips** adapt). Skills mirror these modules; metagenome R skills are out of scope for this refactor.

### What is collapsed today

| Target stage | Current owner | Problem |
|--------------|---------------|---------|
| adapt (GTF+FNA → MARKED + intersect) | inside `preprocessing.py` | No MARKED FASTA; no `intersect.csv`; TPM and non-coding baked in |
| parse_data (MARKED → PARSED) | same module + `caduceus_ready/` | No stable ID contract independent of Caduceus `.txt` layout |
| parse_target (TARGET → PREDICT) | TPM join inside `process_genome()` | No separate PREDICT tree; missing gene → 0 is implicit |
| stratification / fold tables | M2 stratification in memory | No first-class `stratification.csv` / `fold.csv` |
| split.csv | `splits_log.csv` + per-model `fold_manifest.tsv` | Not a single `ID;train-test;fold` contract |
| materialize SPLIT | `materialize_fold()` in `splits/common.py` | Caduceus `sequences/*.txt` + `labels.tsv` only |
| train | `caduceus.py` / `legnet.py` | Two incompatible input schemas |

### Skill / package debt (high signal)

- `@adapt` + `@legnet-adapt` duplicate raw discovery and TPM join.
- `@genome-fna-gtf-reformat` / `@genome-tpm-caduceus-reformat` overlap adapt discovery.
- Todo `@prepare` is plan execution; pipeline **parse_target** will be run by a new `@prepare` skill — rename the orchestrator on migration. Current Caduceus `@adapt` → `@adapt-legacy`.
- `pyproject.toml` only installs empty `GigaMario` package; real code lives under `src/` without console entry points.
- `src/_archive/` and dual `train_viz` / `train-viz` paths increase drift risk.

---

## 1. Major steps (ordered)

### Step 1 — Freeze contracts (docs + schemas)

**Status intent:** READY → COMPLETED when schemas are Locked in `method-decision.md`.

1. Promote [wiki/architecture.md](wiki/architecture.md) as the normative flowchart.
2. Specify machine-readable schemas (CSV/TSV + version header) for:
   - `intersect.csv`, `stratification.csv`, `fold.csv`, `split.csv`
   - directory layouts: `GTF/`, `FNA/`, `MARKED/`, `PARSED/`, `PREDICT/`, `SPLIT/{FASTA,PREDICT}/{train,test,val}/`
3. Define stable **ID** rules (genome, chrom, start, end, strand, kind) and the “no target → predict 0” rule.
4. Document optional vs required inputs (`stratification.csv` / `fold.csv` may be NULL).

**Exit:** Schema fixtures under `tests/fixtures/contracts/`; method-decision entry **Locked**.

### Step 2 — Extract shared panel I/O

Split discovery out of `src/preprocessing.py` / `src/legnet_preprocess.py` into something like `src/panel/` (or `src/io/`):

- discover FNA/GTF pairs
- load TARGET tables (TPM wide CSV + mapping)
- write/read a single **panel manifest** (replaces ad-hoc reformat skill outputs)

**Exit:** Both Caduceus and LegNet prep import the same discovery API; reformat skills become thin wrappers or deprecate into this module.

### Step 3 — Implement **adapt** as MARKED + intersect only

Refactor CDS±flank / neighbour trim / large-gene crop (current Caduceus policy) into an **adapt profile**, not the only path:

- Inputs: `FNA/` + `GTF/`
- Outputs: `MARKED/ID.fa`, `intersect.csv`
- **Do not** join TPM; **do not** emit `caduceus_ready/`

Keep today’s CDS±10 kb + non-coding policy as profile `caduceus_cds_flank` (parameters from config YAML). Add hooks for future profiles (TSS±100 for LegNet marking, etc.).

**Exit:** CLI `gigamario adapt --profile …`; golden tests on smoketest genomes.

### Step 4 — Implement **parse** (MARKED → PARSED)

Standalone stage: read marked records → `PARSED/ID.ext` (sequence + minimal metadata). No model-specific adapters here.

**Exit:** `gigamario parse`; IDs identical to MARKED.

### Step 5 — Implement **prepare** (TARGET → PREDICT)

Join raw TARGET onto PARSED IDs → `PREDICT/ID.ext` (+ optional panel `predict.csv`). Default TARGET = TPM; missing → **0**.

Migrate TPM logic out of `process_genome()` into this stage. `@summarize_GEO` remains upstream of prepare (merged TARGET files).

**Exit:** `gigamario prepare`; round-trip tests with known gene symbols / missing genes.

### Step 6 — Split assignment vs materialization

**6a. split prediction (`E2` → `split.csv`)**  
Strategies (start with random; then SBS / hashFrag / MMseqs / etc. from `tasks.md`) write only:

```text
ID;train-test;fold
```

Inputs: IDs (+ optional `stratification.csv`, `fold.csv`, `intersect.csv`, FNA/GTF for similarity). No sequence copies.

**6b. materialize (`E3` → SPLIT)**  
Apply `split.csv` to `PARSED` + `PREDICT` → `SPLIT/FASTA/...` and `SPLIT/PREDICT/...`.

Refactor `src/splits/random.py` + `common.materialize_fold()` into this two-phase API. Update `@split` skill accordingly.

**Exit:** Random strategy produces bit-identical partitions to today’s M1 assignment (seed 42) when mapped through the new tables; materializer can still emit a **Caduceus adapter** view for backward compatibility.

### Step 7 — Model adapters + train

- **Caduceus adapter:** SPLIT → today’s `sequences/*.txt` + `labels.tsv` (or teach `caduceus.py` to read SPLIT directly).
- **LegNet adapter:** SPLIT or PARSED+PREDICT → 230 bp TSV; stop forking the full raw→ready path where possible.
- Keep `metrics.md` / `train_metrics.jsonl` as the shared logging contract for `@train-viz`.

**Exit:** `gigamario train --model caduceus|legnet …` over SPLIT roots; old CLIs remain as shims during transition.

### Step 8 — Package for “anyone can use it”

1. Expand `pyproject.toml`: package `src` (or move to `gigamario/`), declare console scripts, pin deps via `environment.yml`.
2. Public CLI surface:

   ```text
   gigamario adapt | parse | prepare | split-assign | split-materialize | train | viz
   ```

3. Minimal README: install (conda), one smoketest command, link to wiki architecture.
4. Ship contract tests + a tiny fixture panel (not full `raw/`).
5. Align skills: rename/clarify stages; map `@adapt` → adapt+parse profile; add `@prepare-target` (name TBD) distinct from todo `@prepare`; update `@caduceus-full` to orchestrate new stages.

**Exit:** `pip install -e .` / conda env; stranger can run smoketest end-to-end from docs alone.

### Step 9 — Cleanup and compatibility

- Mark `src/_archive/` read-only / document removal timeline.
- Collapse `train-viz` symlink duplication.
- Compatibility shims: `data_ready/` builders that call adapt+parse+prepare under the hood for one release.
- Update wiki [[conversion]], [[split]], [[Split & train]] to point at architecture + new CLIs; keep historical pages marked legacy.
- Register deliverables in `docs/artifact-registry.md`.

---

## 2. Suggested module map (after refactor)

| New module (proposed) | Responsibility | Seeds from |
|-----------------------|----------------|------------|
| `src/panel/` | FNA/GTF/TARGET discovery + manifests | `preprocessing.discover_raw`, reformat skills |
| `src/adapt/` | MARKED + intersect profiles | `preprocessing.process_genome` (sequence half) |
| `src/parse/` | MARKED → PARSED | `write_ready_*` / caduceus export |
| `src/prepare/` | TARGET → PREDICT | TPM join in `process_genome` |
| `src/splits/assign/` | → `split.csv` | `splits/random.py` assignment logic |
| `src/splits/materialize/` | → SPLIT trees | `splits/common.materialize_fold` |
| `src/models/caduceus.py` | train adapter | `caduceus.py` |
| `src/models/legnet.py` | train adapter | `legnet.py` + preprocess stitch |
| `src/train_viz/` | unchanged contract | existing |
| `src/cli.py` | console entry | new |

---

## 3. Compatibility matrix (migration)

| Existing artifact | Maps to | Migration note |
|-------------------|---------|----------------|
| `raw/fna`, `raw/gtf` | `FNA/`, `GTF/` | Layout rename or dual-root config |
| `raw/tpm` + mapping | raw TARGET | feed **prepare** only |
| `ready.fna` / `ready.csv` | MARKED + PARSED (+ legacy bundle) | split writers |
| `caduceus_ready/` | Caduceus materializer output or PARSED view | stop treating as universal ready |
| `legnet_ready/*.tsv` | LegNet adapter output | optional direct path kept for vendor demo |
| `splits/.../M1` | SPLIT + Caduceus adapter | M2 = PREDICT derived from fold labels, still explicit |
| `splits_log.csv` | superseded by `split.csv` | keep as derived report if needed |

---

## 4. Risks and non-goals

| Risk | Mitigation |
|------|------------|
| Breaking in-flight Caduceus runs | Shims + freeze seed-42 random parity tests before deleting old paths |
| Scope creep into metagenome skills | Out of scope; do not fold R phyloseq pipeline into this tool |
| Naming clash `@prepare` | New stage skill/CLI must not reuse todo-orchestrator name |
| Over-generalizing before schemas lock | Step 1 gates Steps 3–7 |

**Non-goals for this refactor:** rewriting Caduceus/LegNet model code; full SBS/hashFrag implementations (those land after Step 6a API exists — see `tasks.md`).

---

## 5. First concrete milestone (recommended)

1. Lock schemas (Step 1).  
2. Implement `split.csv` emission from current random assignment without changing fold membership (Step 6a thin cut).  
3. Implement materialize-from-`split.csv` that reproduces today’s M1 tree (Step 6b).  
4. Wire CLI + smoketest (Step 8 partial).  

This delivers a usable universal **split** spine while adapt/parse/prepare are still factored out of `preprocessing.py`.
