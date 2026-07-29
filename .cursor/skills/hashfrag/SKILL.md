---
name: hashfrag
description: >-
  Run hashFrag to detect and mitigate homology-based train/test leakage in
  genomic sequence datasets (BLAST candidates → score filter → filter, stratify,
  or create orthogonal splits/folds). Primary project inputs are MARKED folders
  (per-ID FASTA); convert to multi-FASTA before calling hashFrag. Use when the
  user mentions hashFrag, MARKED→hashFrag, homology leakage, orthogonal splits,
  homology-aware folds, or BLAST-based sequence similarity for
  sequence-to-expression models.
disable-model-invocation: true
---

# hashFrag

## Purpose

Use the [hashFrag](https://github.com/de-Boer-Lab/hashFrag) CLI to find homologous sequence pairs and reduce **homology-based data leakage** between train and evaluation sets.

Official docs: [hashfrag.readthedocs.io](https://hashfrag.readthedocs.io/en/latest/).  
Paper: [bioRxiv 2025.01.22.634321](https://www.biorxiv.org/content/10.1101/2025.01.22.634321v2).

Follow: **validation-first**, **reproducibility**, **missing-data-policy**, **scientific-integrity**, **artifact-registry**, **method-decision-tracking**. Prefer **SLURM** for large BLAST jobs (project slurm-execution-policy).

Do **not** reimplement BLAST / union-find / split logic in chat. Call `hashFrag` (and thin `./src` adapters only when project format conversion is required).

## When to use

| Goal | hashFrag command |
|------|------------------|
| Remove test seqs homologous to train | `filter_existing_splits` |
| Bin test by max homology to train | `stratify_test_split` |
| Build homology-aware train/test | `create_orthogonal_splits` |
| Build homology-aware k-folds | modules → `create_orthogonal_folds_module` |
| Large panels on cluster | `blastn_array_module` (SLURM/SGE) then continue modules |

## Primary input: MARKED folders

In this project, **hashFrag inputs start from `MARKED/`** — the universal adapt-stage sequence panel (`wiki/architecture.md`, `src.pipeline.adapt`).

Do **not** feed Caduceus `.txt` trees, LegNet `PARSED/*.ext`, or TPM tables into hashFrag. Homology search runs on **genomic MARKED windows**, not adapter-stitched or model-serialized strings.

### MARKED contract (verified)

| Item | Spec |
|------|------|
| Path | `{panel}/MARKED/` (examples: `ready_caduceus/MARKED`, `ready_legnet/MARKED`, `run/run0/MARKED`) |
| Layout | **One file per region:** `MARKED/{ID}.fa` (also `.fasta` accepted by loaders) |
| Records | Exactly **one** FASTA record per file |
| Filename stem | Region `ID` — same as `ID.csv` `ID` and `split.csv` `ID` |
| Header | `>\|genome\|chr\|pos1\|pos2\|gene_nameORnon_coding_ID\|raw_target_ID\|ID` (leading `\|` after `>`; 7 pipe fields; see `MARKED_HEADER_FIELDS` / `parse_marked_header` in `src.pipeline.common`) |
| Sequence | Genomic DNA from adapt (`environment` + `window`); often 80-col wrapped; length = window (e.g. 200 bp for ±100) |
| Producer | `python -m src.pipeline.adapt` → `MARKED/` + `intersect.csv` |

Example header (real panel):

```text
>|GCF_000001635.27|NC_000077.7|120333357|120333557|Tspan10|Tspan10|100000
```

**Join key:** prefer **filename stem** as the stable ID (matches SBS `load_fna_directory`). Trailing header field must equal the stem when both are present; if they disagree, **stop** and report (do not guess).

Full schema notes: [reference.md](reference.md#marked-schema).

### Gap vs hashFrag FASTA

| MARKED | hashFrag expects |
|--------|------------------|
| Directory of per-ID `.fa` files | One or more **multi-FASTA** files (`train.fa` / `test.fa` / `all.fa`) |
| Pipe-rich headers | Unique FASTA headers; BLAST ID = first whitespace-delimited token |
| No fold labels in the folder | Optional `split.csv` to define train vs test membership |

**Safest header rule for conversion:** write `>{ID}` where `ID` is the MARKED filename stem (joinable to `split.csv` / `ID.csv`). Do **not** use the full pipe string as the hashFrag header — it is unique but is not the panel join key and complicates downstream mapping.

MARKED windows are single-orientation genomic extracts — do **not** invent `_Reversed` mates; leave hashFrag RC generation on (omit `--skip-revcomp`) unless the user already supplied RC records.

## Pre-flight

1. **Python ≥ 3.10** and `hashFrag` on PATH (`pip show hashFrag` or `hashFrag -h`).
2. **BLAST+** on PATH: `blastn -version` and `makeblastdb -version` must succeed.
3. **MARKED** path exists, is a directory, and contains non-empty `*.fa` / `*.fasta` (or an explicit multi-FASTA already converted).
4. For filter / stratify: `split.csv` present with columns `ID|train_test|fold` (or user-locked train/test ID lists).
5. Homology **threshold** (`-t`) required for filter / orthogonal-split pipelines. If unset, stop and ask — or recommend calibrating from dinucleotide-shuffled negatives (see docs). Record in `method-decision.md`.
6. Missing MARKED / FASTA / BLAST / threshold → **stop** (missing-data-policy). Do not invent scores or leakage flags.

Install (preferred):

```bash
pip install hashFrag
# BLAST+ from NCBI: https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/
```

Optional env: clone repo and `conda env create -n hashFrag -f environment.yml`.

## Workflow

```
hashFrag:
- [ ] Step 1: Choose use case + validate MARKED / BLAST / threshold
- [ ] Step 2: Convert MARKED → unzipped multi-FASTA (train/test/all)
- [ ] Step 3: Run hashFrag (local or SLURM array)
- [ ] Step 4: Verify outputs (non-empty, ID consistency)
- [ ] Step 5: Write report + register artifacts + method-decision
```

### Step 1 — Use case

Ask which of the three primary pipelines applies. Defaults for scoring (BLAST-compatible): `word_size=11`, `gapopen=2`, `gapextend=1`, `penalty=-1`, `reward=1`. Changing scoring changes homology calls — document any non-defaults.

**Modes**

| Mode | Meaning |
|------|---------|
| `lightning` (default) | Use corrected top BLAST alignment scores |
| `pure` | After BLAST candidates, supply exact local alignment scores (`id_i`, `id_j`, `score` TSV) via modules |

Prefer `lightning` unless user requests `pure` / higher recall.

### Step 2 — MARKED → hashFrag FASTA

**Canonical conversion** (document in runner; prefer thin `src/run/…` calling existing helpers — do **not** paste one-off conversion only in chat):

1. Resolve `marked_dir` = `{panel}/MARKED` (must be a directory).
2. Load `{ID: sequence}` with `src.splits.sbs.fna_io.load_fna_directory(marked_dir, ids=…)` (region id = **filename stem**). Optional: `parse_marked_header` on each file to assert header `ID` == stem.
3. Choose ID sets by use case (see table below). Prefer `src.pipeline.common.read_csv` on `split.csv`.
4. Write multi-FASTA records as `>{ID}\n{sequence}` (80-col wrap optional). Headers must be unique; one record per ID.
5. Emit under a run workdir, e.g. `output/hashfrag/<run>/fasta/{train,test,all}.fa` (relative paths).
6. Validate: every requested ID has a sequence; no empty seqs; train ∩ test = ∅ for filter/stratify.

| Use case | FASTA to build |
|----------|----------------|
| `filter_existing_splits` | `train.fa` + `test.fa` from `split.csv` |
| `stratify_test_split` | same |
| `create_orthogonal_splits` | `all.fa` = full MARKED (or user ID subset) |

**`split.csv` bucketing (default when labels exist)**

| `train_test` | Default role |
|--------------|--------------|
| `train` | Subject / train FASTA |
| `test` | Query / test FASTA |
| `val` | **Exclude** from both unless user asks to fold val into train subject |
| `zsv` / zeroshot | **Exclude** (ZSV is held out of train materialization; do not mix into train/test homology sets unless user locks otherwise) |

If `split.csv` is missing and the use case needs train vs test, **stop** and ask for labels or switch to `create_orthogonal_splits` on `all.fa`.

**Reuse, do not reinvent:** loaders already live in `src.splits.sbs.fna_io` and CSV helpers in `src.pipeline.common`. A future thin writer under `src/run/` should call those — do **not** add a parallel MARKED parser.

Keep relative paths under the project root. Seed any sampling with an explicit seed (default **42** unless user locks another; hashFrag orthogonal-split default seed is **21** — record which seed was used).

### Step 3 — Execute

**Filter leakage from existing splits**

```bash
hashFrag filter_existing_splits \
  --train-fasta-path output/hashfrag/<run>/fasta/train.fa \
  --test-fasta-path output/hashfrag/<run>/fasta/test.fa \
  -t 60 \
  -T 16 \
  -o output/hashfrag/<run>/filter_work
```

**Stratify test by homology to train**

```bash
hashFrag stratify_test_split \
  --train-fasta-path output/hashfrag/<run>/fasta/train.fa \
  --test-fasta-path output/hashfrag/<run>/fasta/test.fa \
  -T 16 \
  -o output/hashfrag/<run>/stratify_work
```

**Create orthogonal train/test**

```bash
hashFrag create_orthogonal_splits \
  -f output/hashfrag/<run>/fasta/all.fa \
  -t 60 \
  --p-train 0.8 --p-test 0.2 \
  -n 1 -s 42 \
  -T 16 \
  -o output/hashfrag/<run>/orthogonal_work
```

**Large jobs (SLURM)** — generate array script, then submit (do not babysit; `@monitor` under `/do`):

```bash
hashFrag blastn_array_module \
  --train-fasta-path output/hashfrag/<run>/fasta/train.fa \
  --test-fasta-path output/hashfrag/<run>/fasta/test.fa \
  --query-partition-size 1000 \
  --job-scheduler slurm \
  --job-account "<account>" \
  --job-time "6:00:00" \
  --job-memory "16GB" \
  --num-cpus 4 \
  --environment-path path/to/set_env.sh \
  -o output/hashfrag/<run>/blastn_array.work
sbatch output/hashfrag/<run>/blastn_array.work/hashFrag.blastn_array_module.array_jobs.sh
```

Even CPU counts (16/32). Pass `${SLURM_CPUS_PER_TASK}` into `-T` / `--num-cpus` when writing wrappers.

Log every command + `hashFrag` / `blastn` versions to `data/logs/` or the run outdir.

### Step 4 — Verify

- Output dir non-empty; expected FASTA / TSV / cluster files present.
- Filtered test IDs are a subset of input test IDs; no invented IDs; IDs still match MARKED stems / `split.csv`.
- For orthogonal splits: train ∩ test = ∅ at the homology-cluster level (per hashFrag design); check counts vs `--p-train` / `--p-test`.
- On failure: report paths and stderr; do **not** mark COMPLETED or fabricate leakage metrics.

### Step 5 — Report and registry

Write a short run note (suggest `docs/hashfrag-report.md` or `output/hashfrag/<run>/report.md`) with:

- use case, MARKED path, FASTA paths, threshold, scoring params, seed, software versions
- n train / test before and after (or fold sizes)
- path to workdir outputs

Register artifacts in `docs/artifact-registry.md` (producer: `hashfrag`).  
Append **method-decision** for threshold, mode (`lightning`/`pure`), seed, and any val/zsv inclusion choice when chosen (not user-pre-locked).

## Coordination with this project

| Skill / module | Role |
|----------------|------|
| `/preprocess`, `adapt` | Produce `MARKED/` — **canonical hashFrag sequence source** |
| `/split`, `/split-generate` | `split.csv` for train/test membership; hashFrag can **audit** random/SBS splits or **propose** homology-aware alternatives |
| `/train` | Prefer evaluating on hashFrag-filtered or stratified test sets when leakage is a concern |
| `/dataset-auditor` | Completeness of sequence panels before homology search |
| `src/splits/sbs` | Similarity-based split family — complementary; reuse `fna_io` for MARKED load; do not duplicate SBS distance logic inside hashFrag skill |

If integrating results back into universal `split.csv` / `SPLIT/`, write a `src/run/…` adapter that maps hashFrag ID lists → fold labels via existing `src.pipeline` APIs. Do not silently rewrite validated splits.

**Not MARKED:** mapped training IDs `sample__region` appear in `SPLIT/` after mapped `parse_target` — those are downstream of MARKED region IDs. Run homology on **region** MARKED unless the user explicitly requests mapped-panel sequences.

## Deliverables

| Output | Typical path |
|--------|----------------|
| Converted FASTA | `output/hashfrag/<run>/fasta/{train,test,all}.fa` |
| Work directory | `output/hashfrag/<run>/` |
| Filtered / orthogonal FASTA or ID lists | under workdir (hashFrag layout) |
| Report | `docs/hashfrag-report.md` or run-local `report.md` |
| Method entry | `method-decision.md` |
| Registry row | `docs/artifact-registry.md` |

## Rules

- Primary inputs are **MARKED folders**; convert to multi-FASTA before CLI calls.
- Never claim “no leakage” without hashFrag (or equivalent) evidence and a documented threshold.
- Never treat GEO counts / TPM / PREDICT as sequence homology evidence.
- Never run hashFrag on LegNet-stitched PARSED sequences by default (adapter homology artifact).
- Never skip BLAST+ / install / MARKED existence checks.
- Prefer official CLI over reimplementation.
- For API details, MARKED schema, and module graph, see [reference.md](reference.md).
