# hashFrag reference

Upstream: [github.com/de-Boer-Lab/hashFrag](https://github.com/de-Boer-Lab/hashFrag) · Docs: [hashfrag.readthedocs.io](https://hashfrag.readthedocs.io/en/latest/) · PyPI: `hashFrag` (CLI entry `hashFrag`).

## MARKED schema

Canonical project sequence panel for hashFrag. Contracts: `wiki/architecture.md` (stage `adapt`), `src.pipeline.adapt`, `src.pipeline.common`, `src.pipeline.parse_data`, `src.splits.sbs.fna_io`.

### Directory layout

```text
{panel}/
  ID.csv                 # optional for hashFrag; join table
  split.csv              # optional; required for filter/stratify defaults
  MARKED/
    {ID}.fa              # one region per file
    …
  intersect.csv          # adapt pairwise overlaps; not a hashFrag input
```

Verified panel roots in this workspace (examples): `ready_caduceus/MARKED`, `ready_legnet/MARKED`, `run/run0/MARKED`.

### Per-file FASTA

| Property | Value |
|----------|-------|
| Suffixes | `.fa` (adapt default); loaders also accept `.fasta` / `.fna` / `.fas` in SBS |
| Records | Exactly one `>` record per file (`parse_data.read_marked_fasta`) |
| Filename | `{sanitize_filename(ID)}.fa` — stem is the region id |
| Header fields | `genome\|chr\|pos1\|pos2\|gene_nameORnon_coding_ID\|raw_target_ID\|ID` |
| Header form | `>` + leading `|` + seven fields joined by `\|` |
| Writer | `write_fasta_record` in `src.pipeline.common` |
| Parser | `parse_marked_header` (requires ≥7 pipe fields) |

Concrete example:

```text
>|GCF_000009045.1|NC_000964.3|2846961|2847161|nadA|BSU_27850|10000
TTAATGGCGGCAACGCGCGTTTCCATATCCTTTCTCGATAGTTCTTTATAACTTTCGGGCATCATATCATTCGATTGTTT
…
```

Sequence length equals the adapt window (not fixed by the MARKED contract). Gene ±100 → 200 bp is common for prokaryote panels; Caduceus legacy windows may differ. LegNet `parse_data` requires 200 bp CRS and may **skip** non-200 MARKED files — that skip policy is for PARSED, not for hashFrag.

### ID conventions

| Source | ID meaning |
|--------|------------|
| `MARKED/{ID}.fa` stem | Region id (preferred join key for hashFrag headers) |
| Trailing FASTA header field | Same `ID` when adapt wrote the file |
| `ID.csv` column `ID` | Same region id |
| `split.csv` column `ID` | Same region id (`ID\|train_test\|fold`) |
| `SPLIT/` mapped `sample__region` | **Downstream** composite; not the MARKED stem |

SBS loader rule (`_region_id_from_header`): if loading a multi-FASTA later, prefer the **trailing** pipe field; for MARKED directories, use **filename stem**.

### Train / test / fold relationship

`MARKED/` itself has **no** train/test labels. Membership comes from:

- `split.csv` (`train` / `test` / `val` / `zsv`) produced by `split-predict`, or
- hashFrag `create_orthogonal_splits` on a consolidated `all.fa`.

Default conversion bucketing: train → `train.fa`; test → `test.fa`; **exclude** `val` and `zsv` unless the user locks otherwise.

### What MARKED is not

| Artifact | Why not for hashFrag (default) |
|----------|--------------------------------|
| `PARSED/*.ext` (Caduceus) | Same DNA as MARKED in spirit, but not the panel contract path |
| `PARSED/*.ext` (LegNet) | 230 bp with fixed adapters — artificial shared flanks |
| `PREDICT/` / TPM | Labels, not sequences |
| `SPLIT/FASTA/…` | Post-split trees; may use mapped composite IDs — prefer MARKED + `split.csv` |

---

## MARKED → multi-FASTA algorithm

Agents implement this in a thin `src/run/…` runner (call existing helpers; do not invent a second MARKED parser).

```text
inputs:  marked_dir, optional split_csv, use_case
outputs: train.fa / test.fa / all.fa under output/hashfrag/<run>/fasta/

1. Assert marked_dir.is_dir() and has *.fa|*.fasta
2. seqs = load_fna_directory(marked_dir, ids=optional_subset)
   # from src.splits.sbs.fna_io — keys = filename stems
3. Optional integrity: for each path, parse_marked_header → ID == stem
4. If use_case needs train/test:
     rows = read_csv(split_csv)  # src.pipeline.common; delimiter |
     train_ids = {r["ID"] for r in rows if r["train_test"] == "train"}
     test_ids  = {r["ID"] for r in rows if r["train_test"] == "test"}
     # default: ignore val / zsv
     assert train_ids ∩ test_ids == ∅
     write multi-FASTA train.fa / test.fa with headers ">ID"
5. If use_case is orthogonal splits:
     write all.fa for all seqs (or user ID list) with headers ">ID"
6. Fail early on missing IDs, empty sequences, or duplicate headers
```

**Header policy:** `>{stem_ID}` only. Pipe metadata stays in MARKED source files for audit; hashFrag/BLAST IDs must match `split.csv`.

**RC policy:** MARKED is single-strand; omit `--skip-revcomp` so hashFrag can add `_Reversed` mates.

---

## Pipeline → modules

| Pipeline | Modules (in order) |
|----------|--------------------|
| `filter_existing_splits` | `blastn_module` → `process_blast_results_module` → `filter_candidates_module` → `filter_test_split_module` |
| `stratify_test_split` | `blastn_module` → `process_blast_results_module` → `stratify_test_split_module` |
| `create_orthogonal_splits` | `blastn_module` → `process_blast_results_module` → `filter_candidates_module` → `identify_homologous_groups_module` → `create_orthogonal_splits_module` |

Each module: `hashFrag <module> -h`.

## Shared BLAST arguments (common)

| Flag | Default | Notes |
|------|---------|-------|
| `-w/--word-size` | 11 | Exact match seed length |
| `-g/--gapopen` | 2 | Positive (BLAST convention) |
| `-x/--gapextend` | 1 | Positive |
| `-p/--penalty` | -1 | Mismatch (negative) |
| `-r/--reward` | 1 | Match |
| `-m/--max-target-seqs` | 500 | Max hits per query |
| `-e/--e-value` | 10.0 | |
| `-d/--dust` | `no` | `{yes,no}` |
| `-T/--threads` | 1 | Prefer 16/32 on HPC |
| `--skip-revcomp` | off | Set only if FASTA already has `_Reversed` mates |
| `--force` | off | Overwrite blastn outputs |
| `-o/--output-dir` | `.` | Work directory |

hashFrag **corrects** BLAST gap scoring for lightning mode (gap open should not double-count extend). See installation docs note on corrected alignment scores.

## Use-case-specific flags

### `filter_existing_splits`

- `--train-fasta-path`, `--test-fasta-path` (required)
- `-t/--threshold` (required) — alignment score ≥ threshold ⇒ homologous

### `stratify_test_split`

- `--train-fasta-path`, `--test-fasta-path`
- `-s/--step` (default 10) — score bin width
- Stratified bins need not be balanced

### `create_orthogonal_splits`

- `-f/--fasta-path` — full population FASTA (`all.fa` from MARKED)
- `-t/--threshold` (required)
- `--p-train` / `--p-test` (default 0.8 / 0.2)
- `-n/--n-splits` (default 1)
- `-s/--seed` (hashFrag default **21**; this project often uses **42** — record explicitly)

Uses union-find over homologous pairs, then proportionally assigns clusters to splits.

### `create_orthogonal_folds_module`

```bash
hashFrag create_orthogonal_folds_module -i $HOMOLOGY_PATH -f 10 -o $OUTPUT_DIR
```

- `-i` — homologous groups file from `identify_homologous_groups_module`
- `-f` — number of folds
- Greedy fill by cluster size; unstable if huge clusters ≫ target fold size

## Pure mode (exact scores)

1. Run BLAST + `process_blast_results_module` for candidates.
2. Compute Smith–Waterman (or other) scores for candidate pairs.
3. Provide TSV with columns: `id_i`, `id_j`, `score` (tab-separated).
4. Continue with `filter_candidates_module` / downstream modules using that TSV.

Higher recall than lightning; higher cost.

## HPC: `blastn_array_module`

Partitions query FASTA (`--query-partition-size`), builds DB on full subject set, emits scheduler array script.

Required extras:

- `--job-scheduler` `slurm` | `sge`
- `--environment-path` — shell script sourced in each task (activate env, load BLAST+, PATH)
- SLURM: `--job-account`, `--job-time`, `--job-memory`, `--num-cpus`

After array completes, concatenate/process blast outputs and continue the module chain for the chosen use case.

## Reverse complement convention

- Default: hashFrag adds RC records with header suffix `_Reversed`.
- If both orientations already present, every RC header must use `_Reversed`, and pass `--skip-revcomp`.
- Warning if `_Reversed` exists but `--skip-revcomp` is omitted.
- MARKED panels do not include RC mates by default.

## Threshold calibration (recommended when unknown)

Compute pairwise alignment scores on **dinucleotide-shuffled** (or other null) sequences with the **same** scoring params; set `-t` above the null distribution. Document calibration set size and chosen quantile/cutoff in `method-decision.md`.

## Project adapter checklist

When bridging this repo:

1. Start from `MARKED/`; emit unzipped multi-FASTA with headers `>{ID}` matching `split.csv` / `ID.csv`.
2. Run hashFrag into `output/hashfrag/<run>/`.
3. Map filtered or reassigned IDs back via `src.pipeline` / `id_rule` — do not invent fold labels.
4. Register outputs; pytest any new `./src` conversion helpers.

## Open / ambiguous points

| Topic | Safest default | Ask user when |
|-------|----------------|---------------|
| Include `val` in train subject for filter/stratify | Exclude | User wants val homology-checked against test |
| Include `zsv` | Exclude | User wants ZSV leakage audit |
| Homology on PARSED LegNet 230 bp | No — use MARKED | User explicitly requests adapter-inclusive search |
| Filename stem ≠ header `ID` | Stop / fail | Never auto-pick |
| Mapped `sample__region` panels | Homology on region MARKED + map results | User supplies mapped FASTA explicitly |
