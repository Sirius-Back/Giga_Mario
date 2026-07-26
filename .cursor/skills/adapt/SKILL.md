---
name: adapt
description: >-
  Mandatory stage between raw or split genomic data and Caduceus: audit repo,
  build gene±200bp DNA windows with continuous TPM labels, write adapt/
  Caduceus-ready dataset. Never splits folds. Use for /adapt or Caduceus prep.
disable-model-invocation: true
---

# Adapt

## Goal

`adapt` is the mandatory intermediate stage between raw genomic data (or already split data) and Caduceus training.

Pipeline:

```
Raw data
      │
      ├─ Caduceus-like ──► adapt ──► split (regions + TPM) ──► caduceus
      │
      └─ other / already-split ──► (data/convert as needed)
                                    │
                                    └─ adapt if windows still needed
                                         │
                                         └─ split or align predictions ──► caduceus
```

`@split` owns folds. This skill MUST NOT perform train/validation/test splitting.

It MUST ONLY prepare a Caduceus-ready dataset (gene windows + TPM). When `/split` classifies input as Caduceus-like, it invokes **`@adapt` before** region splitting.

--------------------------------------------------
GENERAL PRINCIPLES
--------------------------------------------------

The skill follows the same philosophy as all existing project skills.

Requirements:

• reproducible
• deterministic
• restartable
• idempotent
• scientific
• fully documented

Never silently guess.

Every assumption must be documented.

Follow project rules: **validation-first**, **missing-data-policy**, **reproducibility**, **scientific-integrity**, **method-decision-tracking**, **artifact-registry**, **slurm-execution-policy**, **task-status**.

--------------------------------------------------
STAGE 1 — REPOSITORY AUDIT
--------------------------------------------------

Determine automatically:

• raw repository
or
• already split repository

Detect:

genomes
annotations
expression tables
metadata
splits

Verify consistency.

Report missing files.

Abort if critical files are missing.

--------------------------------------------------
SUPPORTED INPUTS
--------------------------------------------------

By default use ALL available genomes.

Expected input may include:

FASTA
FNA
GTF
GFF3
TPM tables
gene expression matrices
metadata

Multiple genomes are expected.

Each genome may be located inside a separate directory.

Automatically pair

genome
annotation
expression

using filenames and metadata.

--------------------------------------------------
CURRENT TASK
--------------------------------------------------

Current prediction target:

continuous transcript abundance

Target:

TPM

Prediction input:

DNA sequence only

Sequence generated from

gene
plus
flanking regions

No RNA sequence.

No protein sequence.

--------------------------------------------------
STAGE 2 — VERIFY CADUCEUS FORMAT
--------------------------------------------------

Search online for the latest Caduceus fine-tuning format.

Document:

required file formats
supported labels
recommended sequence length
recommended chunk sizes
required metadata
directory structure
tokenization assumptions

Save summary into

docs/caduceus_format.md

Document all references.

Never hardcode assumptions if documentation changed.

(Re-fetch / refresh this doc when Caduceus upstream changes; do not invent formats.)

--------------------------------------------------
STAGE 3 — BUILD TRAINING WINDOWS
--------------------------------------------------

Training sample = ONE genomic window.

Current baseline strategy (LOCKED):

One window corresponds to one gene.

Window contains:

200 bp upstream
gene body
200 bp downstream

Total window size:

configured by user.

If gene does NOT completely fit inside

window_size - 400 bp

exclude this gene.

No chunking.

No overlapping windows.

No multi-window genes.

No partial genes.

Future strategies may exist but are NOT implemented.

--------------------------------------------------
LONG GENE POLICY
--------------------------------------------------

Baseline v1:

Reject genes longer than

window_size - 2 × 200 bp

Generate

excluded_genes.tsv

including

gene_id
length
reason

Produce summary:

accepted
rejected
percentage
per genome

--------------------------------------------------
WINDOW EXTRACTION
--------------------------------------------------

Automatically respect strand.

Positive strand:

200 bp upstream
gene
200 bp downstream

Negative strand:

extract identical biological region.

Support optional reverse-complement export.

Current default:

export forward orientation only.

Design implementation to allow future RC augmentation.

--------------------------------------------------
TARGET TABLE
--------------------------------------------------

Produce one record per accepted gene.

Each sample contains

sample_id
genome
chromosome
gene_id
coordinates
strand
sequence
TPM
window_length
metadata

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

Generate Caduceus-ready dataset.

Output directory:

adapt/

Include:

manifest.tsv
samples.tsv
labels.tsv
metadata.json
config.yaml
statistics.json
excluded_genes.tsv
README.md

Also write `adapt/caduceus_ready/` (fold-preserving if splits exist) so `/caduceus` can consume sequences + continuous TPM without re-adapting. See [docs/caduceus_format.md](../../docs/caduceus_format.md).

--------------------------------------------------
DOCUMENTATION
--------------------------------------------------

Automatically document

window strategy
context size
filter thresholds
rejected genes
Caduceus assumptions
directory structure

Store:

METHOD_DECISIONS.md
docs/adapt.md
docs/caduceus_format.md

(Also append Locked/Tentative entries to project-root `method-decision.md`.)

--------------------------------------------------
QUALITY CONTROL
--------------------------------------------------

Validate:

matching chromosome names
gene coordinates
window boundaries
duplicate genes
missing TPM
invalid strands
missing FASTA entries
empty sequences
negative coordinates
coordinates outside chromosome

Produce complete QC report.

--------------------------------------------------
FAILURE POLICY
--------------------------------------------------

Never silently continue.

Critical problems:

abort.

Recoverable problems:

skip
document
continue.

--------------------------------------------------
INTEGRATION
--------------------------------------------------

Must integrate seamlessly with

split
and
caduceus

without modifying either skill.

Input accepted from:

raw repository
or
split repository

Output must always be accepted directly by

/caduceus

--------------------------------------------------
EXACT COMMAND
--------------------------------------------------

**Do not reimplement windowing in-chat** — run the script.

```bash
conda run -n caduceus_env python .cursor/skills/adapt/scripts/adapt.py \
  --config .cursor/skills/adapt/scripts/config.default.yaml \
  --input auto \
  --out adapt \
  --window-size 8192
```

| Flag | Default | Notes |
|------|---------|-------|
| `--input` | `auto` | `auto` \| path to split root (`data_splits/full`) \| raw root |
| `--out` | `adapt` | Output directory |
| `--window-size` | from config | Max accepted window length (bp) |
| `--flank` | `200` | Upstream/downstream flank (LOCKED baseline = 200) |
| `--rc-export` | off | If set, also write RC sequences (default: forward only) |
| `--seed` | `42` | Only for deterministic tie-breaks / sampling hooks (no split) |

Heavy panels → wrap in sbatch (even CPUs, mem, time, logs) per **slurm-execution-policy**.

## Workflow checklist

```
adapt:
- [ ] Stage 1: Repository audit (raw vs split); abort on critical gaps
- [ ] Stage 2: Refresh/verify docs/caduceus_format.md against upstream
- [ ] Stage 3: Build gene±flank windows; exclude long genes
- [ ] QC report; excluded_genes.tsv; statistics.json
- [ ] Write adapt/* + adapt/caduceus_ready/*
- [ ] METHOD_DECISIONS.md + docs/adapt.md + method-decision.md + artifact-registry
```

## Additional resources

- [README.md](README.md) — overview + commands
- [description.md](description.md) — scope boundary
- [examples.md](examples.md) — before/after split examples
- [best-practices.md](best-practices.md)
- [error-handling.md](error-handling.md)
- [checklists.md](checklists.md)
- [workflow.md](workflow.md) — Mermaid diagram
- [scripts/adapt.py](scripts/adapt.py) — implementation
- [scripts/config.default.yaml](scripts/config.default.yaml)
