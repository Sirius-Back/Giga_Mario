---
name: caduceus
description: >-
  Caduceus DNA LM (kuleshov-group/caduceus): HF checkpoints, tokenization,
  embeddings, RC/Ph/PS, pretrain/fine-tune, GenomicBenchmarks / Nucleotide
  Transformer / eQTL-VEP; TPM/expression epoch logs follow metrics.md
  (TorchMetrics). Depends on @split; executes via @do-fast. Use for Caduceus
  inference, training, fine-tuning, or repo reproduction.
disable-model-invocation: true
---

# Caduceus DNA language model

Caduceus (Schiff et al., 2024, arXiv:2403.03234) is a bi-directional, reverse-complement-equivariant
DNA language model built on the Mamba (SSM) architecture, pretrained on the human reference genome
at sequence lengths up to 131,072 bp.

- Repo: https://github.com/kuleshov-group/caduceus
- HF collection: https://huggingface.co/collections/kuleshov-group/caducues-65dcb89b4f54e416ef61c350
- Paper: https://arxiv.org/abs/2403.03234

Treat the GitHub README, the files in the repo, and the HF model cards as the source of truth over
anything below if they've since changed — check them before assuming a config value, checkpoint name,
or Hydra key.

## Orchestration (required)

This skill supplies **Caduceus-specific facts** (checkpoints, Hydra keys, RC/Ph/PS, env pins). It does
**not** own data splitting or project-level execution looping.

| Dependency | Role | When |
|------------|------|------|
| **`@split`** | **Required** for any train / val / test / zero-shot partitioning | Before fine-tuning, pretraining data folds, GenomicBenchmarks 90/10, NT splits, eQTL-VEP splits, custom datasets. Pass **data** + **split** (`splits/<id>.md`). Do not invent splits or set `dataset.train_val_split_seed` ad hoc. |
| **`@do-fast`** | **Required** execution engine for multi-step Caduceus work | Env setup, clone, pretrain, fine-tune, VEP pipeline, SLURM jobs, anything that needs verify-todo / review / monitor / sync until exit. Read `@do-fast` and invoke it as that skill requires (one orchestration prompt). |

**Order:** resolve data + split strategy → invoke **`@split`** (which itself runs via `@do-fast`) → then for Caduceus train/eval/reproduce beyond folds, materialize todos and invoke **`@do-fast`** with Caduceus overrides (this skill’s facts + `model-train.mdc`). Do not reimplement `@do-fast` or `@split` here.

CV / LOO / other training extras: implement in **both** `@split` and this skill when the model supports them (`AGENTS.md`).

One-shot HF load/embed answers may use Part 1 facts in-chat; any project pipeline still goes through `@split` (if folds needed) and `@do-fast`.

## Project conventions — this repo's AGENTS.md

This project is governed by its own `AGENTS.md` at the repo root. This skill supplies Caduceus-specific
technical facts; it does not replace that governance layer — read `AGENTS.md` and follow it alongside
this skill for any Caduceus task, inference or training.

- `AGENTS.md`'s conditional rules route model training/fine-tuning tasks — Caduceus included — to
  `.cursor/rules/model-train.mdc`. Load and follow that rule together with this skill whenever the
  task is training/fine-tuning (the "Repo reproduction" half below). Where they overlap,
  `model-train.mdc` governs *process* (how a run is launched, tracked, and reported); this skill
  supplies *Caduceus-specific facts* (env pins, Hydra keys, checkpoint/RC semantics); **`@do-fast`**
  runs the process until exit.
- Data folds: **`@split`** + `splits/*.md` (see `AGENTS.md` `splits/{}.md` shape) — never ad-hoc
  split scripts or silent 90/10 defaults without `@split`.
- **Metrics logging:** for expression / continuous-TPM (and any regression) training, log every
  epoch using project **`metrics.md`** (TorchMetrics). See [Metrics logging](#metrics-logging-required).
  Do not invent alternate metric sets or average batch-wise correlations manually.
- Check `todo.md` for the task's current status before starting, and update it as you go
  (`TODO → READY → RUNNING → BLOCKED / FAILED / RECOVERABLE → COMPLETED / SKIPPED`).
- Log the reasoning behind nontrivial choices — Ph vs. PS checkpoint, `bidirectional_strategy`,
  pooling strategy, hyperparameters, which benchmark/config to fine-tune against — in
  `method-decision.md`.
- Register every deliverable this work produces (checkpoints, extracted embeddings,
  `model_config.json`, plots, reports) in the artifact registry: prefer `docs/artifact-registry.md`
  if a `docs/` directory exists, otherwise `artifact-registry.md` at the root.
- Write job-monitoring output — pretraining/fine-tuning progress, SLURM job status, loss curves — to
  `monitoring-report.md` (and let `@do-fast` / `@monitor` own long job supervision).

## Don't reimplement what's already in the repo

Before writing any new script, dataloader, config, notebook cell, or model-loading/embedding/scoring
code for a Caduceus task, search the repo first (`src/`, `caduceus/`, `configs/`, `slurm_scripts/`, and
any project-specific scripts or notebooks outside the upstream repo) for an existing implementation
that already does it. Extend or parameterize what exists rather than writing a parallel version — this
codebase already has dataloaders, Hydra configs, and SLURM scripts for the standard
pretraining/fine-tuning/eQTL-VEP workflows described below; a new one should only be added for a
genuinely new task shape that nothing here covers.

---

# Metrics logging (required)

**Source of truth:** project-root [`metrics.md`](../../../metrics.md).

For **expression / continuous TPM** fine-tunes (this project's primary Caduceus target), every
selected split (`train`, `validation`, `test`; plus `zero-shot-validation` only if requested) must
log **each epoch**:

```text
{split}_loss
{split}_pearson
{split}_spearman
{split}_mse
{split}_rmse
{split}_mae
{split}_r2
{split}_genewise_pearson_median
{split}_samplewise_pearson_median
```

Rules (from `metrics.md`):

- Use **TorchMetrics** (`MetricCollection` of Pearson, Spearman, MSE, RMSE, MAE, R²).
- Compute metrics **once per epoch** after the full loader — do **not** average batch-wise
  correlations.
- `val_loss` (validation loss) is the early-stopping / checkpoint metric unless the user Locks
  another (e.g. `pearson`).
- Gene-/sample-wise median Pearson: use the helpers in
  [scripts/metrics_logging.py](scripts/metrics_logging.py). If predictions are scalar TPM per
  region `(N,)`, treat as `(N,1)`: `genewise_pearson_median` is defined; `samplewise_pearson_median`
  is `nan` when fewer than 2 genes (document in the epoch payload).
- Persist epoch payloads under the run dir (e.g. `runs/<name>/epoch{N}/metrics.json`) and a
  human `metrics.log` line; register artifacts.

**Classification-only** GenomicBenchmarks / NT runs may log task accuracy in addition, but any
**regression / TPM** head must still emit the `metrics.md` suite — never replace it with accuracy-only
logging.

Exact helpers:

```python
from metrics_logging import (
    build_regression_metrics,
    compute_epoch_regression_metrics,
    format_epoch_log,
)
# Prefer: .cursor/skills/caduceus/scripts/metrics_logging.py
```

---

# Part 1 — Using a pretrained checkpoint (Hugging Face)

Use this part for inference, embeddings, or fine-tuning on your own downstream task using plain
`transformers` code, with no need to clone the source repo.

## Released checkpoints (as of the current HF collection)

Two variants at seqlen 131k / d_model 256 / n_layer 16 (~7.73M params each) — always check the
collection page for other sizes before assuming these are the only ones:

| Checkpoint | RC handling |
|---|---|
| `kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16` | **Ph** ("post-hoc"): trained with RC data augmentation, **not** internally RC-equivariant. |
| `kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16` | **PS** ("parameter-sharing"): internally RC-equivariant (`rcps=True`), no RC augmentation needed. |

**Ph-specific inference rule (from the HF model card):** for downstream tasks, run the model once on
the sequence and once on its reverse complement, then average the two outputs. Skipping this step is
a real accuracy regression in the paper's own ablations, not just a nicety.

## Loading (always needs `trust_remote_code=True`)

The model ships custom modeling code, not upstreamed into `transformers`. Both the tokenizer *and*
the model need the flag — people commonly remember it only for the model:

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_name = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
```

To initialize a fresh (untrained) Caduceus with the same architecture, to train on your own data,
still without cloning the repo:

```python
from transformers import AutoConfig, AutoModelForMaskedLM

config_overrides = {}  # e.g. {"n_layer": 8}
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, **config_overrides)
model = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)
```

## Tokenizer (`CaduceusTokenizer`)

Character-level, **one token per base pair** — this is not a BPE/subword tokenizer, so a 131,072 bp
sequence is a 131,072-token sequence. Budget GPU memory and batch size on bp count directly.

- Default alphabet: `A, C, G, T, N`. Any other character becomes `[UNK]`.
- Special token ids: `[CLS]=0, [SEP]=1, [BOS]=2, [MASK]=3, [PAD]=4, [RESERVED]=5, [UNK]=6`; nucleotide
  ids start at 7.
- Default `padding_side="left"`.
- A `complement_map` (default `{"A":"T","C":"G","G":"C","T":"A","N":"N"}`) is used internally to
  compute reverse complements at the token-id level for the RC-equivariant (PS) layers.

## `CaduceusConfig` fields you'll actually touch

- `d_model`, `n_layer`, `vocab_size` — standard.
- `bidirectional` (bool) — whether the SSM runs forward and backward.
- `bidirectional_strategy` — how the two directions are combined; the released checkpoints use
  `"add"`.
- `bidirectional_weight_tie` (bool) — tie forward/backward layer weights.
- `rcps` (bool) — whether the model uses reverse-complement parameter sharing (`True` for PS
  checkpoints, `False` for Ph).
- `complement_map` (dict) — required by the RCPS layers when `rcps=True`.

## Model classes (in `modeling_caduceus.py`)

- `Caduceus` — base backbone.
- `CaduceusForMaskedLM` — what `AutoModelForMaskedLM` resolves to; use for MLM scoring / zero-shot.
- `CaduceusForSequenceClassification` — classification head, used for GenomicBenchmarks / Nucleotide
  Transformer-style downstream tasks.

## Embeddings and variant-effect scoring

For anything beyond fill-mask, don't reach for the `pipeline()` helper — call the model directly with
`output_hidden_states=True` (or use the base `Caduceus` backbone) and pool the hidden states yourself
(mean pooling or `[CLS]`, matching whatever your downstream head expects).

The repo's own eQTL variant-effect pipeline (see Part 2) is a useful pattern to mirror: extract
*frozen* embeddings for reference and alternate alleles, then fit a lightweight downstream model (the
paper uses an SVM) on top — Caduceus itself isn't fine-tuned per-variant in that workflow.

## Inference gotchas

- Missing `trust_remote_code=True` on the tokenizer specifically (not just the model).
- Treating "sequence length" as anything other than raw bp count when setting batch size.
- Using a Ph checkpoint on a downstream task without also embedding/scoring the reverse complement
  and combining the two.
- Assuming inference beyond ~131,072 bp works well — that's the length the released checkpoints were
  actually trained at; shorter is fine (SSM, not a fixed attention window), longer is unproven.
- Compiled CUDA kernels (`mamba-ssm`, `causal-conv1d`) underlie this model even at inference time —
  if you hit build/import errors, see Part 2 for the exact pinned dependency versions used upstream.

---

# Part 2 — Repo reproduction (pretraining / fine-tuning)

Use this part when cloning or working inside the `kuleshov-group/caduceus` repo itself: environment
setup, launching `train.py` (Hydra CLI), reproducing GenomicBenchmarks/Nucleotide Transformer/eQTL-VEP
experiments, or debugging `mamba-ssm`/`causal-conv1d`/`flash-attn` build issues.

**Execution:** run Part 2 pipelines through **`@do-fast`**. **Folds:** obtain via **`@split`** before
setting Hydra split seeds or writing fold dirs. This section is the Caduceus fact sheet for those
skills — not a license to bypass them.

Before overriding any Hydra config key or assuming a script's arguments, open the actual file under
`configs/` or the script itself in this repo — the CLI silently accepts unknown dotted keys with a
`+` prefix but errors without it if the key doesn't already exist in the base config, so guessing is
a fast way to get a silently-wrong run.

## Repo layout (top level)

```
assets/            images etc.
caduceus/           the HF-format model package
  configuration_caduceus.py
  modeling_caduceus.py
  modeling_rcps.py
  tokenization_caduceus.py
  tests/
configs/            Hydra YAML configs (experiment/, model/, dataset/, etc.)
slurm_scripts/       sample SLURM batch scripts
src/                 training infrastructure (inherited from HyenaDNA repo)
  callbacks/  dataloaders/  models/  ops/  tasks/  utils/
train.py             main Hydra entry point
vep_embeddings.py     step 1 of the eQTL VEP pipeline (embedding extraction)
vep_svm.ipynb         step 2 of the eQTL VEP pipeline (SVM fit on embeddings)
caduceus_env.yml      pinned conda environment
setup_env.sh
```

## Environment setup

```bash
conda env create -f caduceus_env.yml
conda activate caduceus_env
mkdir outputs watch_folder
```

Pinned versions in `caduceus_env.yml` as of the current README — **re-check this file in the repo
before assuming these haven't moved**:

- Python 3.8
- PyTorch 2.2.0 (+ matching torchaudio/torchvision/torchtext/torchdata), `pytorch-cuda=12.1`
  (`cuda-nvcc=11.7.99` is also pinned alongside it — this mixed pinning is intentional in the repo,
  not a typo to "fix").
- `transformers==4.38.1`, `hydra-core==1.3.2`, `pytorch-lightning==1.8.6`, `omegaconf==2.3.0`
- `mamba-ssm==1.2.0.post1`, `causal-conv1d===1.2.0.post2`, `flash-attn==2.5.6`

### GPU / build gotchas

`mamba-ssm`, `causal-conv1d`, and `flash-attn` are **compiled CUDA extensions**, not universal
pure-Python wheels. If `conda env create` fails, or the extensions fail to import at runtime:

- Confirm `nvcc` is on `PATH` and its version lines up with the pinned torch/CUDA build.
- These packages are sensitive to the exact torch version. If torch gets upgraded after the fact
  (e.g. by another package), reinstall `mamba-ssm`/`causal-conv1d` afterward.
- On an **L40S** (Ada Lovelace, compute capability 8.9), CUDA 11.8+ is required for that arch — the
  pinned `pytorch-cuda=12.1` satisfies this, so the pin itself should be fine; check `nvidia-smi`'s
  reported driver/CUDA version is new enough before spending time on the build itself.

## Pretraining entry point (`train.py`)

This is a **Hydra CLI**, not argparse — config groups are composed with `key=value` and existing
nested config values are overridden with dotted paths:

```bash
python -m train \
  experiment=hg38/hg38 \
  callbacks.model_checkpoint_every_n_steps.every_n_train_steps=500 \
  dataset.max_length=1024 \
  dataset.batch_size=1024 \
  dataset.mlm=true \
  dataset.mlm_probability=0.15 \
  dataset.rc_aug=false \
  model=caduceus \
  model.config.d_model=128 \
  model.config.n_layer=4 \
  model.config.bidirectional=true \
  model.config.bidirectional_strategy=add \
  model.config.bidirectional_weight_tie=true \
  model.config.rcps=true \
  optimizer.lr="8e-3" \
  train.global_batch_size=1024 \
  trainer.max_steps=10000 \
  wandb=null
```

- `wandb=null` disables Weights & Biases logging if you don't want to authenticate against it.
- `model.config.rcps=true` + `dataset.rc_aug=false` is the **PS** (RC-equivariant) recipe.
  `model.config.rcps=false` + `dataset.rc_aug=true` is closer to the **Ph** recipe.
- Sample SLURM scripts live in `slurm_scripts/` (`run_pretrain_caduceus.sh`, etc.) — adapt an existing
  one to your cluster's partition/paths rather than writing a SLURM script from scratch.

### Pretraining data (hg38)

Data-download instructions are copied from the HyenaDNA repo. Needs two files under `data/hg38/`:

```bash
mkdir -p data/hg38/
curl https://storage.googleapis.com/basenji_barnyard2/hg38.ml.fa.gz > data/hg38/hg38.ml.fa.gz
gunzip data/hg38/hg38.ml.fa.gz
curl https://storage.googleapis.com/basenji_barnyard2/sequences_human.bed > data/hg38/human-sequences.bed
```

Note this comes from Google Cloud Storage (Basenji's bucket), not UCSC/Ensembl directly.

## Downstream fine-tuning experiments

All three reuse the same `train.pretrained_model_path=<.ckpt>` /
`model.config_path=<model_config.json from the pretraining run>` pattern, plus
`model._name_=dna_embedding_caduceus`.

**GenomicBenchmarks** (Grešová et al. 2023 — 8 classification tasks, e.g.
`dummy_mouse_enhancers_ensembl`):

```bash
python -m train \
    experiment=hg38/genomic_benchmark \
    dataset.dataset_name="dummy_mouse_enhancers_ensembl" \
    dataset.train_val_split_seed=1 \
    dataset.batch_size=256 \
    dataset.rc_aug=false \
    +dataset.conjoin_train=false \
    +dataset.conjoin_test=false \
    model=caduceus \
    model._name_=dna_embedding_caduceus \
    +model.config_path="<path to model_config.json>" \
    +model.conjoin_test=false \
    +decoder.conjoin_train=true \
    +decoder.conjoin_test=false \
    optimizer.lr="1e-3" \
    trainer.max_epochs=10 \
    train.pretrained_model_path="<path to .ckpt file>" \
    wandb=null
```

- `dataset.conjoin_train`/`dataset.conjoin_test`: whether the dataset returns a single sequence or
  the sequence concatenated with its reverse complement along a new axis, for train vs. test/val.
- `decoder.conjoin_train`/`decoder.conjoin_test`: whether the decoder head expects
  `(batch, seq_len, d_model)` or `(batch, seq_len, d_model, 2)`. When `true`, the decoder runs
  separately on `input[..., 0]` and `input[..., 1]` and the two predictions are averaged.
- Only a train/test split exists upstream, so the repo can do a train/val split via
  `dataset.train_val_split_seed`, with early stopping on validation accuracy, repeated over 5 seeds —
  **invoke `@split` first** (e.g. `splits/random.md` + this dataset) and consume its `train/` /
  `val/` / `test/` (and zero-shot if requested). Do not reuse the paper's 90/10 default without `@split`.
  Launch the fine-tune itself through **`@do-fast`**.

**Nucleotide Transformer benchmark** (Dalla-Torre et al. 2023; data:
`InstaDeepAI/nucleotide_transformer_downstream_tasks` on HF): same pattern, with
`experiment=hg38/nucleotide_transformer`. Same note — **`@split` then `@do-fast`**; do not accept the
task default split without `@split`.

**eQTL SNP variant effect prediction** (Long Range Benchmark; data:
`InstaDeepAI/genomics-long-range-benchmark` on HF) — a **two-step, train-free-on-Caduceus** pipeline
(run under **`@do-fast`**; folds via **`@split`**):

1. Extract frozen embeddings with `vep_embeddings.py` (uses `torchrun` for distributed extraction):
   ```bash
   torchrun --standalone --nnodes=1 --nproc-per-node=8 \
     vep_embeddings.py \
       --num_workers=2 --seq_len=131072 --bp_per_token=1 --embed_dump_batch_size=1 \
       --name="caduceus-ps_downstream-seqlen=131k" \
       --model_name_or_path="kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16" \
       --rcps
   ```
   Pass `--rcps` for RC-equivariant (PS-type) checkpoints, `--no-rcps` otherwise.
2. Fit an SVM on the dumped embeddings in `vep_svm.ipynb` — train/test folds from **`@split`**, not ad hoc.

## Coordination

| Skill / doc | Role |
|-------------|------|
| `@split` | Required dependency — all folds / strategies under `splits/*.md` |
| `@adapt` | Caduceus-ready windows + TPM before region split when Caduceus-like |
| `@do-fast` | Required execution engine for Caduceus pipelines |
| `model-train.mdc` | Training process (epochs, checkpoint, monitor duration) |
| `metrics.md` | **Required** TorchMetrics suite for expression/TPM epoch logging |
| `scripts/metrics_logging.py` | Shared implementation of `metrics.md` |
| `@monitor` | Invoked only via `@do-fast` for long jobs |
| `@train-viz` | Plots logged metrics from `logs/*` / run epoch JSON |

## Sources

Everything above is drawn directly from the repo's `README.md`, `caduceus_env.yml`,
`caduceus/configuration_caduceus.py`, `caduceus/tokenization_caduceus.py`, the HF model cards,
and project `metrics.md` — re-check those files if behavior here seems out of date.
