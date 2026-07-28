# GigaMario

Toolkit for preparing genomic intervals, linking prediction targets, building leakage-aware train/val/test partitions, and training DNA foundation models (Caduceus, LegNet, …).

Contracts: [wiki/architecture.md](wiki/architecture.md). Skills: [skills.md](skills.md). Split strategy authoring: [wiki/split-generate.md](wiki/split-generate.md).

## Skill pipeline

```mermaid
flowchart TD
    raw[GTF + FNA + TARGET] --> PP["/preprocess"]
    PP --> runnerP["src/run/preprocess_{data}.py"]
    runnerP --> arts["ID.csv · MARKED · PARSED · PREDICT · parse.md"]
    SG["/split-generate"] --> stratCode["src/splits/{id}.py"]
    arts --> SP["/split"]
    stratCode --> SP
    SP --> runnerS["src/run/run_id/{data}_{split}_{direct|adversarial}.py"]
    runnerS --> SPLIT["SPLIT/ + optional ZSV"]
    SPLIT --> TR["/train"]
    TR --> runnerT["src/run/run_id/{data}_{split}_{train}_{direct|adversarial}.py"]
    runnerT --> logs["logs · TensorBoard · figures · final_model"]
    SPLIT --> ADV["/adversarial"]
    ADV --> runnerA["src/run/run_id/{data}_{split}_adversarial.py"]
    runnerA --> ADVpanel["adversarial panel + random SPLIT"]
    ADVpanel --> TRadv["/train mode=adversarial"]
    PIPE["/pipeline dry|run"] --> SP
    PIPE --> TR
    PIPE --> ADV
    PIPE --> TRadv
```

## Stage graph (modules)

```mermaid
flowchart LR
    subgraph prep["Prepare"]
        GTF[(GTF)] --> adapt
        FNA[(FNA)] --> adapt
        adapt --> MARKED[MARKED]
        MARKED --> parse_data
        parse_data --> PARSED[PARSED]
        TARGET[raw TARGET] --> parse_target
        parse_target --> PREDICT[PREDICT]
        IDcsv[ID.csv] --> generate_fold
        generate_fold --> Fold[fold.csv]
    end

    subgraph assign["Assign & materialize"]
        Fold --> SPmod[split-predict]
        SPmod --> SplitCSV[split.csv]
        SplitCSV --> SPLmod[split]
        PARSED --> SPLmod
        PREDICT --> SPLmod
        SPLmod --> Trees["SPLIT/ + ZSV"]
    end

    Trees --> train --> logs2[logs] --> train_viz
```

1. **`/preprocess`** — write-and-exec `preprocess_{data}.py`: get_mpra (optional) → id_gen/id_rule → adapt → parse_data → parse_target → optional generate_fold → `preprocess_report` → `parse.md`.
2. **`/split-generate`** — implement strategies from `splits/*.md` into `src/splits/`.
3. **`/split`** — split-predict + split via `src/run/run_id/…`.
4. **`/train`** — reuse `src.pipeline.train` / caduceus / legnet / train_viz; TensorBoard; optional ZSV eval.
5. **`/adversarial`** — adversarial combine + random split.
6. **`/pipeline`** — `src/run/run_id/pipeline.py` with `dry` (generate/review/smoke) or `run` (execute).

## Install (conda-preferred)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
conda env update -f environment.yml --prune
python -m pip install -e .
```

## Archive

Prior run outputs live under [`archive/results/`](archive/results/) (gitignored). Legacy skills under [`archive/skills/`](archive/skills/).
