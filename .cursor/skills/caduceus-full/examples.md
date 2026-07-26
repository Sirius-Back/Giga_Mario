# Caduceus-full — examples

## Minimal (no zero-shot)

```text
/caduceus-full
DATA=random
SPLIT_MD=splits/random.md
EPOCHS=10
GPUS=4
```

Expected stages: convert if needed → split1 (TPM, excl none) → split2 on split1 → adapt1 → adapt2 → train TPM + viz → train predict-split1 → report.

## With zero-shot genomes

```text
/caduceus-full
DATA=random
SPLIT_MD=splits/random.md
ZS_GENOMES=GCF_000001405.40,GCF_000001635.27
```

Split-1 raw pool excludes those GCFs. ZS adapt runs in parallel with training; TPM checkpoint evaluates on adapted ZS.

## Smoketest panel

```text
/caduceus-full
DATA=genomes_smoketest
SPLIT_MD=random
```
