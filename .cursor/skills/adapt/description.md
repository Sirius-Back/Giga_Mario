# Adapt — description

## What

Prepare a **Caduceus-ready** dataset of DNA sequence windows labeled with **continuous TPM**.

## What not

- No train/validation/test / zero-shot splitting (that is `@split`)
- No model training (that is `@caduceus` + `@do-fast`)
- No RNA/protein sequences
- No gene chunking / multi-window genes / partial genes (baseline v1)

## When

Use `/adapt` after genomes+TPM are available (raw or already under `data_splits/`), and **before** Caduceus fine-tuning on transcript abundance.
