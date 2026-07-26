# Adapt — description

## What

Prepare a **Caduceus-ready** dataset of DNA sequence windows labeled with **continuous TPM**, including matched **non-coding** regions.

## What not

- No train/validation/test / zero-shot splitting (that is `@split`)
- No model training (that is `@caduceus`)
- No RNA/protein sequences
- No inventing missing TPM files

## When

Use `/adapt` on `raw/{fna,gtf,tpm}` before Caduceus fine-tuning. Entry: `src/preprocessing.py` → `data_ready/`.
