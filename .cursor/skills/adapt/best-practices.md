# Adapt — best practices

1. **Lock `window_size` in `method-decision.md`** before large runs; it controls long-gene rejection.
2. Prefer **split-then-adapt** so `caduceus_ready/{train,val,test}` mirrors `@split` folds.
3. Always inspect `excluded_genes.tsv` + `statistics.json` before training.
4. Refresh `docs/caduceus_format.md` when Caduceus/GenomicBenchmarks docs change.
5. Keep flank = **200** unless the user explicitly unlocks a new baseline.
6. Use `caduceus_env` (pyfaidx) for extraction; record software versions in `metadata.json`.
7. Do not binarize TPM inside `@adapt` unless the user explicitly requests a separate label mode — baseline target is **continuous TPM**.
