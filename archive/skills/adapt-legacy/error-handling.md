# Adapt — error handling

## Critical → abort

| Condition | Action |
|-----------|--------|
| No genomes discovered | Abort; report searched paths |
| Missing FASTA for a required genome | Abort that run (or abort if `--require-all`) |
| Zero accepted windows globally | Abort |
| Invalid `window_size` (`<= 2*flank`) | Abort |
| Cannot write output directory | Abort |

## Recoverable → skip + document

| Condition | Reason code in `excluded_genes.tsv` |
|-----------|-------------------------------------|
| Gene longer than `window_size - 2*flank` | `gene_too_long` |
| Missing TPM for gene | `missing_tpm` |
| Chromosome not in FASTA | `chrom_not_in_fasta` |
| Window outside chromosome | `window_out_of_bounds` |
| Invalid strand | `invalid_strand` |
| Empty / all-N sequence | `empty_or_invalid_sequence` |
| Duplicate gene_id within genome | `duplicate_gene_id` (keep first, skip later) |
| Negative / unordered coordinates after normalize | `invalid_coordinates` |

Never invent TPM, coordinates, or sequences.
