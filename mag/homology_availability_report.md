# Homology availability report (raw 11 mammals × mag)

- **Generation date:** 2026-07-31
- **Producer:** agent (mag exploration + ensembl downloader)
- **Scope:** ortholog / paralog markup for the 11 species present in `./raw`

## Summary

| Source | Orthologs | Paralogs | Status for 11 raw species |
|--------|-----------|----------|---------------------------|
| Ensembl Compara TSV (release 116) | yes (`ortholog_*`) | yes (`*_paralog`) | **Present locally for 11/11** after re-download |
| OrthoDB `odb12v2_OG2genes.tab.gz` | yes (shared OG across taxids) | yes (proxy: multi-copy same taxid in OG) | **File present but truncated**; all 11 taxids appear in readable prefix |
| `mag/orthodb/` downloader | — | — | **Stub only** (no download implementation) |
| Ensembl `xref` dirs | ID mapping, not homology | — | Empty locally; FTP path `xref/` empty — real maps under `tsv/<species>/` |

## 1. Ensembl Compara

### Pre-fix finding

- Configured type `compara_homology` pointed at FTP `…/compara/homology/<species>/` — directory exists but is **empty**.
- Real dumps live at `…/tsv/ensembl-compara/homologies/<species>/`.
- Local `compara_homology/` folders existed for most species but contained **0 files**.
- `ovis_aries` was missing from `mag/ensembl/config/config.yaml` and had no local folder.

### Fix applied

- Updated `mag/ensembl/src/config.py`: `compara_homology.subdir` → `tsv/ensembl-compara/homologies`.
- Updated `mag/ensembl/src/downloader.py`: prefer `protein_default` over `ncrna_*` / breed collections.

### Download executed

```bash
cd mag/ensembl
python main.py \
  --species homo_sapiens,mus_musculus,sus_scrofa,capra_hircus,bos_taurus,equus_caballus,canis_lupus_familiaris,ovis_aries,rattus_norvegicus,macaca_mulatta,oryctolagus_cuniculus \
  --data-types compara_homology --release 116 --parallel 2
```

Result: **11 success, 0 errors** (see `queue.md` entry `ensembl-compara-homology-mammals11`).

Local files:

`mag/ensembl/data/<species>/compara_homology/Compara.116.protein_default.homologies.tsv.gz`

### Markup verification (`homology_type` counts)

| Species | Rows | Ortholog rows | Paralog rows | Size (MB) |
|---------|------|---------------|--------------|-----------|
| homo_sapiens | 3 878 214 | 3 737 041 | 141 164 | 109.5 |
| mus_musculus | 4 522 853 | 4 148 949 | 373 873 | 111.9 |
| rattus_norvegicus | 4 503 375 | 4 102 414 | 400 951 | 111.0 |
| sus_scrofa | 4 360 924 | 3 930 614 | 429 092 | 107.5 |
| bos_taurus | 3 970 754 | 3 741 768 | 228 972 | 101.9 |
| ovis_aries | 3 970 784 | 3 821 322 | 149 442 | 103.1 |
| equus_caballus | 4 262 878 | 3 973 627 | 286 575 | 107.3 |
| canis_lupus_familiaris | 3 185 877 | 3 016 545 | 169 306 | 82.9 |
| macaca_mulatta | 3 303 892 | 2 512 198 | 791 517 | 81.0 |
| oryctolagus_cuniculus | 2 564 155 | 2 266 588 | 297 464 | 64.0 |
| capra_hircus | 791 546 | 574 395 | 217 034 | 17.9 |

Observed types (all 11): `ortholog_one2one`, `ortholog_one2many`, `ortholog_many2many`, `within_species_paralog`, `other_paralog`, rarely `gene_split`.

**Note (Ensembl README):** each species-specific TSV is an arbitrary half of pairwise orthologies; for complete A↔B edges download both species files (or the all-species top-level dump).

### Still missing / incomplete (Ensembl auxiliary)

| Item | Status |
|------|--------|
| pep / gtf / gff3 for `sus_scrofa`, `rattus_norvegicus`, `oryctolagus_cuniculus` | Empty local dirs (not required for homology graph itself) |
| `ovis_aries` in default config species list | Still absent — downloaded via CLI `--species` |
| `xref` type | Wrong FTP path; not re-downloaded in this pass |
| abinitio-only gtf for some species | Tiny/placeholder files (e.g. mouse abinitio 110 B) — separate from Compara |

## 2. OrthoDB

### Present

| Path | Role |
|------|------|
| `mag/orthoDB/odb12v2_OG2genes.tab.gz` | OG → gene mapping (ortholog groups) |
| `mag/orthodb/` | Stub CLI only |

Columns: `OG_id`, `gene_id` where `gene_id` ≈ `{ncbi_taxid}_…`.

### Integrity

- Local size: **2.37 GiB** (`2372403200` bytes).
- Official dump size (OrthoDB v12 listing): **~4.5 GB**.
- `gzip` raises `EOFError` after **382 385 653** lines — file is **truncated / incomplete**.
- Readable prefix still contains **all 11** NCBI taxids of the raw panel.

### Counts from readable (truncated) prefix — underestimates

| taxid | Species | Rows | Unique genes | OGs | OGs with ≥2 genes (paralog proxy) |
|-------|---------|------|--------------|-----|-----------------------------------|
| 9606 | Homo sapiens | 203 355 | 39 023 | 87 257 | 84 639 |
| 10090 | Mus musculus | 194 033 | 42 748 | 75 851 | 72 863 |
| 10116 | Rattus norvegicus | 198 932 | 44 093 | 78 059 | 72 979 |
| 9615 | Canis lupus familiaris | 151 106 | 39 804 | 62 532 | 57 079 |
| 9544 | Macaca mulatta | 97 411 | 20 771 | 80 169 | 10 018 |
| 9986 | Oryctolagus cuniculus | 83 076 | 21 360 | 60 304 | 11 570 |
| 9940 | Ovis aries | 79 134 | 20 876 | 58 821 | 9 916 |
| 9913 | Bos taurus | 77 823 | 20 591 | 58 849 | 8 934 |
| 9925 | Capra hircus | 76 184 | 20 093 | 58 583 | 8 406 |
| 9823 | Sus scrofa | 75 325 | 19 903 | 58 063 | 8 015 |
| 9796 | Equus caballus | 69 208 | 20 553 | 51 958 | 8 372 |

### Missing OrthoDB artifacts (for a full graph)

Official dump also ships (not present under `mag/orthoDB/`):

- `odb12v2_genes.tab.gz`, `odb12v2_gene_xrefs.tab.gz` (ID mapping to Ensembl/NCBI)
- `odb12v2_OGs.tab.gz`, `odb12v2_species.tab.gz`, `odb12v2_levels.tab.gz`
- No separate “paralog table” — paralogs are inferred from co-membership in an OG within one taxid

`mag/orthodb/main.py` cannot download these (stub). Re-fetch URL:

`https://data.orthodb.org/v12/download/odb_data_dump/odb12v2_OG2genes.tab.gz`

Connectivity to `data.orthodb.org` from this host was slow/unreliable during the session; resume download was attempted/queued separately if network allows.

## 3. Intersection pointer

Species↔folder↔taxid↔GCF join: [`mag/intersection.md`](intersection.md).

## 4. Recommendation for ortholog/paralog graph

1. **Primary edges:** Ensembl Compara `protein_default` TSVs (already complete for all 11) — filter `homology_type` and optionally `is_high_confidence`.
2. **Within-panel orthologs:** keep edges where both `species` and `homology_species` are in the 11-species set (union of both species files).
3. **Paralogs:** `within_species_paralog` (+ optionally `other_paralog`).
4. **OrthoDB:** useful as orthogonal OG layer after fixing the truncated dump + adding gene xrefs; not blocking if Ensembl graph is the main source.
