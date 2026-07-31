# Intersection: raw (11 mammals) × mag (Ensembl / OrthoDB)

Generation date: 2026-07-31  
Scope: semantic species match between `./raw` genomes and `mag/ensembl` / `mag/orthoDB` (not name equality of folder vs GCF accession).

## Raw panel (11 species)

Source: `raw/fna/*.fna` headers + `raw/random_borzoi_expr_file_mappings.csv`.

| # | Scientific name | Common | NCBI taxid | Genome accession (raw) | Assembly name (raw) | TPM id |
|---|-----------------|--------|------------|------------------------|---------------------|--------|
| 1 | *Homo sapiens* | human | 9606 | GCF_000001405.40 | GRCh38.p14 | ENCSR161RSX |
| 2 | *Mus musculus* | mouse | 10090 | GCF_000001635.27 | GRCm39 | SRX28488332 |
| 3 | *Rattus norvegicus* | rat | 10116 | GCF_036323735.1 | GRCr8 | SRX8083153 |
| 4 | *Sus scrofa* | pig | 9823 | GCF_000003025.6 | Sscrofa11.1 | SRX10021131 |
| 5 | *Bos taurus* | cattle | 9913 | GCF_002263795.3 | ARS-UCD2.0 | SRX5557499 |
| 6 | *Canis lupus familiaris* | dog | 9615 | GCF_011100685.1 | UU_Cfam_GSD_1.0 | SRX13456240 |
| 7 | *Macaca mulatta* | rhesus macaque | 9544 | GCF_049350105.2 | T2T-MMU8v2.0 | SRX2872548 |
| 8 | *Ovis aries* | sheep | 9940 | GCF_016772045.2 | ARS-UI_Ramb_v3.0 | SRX20581436 |
| 9 | *Oryctolagus cuniculus* | rabbit | 9986 | GCF_964237555.1 | mOryCun1.1 | SRX22688362 |
| 10 | *Equus caballus* | horse | 9796 | GCF_002863925.1 | EquCab3.0 | SRX19584896 |
| 11 | *Capra hircus* | goat | 9925 | GCF_001704415.2 | ARS1.2 | SRX6696967 |

## Semantic join keys

| Layer | Identifier style | Join rule |
|-------|------------------|-----------|
| `raw/` | GCF accession + assembly nickname | Species from FASTA definition line / assembly |
| `mag/ensembl/data/<species>/` | Ensembl production name (`homo_sapiens`, …) | Map scientific name → snake_case production name |
| `mag/orthoDB/odb12v2_OG2genes.tab.gz` | OrthoDB gene id `taxid_…` | Map scientific name → NCBI taxid prefix |
| Ensembl Compara TSV | `species` / `homology_species` columns | Same production names as Ensembl folders |

**Not** matched by string equality of folder name to GCF id. Assembly versions may differ across layers (e.g. dog: raw `UU_Cfam_GSD_1.0` vs Ensembl pep `ROS_Cfam_1.0`; cattle: raw `ARS-UCD2.0` vs mixed Ensembl 1.2/2.0 files).

## Intersection table (folders / sources)

| Species | Ensembl folder `mag/ensembl/data/…` | Present locally (pre-download) | OrthoDB taxid in `odb12v2_OG2genes` | In Ensembl `config.yaml` species list | Notes |
|---------|--------------------------------------|--------------------------------|-------------------------------------|---------------------------------------|-------|
| Homo sapiens | `homo_sapiens` | yes | 9606 | yes | |
| Mus musculus | `mus_musculus` | yes | 10090 | yes | |
| Rattus norvegicus | `rattus_norvegicus` | yes (empty payload) | 10116 | yes | dirs exist; pep/gtf empty |
| Sus scrofa | `sus_scrofa` | yes (empty payload) | 9823 | yes | dirs exist; pep/gtf empty |
| Bos taurus | `bos_taurus` | yes | 9913 | yes | |
| Canis lupus familiaris | `canis_lupus_familiaris` | yes | 9615 | yes | |
| Macaca mulatta | `macaca_mulatta` | yes | 9544 | yes | |
| Ovis aries | `ovis_aries` | yes (homology downloaded 2026-07-31) | 9940 | **no** (CLI override used) | Folder created by homology download; still absent from default `config.yaml` species list |
| Oryctolagus cuniculus | `oryctolagus_cuniculus` | yes (empty payload) | 9986 | yes | dirs exist; pep/gtf empty |
| Equus caballus | `equus_caballus` | yes | 9796 | yes | |
| Capra hircus | `capra_hircus` | yes | 9925 | yes | |

### Counts

- raw species: **11**
- ensembl local folders matching raw: **10 / 11** (`ovis_aries` missing)
- OrthoDB taxids present in `odb12v2_OG2genes.tab.gz`: **11 / 11** (sampled + full-file presence confirmed for all taxid prefixes)
- `mag/orthodb/` (lowercase): downloader **stub only** — no per-species data dirs
- `mag/orthoDB/`: flat dump `odb12v2_OG2genes.tab.gz` (~2.3 GiB), not per-species folders

## Homology-relevant paths (expected)

| Source | Orthologs | Paralogs | Local path pattern |
|--------|-----------|----------|--------------------|
| Ensembl Compara | `homology_type` = `ortholog_*` | `homology_type` = `*_paralog` / within-species | `mag/ensembl/data/<species>/compara_homology/Compara.116.protein_default.homologies.tsv.gz` |
| OrthoDB OG2genes | genes sharing OG across taxids | multi-copy genes of same taxid in one OG | `mag/orthoDB/odb12v2_OG2genes.tab.gz` |

See also: `mag/homology_availability_report.md`.


## Post-download status (2026-07-31)

Ensembl Compara `protein_default` homology TSV downloaded for all **11** raw species into
`mag/ensembl/data/<species>/compara_homology/Compara.116.protein_default.homologies.tsv.gz`.

Each file contains both ortholog (`ortholog_one2one`, `ortholog_one2many`, `ortholog_many2many`)
and paralog (`within_species_paralog`, `other_paralog`) rows via `homology_type`.

Details: `mag/homology_availability_report.md`.
