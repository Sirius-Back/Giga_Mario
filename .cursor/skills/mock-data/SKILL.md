---
name: mock-data
description: >-
  Generate reproducible 16S, WGS/Bracken, and misc mock fixtures under ./test/.
  Use when regenerating mock inputs, sessionStart fixtures, or when the user
  mentions mock-data / mock-16s / mock-wgs / mock-misc.
disable-model-invocation: true
---

# Mock Data

## Purpose

Generate reproducible mock fixtures for import and taxonomy-tree skills under `./test/` (gitignored).

## Targets

| Target | Output |
|--------|--------|
| `16s` | `test/16s/` sequences, taxonomy, tree, metadata, optional qza |
| `wgs` | `test/wgs/` Bracken tables/reports + metadata |
| `misc` | `test/misc/` taxons.json + taxids.tsv |
| `all` | all of the above |

## Executable

```bash
python3 .cursor/skills/mock-data/scripts/mock_data.py --out ./test --target all --self-test
python3 .cursor/skills/mock-data/scripts/mock_data.py --out ./test --target 16s --self-test
```

Thin hook wrappers: `.cursor/hooks/mock-all-data.sh`, `mock-16s-data.sh`, `mock-wgs-data.sh`, `mock-misc-data.sh`  
`hooks.json` `sessionStart` → `mock-all-data.sh`
