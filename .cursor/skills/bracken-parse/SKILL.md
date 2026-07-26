---
name: bracken-parse
description: >-
  Fast Bracken/Kraken-report parsing to wide count matrices with host cleanup.
  Use when parsing *.nt.G.bracken or *.bracken.*.report files, or when the user
  mentions bracken-parse.
disable-model-invocation: true
---

# Bracken Parse

## Purpose

Parse Bracken genus tables / Kraken-style reports into wide count matrices; drop host/Chordata-like taxa.

## Executable

```bash
Rscript .cursor/skills/bracken-parse/scripts/bracken_parse.R --self-test
Rscript .cursor/skills/bracken-parse/scripts/bracken_parse.R \
  --indir test/wgs --outdir test/metagenomic-import/bracken-parse
```

Implementation: `.cursor/skills/_shared/import/bracken_parse.R`  
Thin hook wrapper: `.cursor/hooks/bracken-parse.sh`
