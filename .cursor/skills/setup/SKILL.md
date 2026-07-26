---
name: setup
description: >-
  Create or audit Article.Rmd canonical setup (theme_main, Brewer palettes,
  TARGET/BATCH, libraries). Use when initializing an article, fixing plot themes
  to theme_main, or when the user mentions the setup skill/hook.
disable-model-invocation: true
---

# Setup

## Purpose

Create or audit `Article.Rmd` so plots use `+ theme_main()` and setup-defined `scale_*_main()` / `pal_*()` helpers only.

## Workflow

```
Setup:
- [ ] Step 1: Locate Article.Rmd (create if missing)
- [ ] Step 2: Ensure canonical setup chunk (theme_main, palettes, TARGET/BATCH)
- [ ] Step 3: Audit/rewrite ad-hoc themes and scales outside setup
- [ ] Step 4: Write test/setup/setup-report.json
```

## Executable

```bash
python3 .cursor/skills/setup/scripts/setup.py --self-test
python3 .cursor/skills/setup/scripts/setup.py \
  --article Article.Rmd --target group --batch batch --title Article \
  --report ./test/setup/setup-report.json
```

Thin hook wrapper: `.cursor/hooks/setup.sh`
