# Giga_Mario Wiki

AIRI project — anti-data leakage methods for DNA foundation models.

## Pages

- [[architecture]] — target universal tool data-flow (adapt → parse_data → parse_target → split → train)
- [[conversion]] — `raw/` → `data_ready/` preprocessing
- [[split]] — legacy ready → M1/M2 folds (see `/split` skill for universal split-predict+split)
- [[split-generate]] — generate `src/splits/<id>.py` from `splits/*.md` (fold/strat, id_rule)
- [[Split & train]] — `src/` code path for split → train → zero-shot eval → viz

## Syncing this folder to GitHub Wiki

GitHub Wiki is a **separate** git repo (`Giga_Mario.wiki.git`). Pages must sit at the wiki root (`Home.md`, `_Sidebar.md`, page `.md` files).

From the project root (Wiki enabled on GitHub):

```bash
git clone https://github.com/Sirius-Back/Giga_Mario.wiki.git /tmp/Giga_Mario.wiki
rsync -a --delete --exclude .git wiki/ /tmp/Giga_Mario.wiki/
cd /tmp/Giga_Mario.wiki
git add -A
git commit -m "Sync wiki from main repo wiki/"
git push
```

Edit under `wiki/` in the main repo; push to `.wiki.git` when updating the GitHub Wiki tab.
