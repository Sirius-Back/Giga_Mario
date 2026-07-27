# Giga_Mario Wiki

AIRI project — anti-data leakage methods for DNA foundation models.

## Pages

- [[conversion]] — `raw/` → `data_ready/` preprocessing
- [[split]] — ready panels → train/val/test folds
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
