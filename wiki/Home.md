# GigaMario Wiki

AIRI project — anti-data-leakage methods for DNA foundation models.

## Pages

| Page | Contents |
|------|----------|
| [[architecture]] | Universal pipeline contracts: adapt → parse → split-predict → split → train |
| [[sbs]] | Split-by-similarity: feature tables, clustering, PCA diagnostics |
| [[conversion]] | Legacy Caduceus: `raw/` → `data_ready/` / `ready_small/` |
| [[legnet_conversion]] | Promoters → `legnet_ready/` (230 bp human_legnet) |
| [[split]] | Legacy ready → M1/M2 folds (`python -m src.splits.main`) |
| [[split-generate]] | Generate `src/splits/<id>.py` from `splits/*.md` |
| [[Split & train]] | Code path: split → train → zero-shot eval → viz |

## Mermaid palette

Diagrams use a shared nature theme. **Processes are edge labels** (` -->|adapt| `, ` -->|split-predict| `); nodes are data objects. Empty join cells (`E1`–`E4`) are dashed parchment.

| Class | Role |
|-------|------|
| **earth** | Raw inputs (GTF, FNA, TARGET, tables) |
| **ocean** | Intermediate artifacts (`MARKED`, `PARSED`, `SPLIT`, …) |
| **liposome** | Training outputs, figures, models |
| **moss** | Generated code / helpers |
| **detail** | Path / schema annotations (architecture only) |
| **join** | Empty merge points for multi-input stages |

README omits exact folder layouts; see [[architecture]] for path contracts.
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
