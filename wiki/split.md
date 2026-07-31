# Split: raw + ready → folds

**Code:** `src/splits/` · **Entry:** `python -m src.splits.main --strategy <id>`

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#E8F0E6','primaryTextColor':'#2C3E2D','primaryBorderColor':'#6B8F71','lineColor':'#8B7355','secondaryColor':'#E3EEF3','tertiaryColor':'#F4EDE4','clusterBkg':'#FBF8F4','clusterBorder':'#C4B5A0','edgeLabelBackground':'#FBF8F4','fontFamily':'ui-sans-serif, system-ui, sans-serif'}}}%%
flowchart LR
    RAW[(raw/)] --- E1[" "]
    READY[(ready / data_ready)] --- E1
    E1 -->|splits.main| M1["M1 train/val/test\nTPM"]
    E1 -->|splits.main| M2["M2 train/val/test\nfold class"]
    E1 -->|splits.main| LOG[splits_log.csv]

    classDef earth fill:#F4EDE4,stroke:#A67C52,stroke-width:1.5px,color:#3E2723
    classDef ocean fill:#E3EEF3,stroke:#5B8FA8,stroke-width:1.5px,color:#1A3A4A
    classDef join fill:transparent,stroke:#C4B5A0,stroke-width:1px,stroke-dasharray:3 3,color:#8B7355

    class RAW,READY earth
    class M1,M2,LOG ocean
    class E1 join
```

## Inputs

- `raw/` — genomes/annotations/TPM (for strategies that need metadata)
- `ready/` / `data_ready/` / `ready_v2/` — prepared windows (**not** re-converted)

## Random outputs

Default: `splits/random/`. Panel-specific example (2026-07-27): `ready_v2` → `ready_splits/random/` (n=398 292; seed 42; no ZS).

| Tree | Prediction |
|------|------------|
| `M1/{train,val,test}/` | TPM |
| `M2/{train,val,test}/` | M1 fold class (stratified) |
| `splits_log.csv` | `data_input\|M1\|M2` |

```bash
python -m src.splits.main --strategy random --raw raw --ready ready_v2 \
  --seed 42 --out ready_splits/random
```

Skill workflow: **write** `src/splits/<id>.py` → **exec** main → **run** complete.
