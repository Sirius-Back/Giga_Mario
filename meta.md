# meta.md

Навигационный каталог документации и основных артефактов проекта (таблицы, результаты, фигуры).

- **Дата:** 2026-08-03
- **Объём:** пять тематических разделов; смоук-аудиты `docs/execution/T-*`, `docs/code-review/T-*`, `docs/audits/T-*` и сторонние metagenomics-skills **не** перечислены — см. [`docs/artifact-registry.md`](docs/artifact-registry.md)
- **Пересечения:** `splits/vgae.md`, `splits/GCN.md`, `splits/vae.md` кратко в «Сплитах», подробнее в «GCN-VAE»; `AGENTS.md` / `method-decision.md` — в «Обучении моделей»

## Оглавление

1. [Обучение моделей](#обучение-моделей)
2. [Сплиты](#сплиты)
3. [Ортологи и паралоги](#ортологи-и-паралоги)
4. [Эмбеддинги](#эмбеддинги)
5. [GCN-VAE](#gcn-vae)

---

# Обучение моделей

Fine-tune Caduceus и LegNet на панелях `ready_caduceus/` / `ready_legnet/` после сплита: **direct** (регрессия TPM) и **adversarial** (классификация фолда). Результаты — в `runs/` (legacy), `runs_unif/{caduceus,legnet}/` (унифицированные прогоны) и сводных таблицах `docs/run_results_*`.

## Документация (.md)

| путь | описание |
|------|----------|
| `metrics.md` | Контракт метрик эпох: TorchMetrics (Pearson, Spearman, MSE/RMSE/MAE/R², genewise/samplewise medians) для логирования train/val/test. |
| `training-audit.md` | Указатель на канонический отчёт `docs/training-audit.md`. |
| `docs/training-audit.md` | Диагностика динамики обучения: запоминание train, ранний stop, нестабильность; Caduceus и LegNet. |
| `docs/adapt.md` | Обзор `@adapt`: окна CDS±10 kb, TPM, выход перед сплитом и обучением Caduceus. |
| `docs/caduceus_format.md` | Формат fine-tune Caduceus (GB-style vs проектный `caduceus_ready/` с continuous TPM). |
| `docs/combined_results.md` | Мастер-пак: provenance панелей, таблицы direct/adversarial, Spearman лучших чекпоинтов, embed-анализы. |
| `docs/run_results_direct.md` | Таблица метрик direct-прогонов (Pearson, ZSV, split type, best epoch). |
| `docs/run_results_adversarial.md` | Таблица метрик adversarial-прогонов (fold-class / repredict). |
| `docs/best_models_compare_report.md` | Сравнение Spearman на лучших чекпоинтах по стратегиям сплита. |
| `docs/unif_metrics_spearman_report.md` | Spearman unified run на best checkpoint; источники train/val и LegNet repredict. |
| `docs/runs_unif_zsv_viz_queue_audit.md` | Аудит ZSV, train_monitor и очереди незавершённых unified run. |
| `docs/caduceus-full-report.md` | Отчёт smoketest `@caduceus-full`: split1 TPM + split2 fold-class. |
| `docs/manuscript_draft.md` | Черновик рукописи со ссылками на training figures/metrics. |
| `docs/manuscript_missing.md` | Список пробелов рукописи (в т.ч. training-артефакты). |
| `wiki/Split & train.md` | Кодовый путь: prepare → split → `src.caduceus`/`legnet` → `train_viz`. |
| `wiki/conversion.md` | `raw/` → `data_ready/` (`@adapt`) для Caduceus. |
| `wiki/legnet_conversion.md` | `raw/`/BED → `legnet_ready/` 230 bp TSV для human_legnet. |
| `AGENTS.md` | Карта проекта: entry points, pipeline raw→train, `runs_unif/`, `metrics.md`. |
| `method-decision.md` | Locked/Tentative методические решения (метрики, сплит, early stop, Hydra, unified runs). |
| `queue.md` | Журнал крупных локальных CPU/RAM/GPU job (PID, peak RAM, статус). |
| `monitoring-report.md` | Сессионный мониторинг pipeline-прогонов. |
| `docs/monitoring-report.md` | Канон/копия monitoring-report в `docs/`. |
| `README.md` | Обзор: skill-pipeline preprocess → split → train → adversarial. |
| `launch.md` | Launch-гайд: `prepare` + `pipeline`, контракты fold/stratification. |
| `execute.md` | Справочник исполнения на хосте (conda, env, бинарники). |
| `skills.md` | Каталог Cursor skills (`train`, `pipeline`, `split`, …). |
| `src/README.md` | Layout `src/`: caduceus, legnet, train_viz, pipeline CLI. |
| `refactoring.md` | План унификации adapt → parse → split → train. |
| `.cursor/rules/model-train.mdc` | Правило `/train`: validation-first, epoch logging, queue/RAM, 20 ep. |
| `.cursor/skills/train/SKILL.md` | Скилл `/train`: генерация `src/run/<run_id>/*_{direct\|adversarial}.py`. |
| `.cursor/skills/pipeline/SKILL.md` | Скилл `/pipeline`: Hydra split → train → optional adversarial → ZSV. |
| `ready_caduceus/parse.md` | Отчёт preprocess Caduceus-панели (ID/MARKED/PARSED/PREDICT). |
| `ready_legnet/parse.md` | Отчёт preprocess LegNet-панели (230 bp, TPM linkage). |

Стратегии сплита, которые кормят train, описаны в разделе [Сплиты](#сплиты) (`splits/*.md`).

## Таблицы и результаты

| путь | что внутри |
|------|------------|
| `docs/run_results_direct.csv` | CSV direct: run_id, модель, split, Pearson train/val/test/ZSV, best epoch. |
| `docs/run_results_adversarial.csv` | CSV adversarial: accuracy / repredict Pearson, статус. |
| `docs/run_results_tables.json` | JSON-сводка direct/adversarial таблиц. |
| `docs/best_models_compare_metrics.csv` | Spearman лучших чекпоинтов по run×stage. |
| `docs/unif_metrics_spearman.csv` | Spearman unified run (direct/adv) на best epoch. |
| `runs/` | Legacy out: `run{N}_{strategy}_{model}/` с `direct/` и `adversarial/`. |
| `runs_unif/caduceus/` | Unified Caduceus: `run{N}_caduceus_{strategy}_…/`. |
| `runs_unif/legnet/` | Unified LegNet: `run{N}_legnet_{strategy}_…/`. |
| `{run_root}/{direct\|adversarial}/` | `logs/` (epoch metrics, `train_metrics.jsonl`, `zero_shot_metrics.json`), `best_model/`, `final_model/`, `checkpoints/`, `run_config.json`, `best_split_metrics.json`, `tensorboard/`, `figures/train_monitor/`. |
| `ready_caduceus/` | Caduceus panel: `ID.csv`, `MARKED/`, `PARSED/`, `PREDICT/`, `parse_data_stats.json`. |
| `ready_legnet/` | LegNet panel: те же контракты + materialize `SPLIT/` / `legnet_input/`. |
| `configs/train/caduceus.yaml` | Hydra CLI-шаблон Caduceus train. |
| `configs/train/legnet.yaml` | Hydra CLI-шаблон LegNet train. |
| `configs/train_job.yaml` | Top-level Hydra train job. |
| `configs/pipeline.yaml` | Полный pipeline orchestrator. |
| `runs_aligned/` | Корень aligned-reruns (`rerun=true`); placeholder / пусто. |
| `logs/` | Stdout train/pipeline job (`run{N}_*_train.log`, resume, train-viz backfill). |

**Шаблоны имён run (группировка):**

| зона | шаблон | примеры |
|------|--------|---------|
| `runs/` | `run{N}_{strategy}_{model}/` | `run8_2mer_caduceus`, `run11_4mer_legnet`, `run16_hashfrag_caduceus`, `run17_pangenome_CDS_legnet` |
| `runs_unif/` | `run{N}_{model}_random` | `run1_caduceus_random`, `run2_legnet_random` |
| | `run{N}_{model}_gc_kmeans_elbow` | `run3_caduceus_gc_…`, `run4_legnet_gc_…` |
| | `run{N}_{model}_kmer_k{2,4,7}` | `run8_caduceus_kmer_k2`, `run13_legnet_kmer_k7` |
| | `run{N}_{model}_hashfrag` | `run5_legnet_hashfrag`, `run16_caduceus_hashfrag` |
| | `run{N}_{model}_pangenome_k{5,7,10,21}_…` | `run20_…_k10_w0_100`, `run31_…_loo5` |
| | `run{N}_{model}_paralogs_only` | `run24_legnet_…`, `run25_caduceus_…` |
| | `run{N}_{model}_loco` | `run35_legnet_loco`, `run36_caduceus_loco` |
| | `run22_legnet_mmseqs_id08`, `run39_caduceus_blastp` | mmseqs / blastp |
| | `run{41–46}_{model}_{vgae\|gcn}_stage1_…` | graph/VGAE → train (см. [GCN-VAE](#gcn-vae)) |

## Картинки / фигуры

| путь | что изображено |
|------|----------------|
| `figures/best_models_compare/` | Сравнение лучших моделей: multimetric bars, Spearman (Fig01–02). |
| `figures/presentation/strategy_zsv_unified*` | Strategy×ZSV unified / Caduceus-only / LegNet-only панели (+ `*_table.tsv`). |
| `figures/presentation/leak_breaks_early_stopping_{en,ru}.*` | Схема «утечка ломает early stopping». |
| `figures/presentation/article_Fig01*` | Презентационные версии Fig01 (Spearman / scatter). |
| `figures/article/Fig01*` | Fig. 1 статьи: Spearman best checkpoint. |
| `figures/article/ED_Fig01*` | Extended Data: multimetric лучших чекпоинтов. |
| `figures/article/ED_Fig02*` | Extended Data: ZSV по моделям. |
| `figures/article/Fig04_leak_breaks_early_stopping.*` | Article-копия early-stop leakage. |
| `figures/train-viz/all_completed_direct/` | Агрегированные кривые direct `@train-viz`. |
| `figures/train-viz/all_completed_adversarial/` | Агрегированные кривые adversarial. |
| `figures/train-viz/tb_compare/{direct,adversarial}/` | Экспорт TensorBoard-сравнений. |
| `{run_root}/{direct\|adversarial}/figures/train_monitor/` | Per-run learning curves, early-stop, gap plots. |

**Entry points:**

```bash
python -m src.hydra_pipeline mode=run run_id=<run_id>
python -m src.caduceus --splits-dir <SPLIT>/direct/caduceus_input --out <out>/direct
python -m src.legnet --data-path <SPLIT>/legnet_input/all.tsv --out <out>/direct
python -m src.train_viz --models runs_unif/caduceus/<run_id> -o figures/train-viz/<run_id>
```

---

# Сплиты

Анти-утечка разбиение панели на train/val/test (и ZSV): caption `splits/*.md` → `split-predict` → `split.csv` → materialize `SPLIT/` → опциональный OG/PG-аудит.

## Документация (.md)

| путь | описание |
|------|----------|
| `splits/random.md` | Случайный сплит (M1 TPM + M2 fold-class); baseline с высокой утечкой. |
| `splits/gc.md` | SBS: GC% + AAA% → DBSCAN/k-means folds (`type=gc`). |
| `splits/GC.md` | Дубликат/синоним caption `gc` (тот же frontmatter id). |
| `splits/kmer.md` | SBS: k-mer composition (DSK/native; `type=kmer`). |
| `splits/hashfrag.md` | hashFrag + BLAST orthogonal homology split (`type=hashfrag`). |
| `splits/pangenome.md` | C++ k-mer repeat-hash graph → CC → folds (`type=pangenome`). |
| `splits/blastp.md` | BLASTP protein-homology SBS (`type=blastp`). |
| `splits/mmseqs.md` | MMseqs2 `easy-cluster` folds; Locked 60:20:20 (`type=mmseqs`). |
| `splits/paralogs_only.md` | Orthogroup-rep train / paralog remainder 50/50 (`type=paralogs_only`). |
| `splits/LOCO.md` | Leave-one-chromosome-out (`type=loco`). |
| `splits/vgae.md` | VGAE graph split; **см. [GCN-VAE](#gcn-vae)**. |
| `splits/vae.md` | MLP-VAE k-mer split (без GCN); **см. [GCN-VAE](#gcn-vae)**. |
| `splits/GCN.md` | GCN/VGAE labeling cascade; **см. [GCN-VAE](#gcn-vae)**. |
| `wiki/split.md` | Entry `@split`: `src.splits.main` → folds из ready. |
| `wiki/sbs.md` | Контракты Split-by-Similarity (FeatureTable, assign, PCA). |
| `wiki/split-generate.md` | `/split-generate`: caption → `src/splits/<id>.py`. |
| `wiki/architecture.md` | Пайплайн adapt → parse → split-predict → split → train. |
| `wiki/Split & train.md` | End-to-end split → train → ZSV → viz. |
| `wiki/Home.md` | Оглавление wiki. |
| `wiki/_Sidebar.md` | Навигация wiki. |
| `configs/pipeline.yaml` | Hydra knobs: `split`, k-mer, hashfrag, pangenome, plot flags. |
| `.cursor/skills/split-generate/SKILL.md` | Генерация стратегии из caption + wiring `split_predict`. |
| `.cursor/skills/hashfrag/SKILL.md` | Workflow hashFrag homology split. |
| `.cursor/skills/hashfrag/reference.md` | CLI/SLURM reference hashFrag. |
| `.cursor/skills/blastp/SKILL.md` | Контракт BLASTP split. |
| `.cursor/skills/mmseqs/SKILL.md` | Контракт MMseqs2 split. |
| `.cursor/skills/adversarial/SKILL.md` | Adversarial re-split + fold-class PREDICT. |
| `.cursor/skills/pipeline/SKILL.md` | Orchestrator split → train. |
| `.cursor/skills/preprocess/SKILL.md` | Upstream MARKED/PARSED/PREDICT для split. |
| `.cursor/skills/split-check-othoparagroup/SKILL.md` | Аудит split vs OG/PG (подробнее в [Ортологи](#ортологи-и-паралоги)). |

## Таблицы и результаты

| путь | что внутри |
|------|------------|
| `{outdir}/split.csv` | Контракт `ID\|train_test\|fold`. |
| `{outdir}/sbs_assignment.csv` | SBS: region → cluster → train_test/fold. |
| `{outdir}/feature_table.csv` / `.npz` | Feature matrix (GC/AAA, k-mers, …). |
| `{outdir}/pangenome_assignment.csv` | Region-level pangenome assignment. |
| `{outdir}/graph/contingency_graph.npz` | Pangenome graph (cluster_ids, edges). |
| `{outdir}/graph/{ids.txt,nodes.tsv,edges.tsv,contingency_graph_meta.json}` | Graph sidecars. |
| `{outdir}/hashfrag_work/` | `hashFrag.homologous_groups.tsv`, split TSV, fasta. |
| `{outdir}/*_split_meta.json` | Мета стратегии (`gc_`, `kmer_`, `hashfrag_`, `pangenome_`, `blastp_`, `mmseqs_`). |
| `{outdir}/stage_{features,assign}_done.json`, `split_done.json`, `split_cpu_done.json` | Checkpoints стадий. |
| `{outdir}/SPLIT/` | Материализованные FASTA + PREDICT (train/val/test/ZSV). |
| `{outdir}/legnet_input/all.tsv` | LegNet TSV из SPLIT. |
| `splits/random/M1\|M2/` | Legacy random fold trees + `splits_log.csv`. |
| `runs/run1/` … `runs/run18_pangenome_CDS_caduceus/` | Legacy outdirs по стратегиям (см. таблицу ниже). |
| `runs_unif/{caduceus,legnet}/run*_…/` | Unified aligned split+train roots. |
| `runs_unif/splits/INDEX.tsv` | Индекс OG/PG-аудитов всех сплитов. |
| `runs_unif/splits/{model}_{run}/{othologs,paralogs}.csv` | Per-group `sd_random` audit. |
| `results/splits_stratification/summary.tsv` | Агрегат стратификации по методам. |
| `results/splits_stratification/manifest.json` | Манифест сводки стратификации. |

**Legacy runs (`runs/`):**

| путь | стратегия |
|------|-----------|
| `runs/run1/`, `runs/run2/` | random (Caduceus / LegNet) |
| `runs/run3/`, `runs/run4/` | gc |
| `runs/run5/`, `runs/run16_hashfrag_caduceus/` | hashfrag |
| `runs/run7_2mer_*` … `run14_7mer_*` | kmer k=2/4/7 |
| `runs/run15_blastp_legnet/` | blastp |
| `runs/run17_pangenome_CDS_legnet/`, `runs/run18_…_caduceus/` | pangenome (+ `graph/`, figures) |

## Картинки / фигуры

| путь | что изображено |
|------|----------------|
| `runs/run17_pangenome_CDS_legnet/figures/Figure_pangenome_contingency_fold_train_test.*` | Contingency graph, цвет по fold/train-test. |
| `runs/run17_pangenome_CDS_legnet/figures/contingency_graph.*` | Базовый contingency graph. |
| `{outdir}/figures/pca_by_{cluster,train_test,genome}.*` | SBS PCA diagnostics. |
| `{outdir}/figures/Figure_pangenome_fold_size_log10.*` | Размеры pangenome folds (log10). |
| `figures/presentation/split_stratification_{en,ru}.*` | 4-panel: sd_random OG/PG, L_hom, scatter, vs D_hom_emb. |
| `figures/article/Fig02_split_stratification.*` | Fig. 2 статьи (стратификация). |
| `runs_unif/splits/figures/Figure_04_othologs_sd_ks_heatmap.*` | Heatmap OG `sd_random` across splits. |
| `runs_unif/splits/figures/Figure_04_paralogs_sd_ks_heatmap.*` | Heatmap PG `sd_random` across splits. |

**Entry points:**

```bash
python -m src.splits.main --strategy random --raw raw --ready ready --seed 42
python -m src.pipeline.split_predict --outdir <out> --type gc --marked ready_legnet/MARKED --seed 42
python -m src.pipeline.split --split-csv <out>/split.csv --parsed <out>/PARSED --predict <out>/PREDICT
python -m src.hydra_pipeline mode=run run_id=run0 split=random train=legnet
```

---

# Ортологи и паралоги

Ensembl Compara ortholog/paralog граф (`mag/`), orthoparagroups, homology-aware сплиты и аудит утечки OG/PG в fold assignment.

## Документация (.md)

| путь | описание |
|------|----------|
| `splits/paralogs_only.md` | Orthogroup CC → 1 rep в train; remainder 50/50 test/val. |
| `splits/blastp.md` | Protein BLASTP homology SBS (кратко — homology-aware). |
| `splits/mmseqs.md` | MMseqs2 cluster-first homology folds. |
| `splits/hashfrag.md` | hashFrag BLAST → orthogonal splits (снижение homology leakage). |
| `splits/vgae.md` | Homology firewall: OG/PG только в `L_hom`, не в encoder; **см. [GCN-VAE](#gcn-vae)**. |
| `splits/GCN.md` | Тот же firewall для GCN cascade. |
| `splits/vae.md` | MLP-VAE: homology-first `L_hom`, без OG/PG в encoder. |
| `mag/README.md` | Обзор `mag/`: Compara, граф, orthoparagroups, C++ tools. |
| `mag/intersection.md` | raw × Ensembl/OrthoDB intersection → mammals-11 panel. |
| `mag/homology_availability_report.md` | Доступность Compara/OrthoDB для панели. |
| `mag/homology_graph/README.md` | Сборка undirected ortholog/paralog графа, stats, figures. |
| `mag/orthoparagroups/README.md` | Orthoparagroup FASTA extractor / clusters. |
| `.cursor/skills/split-check-othoparagroup/SKILL.md` | Аудит `split.csv` vs hash-table → `othologs.csv` / `paralogs.csv`. |
| `.cursor/skills/split-check-othoparagroup/reference.md` | Схема hash-table, определения групп, формула `sd_random`. |
| `wiki/split-generate.md` | Генерация стратегий; упоминание pending `paralogs_only`. |

## Таблицы и результаты

| путь | что внутри |
|------|------------|
| `mag/homology_graph/edges.tsv.gz` | Undirected рёбра `ortholog` / `paralog`. |
| `mag/homology_graph/maps/gene_ortho_para_hash.tsv` | Sorted lookup MARKED ↔ orthogroup/paragroup hashes. |
| `mag/homology_graph/maps/nodes_extract.tsv` | Ensembl → MARKED node map. |
| `mag/homology_graph/maps/nodes_enriched.tsv` | Обогащённые узлы. |
| `mag/homology_graph/maps/edges_extract.tsv` | Extracted edge subset. |
| `mag/homology_graph/stats/component_sizes.tsv.gz` | Размеры связных компонент. |
| `mag/homology_graph/stats/ortholog_group_sizes.tsv.gz` | Размеры orthogroups. |
| `mag/homology_graph/stats/paralog_degree.tsv.gz` | Paralog degree per gene. |
| `mag/homology_graph/stats/linked_paralog_clusters.tsv.gz` | Linked paralog clusters. |
| `mag/homology_graph/stats/relation_mix.tsv.gz` | Mix ortholog/paralog edges. |
| `mag/homology_graph/summary.json` | Сводка сборки графа. |
| `mag/homology_graph/manifest.json` | Build manifest. |
| `mag/orthoparagroups/clusters.tsv` | Per-cluster orthoparagroup stats. |
| `mag/orthoparagroups/cluster_*.fna` | Multi-sequence FASTA по кластерам. |
| `mag/orthoparagroups_aligned/cluster_*.aln.fa` | MAFFT alignments. |
| `mag/orthoparagroups_aligned/cluster_*.pos.tsv.gz` | Per-position consensus metrics. |
| `mag/orthoparagroups_aligned/metrics/` | Consensus-rate metrics per cluster. |
| `runs_unif/splits/INDEX.tsv` | Индекс audited splits (n_OG, n_PG, role fractions). |
| `runs_unif/splits/{model}_{run}/othologs.csv` | `orthogroup\|n_train\|n_test\|n_val\|sd_random`. |
| `runs_unif/splits/{model}_{run}/paralogs.csv` | `paragroup\|n_train\|n_test\|n_val\|sd_random`. |
| `runs_unif/splits/{model}_{run}/summary.json` | Global audit summary. |
| `results/embed_legnet/homology_dissim/ranking.tsv` | Ранжирование сплитов по `D_hom_emb`. |
| `results/embed_legnet/homology_dissim/per_store_*.tsv` | Per-run / per-layer scores. |
| `results/embed_legnet/homology_dissim/manifest.json` | Манифест homology dissim. |

**Референсные homology-aware runs:**

| run | путь |
|-----|------|
| paralogs_only LegNet | `runs_unif/legnet/run24_legnet_paralogs_only/` |
| paralogs_only Caduceus | `runs_unif/caduceus/run25_caduceus_paralogs_only/` |
| mmseqs LegNet | `runs_unif/legnet/run22_legnet_mmseqs_id08/` |
| blastp Caduceus | `runs_unif/caduceus/run39_caduceus_blastp/` |

## Картинки / фигуры

| путь | что изображено |
|------|----------------|
| `mag/homology_graph/figures/Figure_01_component_size_hist.*` | Гистограмма размеров CC. |
| `mag/homology_graph/figures/Figure_02_paralog_degree_hist.*` | Paralog degree. |
| `mag/homology_graph/figures/Figure_03_orthogroup_size_hist.*` / `Figure_03_orthogroup_span.*` | Orthogroup size / span. |
| `mag/homology_graph/figures/Figure_04_linked_cluster_sizes.*` | Linked cluster sizes. |
| `mag/homology_graph/figures/Figure_05_linked_cluster_size_ratio.*` | Ratio linked clusters. |
| `mag/homology_graph/graph_network.png` / `graph_network_full.{png,pdf}` | Сетевые визуализации. |
| `mag/orthoparagroups/figures/` | Распределения по `clusters.tsv`. |
| `mag/orthoparagroups_aligned/figures/` | Pair-corr, similar-length, rate violin. |
| `mag/orthoparagroups_aligned/figures_meta/Figure_10_meta_profile_orthologs.*` | Meta-profile ортологов. |
| `mag/orthoparagroups_aligned/figures_meta/Figure_10_meta_profile_paralogs.*` | Meta-profile паралогов. |
| `figures/presentation/paralog_ortholog_dissim_{en,ru}.*` | Барчарт `D_hom_emb` по сплитам. |
| `figures/presentation/split_stratification_{en,ru}.*` | OG/PG stratification panels. |
| `figures/article/Fig03_paralog_ortholog_dissim.pdf` | Fig. 3 статьи. |
| `runs_unif/splits/figures/Figure_04_*_sd_ks_heatmap.*` | Cross-split OG/PG heatmaps. |

**Entry points:**

```bash
python -m src.run.homology_graph.build_mammals11 --ensembl-data mag/ensembl/data --outdir mag/homology_graph
python3 mag/src/orthoparagroups/build_hash_table.py --id-csv ready_legnet/ID.csv \
  --nodes mag/homology_graph/maps/nodes_extract.tsv --edges mag/homology_graph/edges.tsv.gz \
  --out mag/homology_graph/maps/gene_ortho_para_hash.tsv
./mag/src/split_check_othoparagroup/split_check_othoparagroup \
  --split runs_unif/legnet/run24_legnet_paralogs_only/split.csv \
  --hash-table mag/homology_graph/maps/gene_ortho_para_hash.tsv
python -m src.embed.run_homology --embed-root results/embed_legnet --out results/embed_legnet/homology_dissim
```

---

# Эмбеддинги

Извлечение layer-эмбеддингов обученных LegNet/Caduceus, анализ утечки (L(τ)), геометрии ортолог/паралог (`D_hom_emb`) и pairwise RSA/CKA между сплитами.

## Документация (.md)

| путь | описание |
|------|----------|
| — | **Пробел:** отдельной wiki / `splits/*.md` для embed-пайплайна нет. |
| `docs/combined_results.md` | Секция Embed analyses (homology_dissim, pairwise, статусы). |
| `figures/article/MANIFEST.md` | Манифест article-фигур (включая Fig. 5–6), если собран assembler'ом. |
| `src/embed/` | Код и CLI; **отдельных `.md` в каталоге нет** — контракт слоёв в `src/embed/__init__.py`. |

## Таблицы и результаты

| путь | что внутри |
|------|------------|
| `results/embed_legnet/` | Phase-1 LegNet embed root. |
| `results/embed_caduceus/` | Phase-2 Caduceus embed root. |
| `results/embed_{legnet\|caduceus}/run_complete.json` | Статус pipeline, stages, `finished_at`. |
| `results/embed_{legnet\|caduceus}/validation_report.json` | Pre-extract validation (`n_ready`, failures). |
| `results/embed_legnet/<run_name>/` | Per-run store: `ids.npy`, `roles.npy`, `layer_*.npy`, `manifest.json`. |
| `results/embed_legnet/<run_name>/foldN/` | LOO fold stores (тот же набор). |
| `results/embed_caduceus/<run_name>/` | Аналогичные Caduceus stores. |
| `results/embed_legnet/homology_dissim/ranking.tsv` | `D_hom_emb` ranking (primary figure input). |
| `results/embed_legnet/homology_dissim/per_store_*.tsv` | Per-layer / per-run scores. |
| `results/embed_legnet/homology_dissim/manifest.json` | Манифест. |
| `results/embed_legnet/pairwise/pairwise_compare.tsv` | Cross-split geometry compare. |
| `results/embed_legnet/pairwise/pairwise_compare.json` | JSON-сводка pairwise. |
| `results/embed_legnet/pairwise/matrix_{layer}_{score}.npy` | RDM / CKA matrices. |
| `results/embed_legnet/leakage_ranking.tsv` | Leakage ranking across runs (если прогон выполнен). |
| `results/embed_legnet/leakage/` | `summary_*.json`, `L_tau_*.{pdf,npz}` per store. |
| `results/embed_caduceus/pairwise/` | Pairwise Caduceus (shell `src/embed/run_caduceus_pairwise.sh`). |
| `results/embed_caduceus_smoke/` | Smoke extract. |

## Картинки / фигуры

| путь | что изображено |
|------|----------------|
| `results/embed_legnet/pairwise/heatmap_{layer}_{score}.{pdf,svg}` | Triangle heatmaps RSA/CKA/… |
| `figures/article/Fig05_embed_pairwise_rsa_pooled.*` | Fig. 5 — pairwise RSA (pooled). |
| `figures/article/Fig06_embed_pairwise_cka_pooled.*` | Fig. 6 — pairwise CKA (pooled). |
| `figures/presentation/article_Fig05_embed_pairwise_rsa_pooled.*` | Presentation-копия Fig. 5. |
| `figures/presentation/article_Fig06_embed_pairwise_cka_pooled.*` | Presentation-копия Fig. 6. |
| `figures/presentation/paralog_ortholog_dissim_{en,ru}.*` | **cross-ref** → [Ортологи](#ортологи-и-паралоги): `D_hom_emb` bars. |
| `figures/presentation/split_stratification_{en,ru}.*` | Overlay stratification vs `D_hom_emb`. |

**Entry points:**

```bash
python -m src.embed.run_legnet      # validate → extract → leakage → results/embed_legnet/
python -m src.embed.run_caduceus   # extract + validation → results/embed_caduceus/
python -m src.embed.run_homology   # D_hom_emb → homology_dissim/
python -m src.embed.run_pairwise   # RSA/CKA + heatmaps → pairwise/
```

---

# GCN-VAE

В репозитории: **VGAE** = classic GCN-VAE (Kipf & Welling) на pangenome/hash-графе (`splits/vgae.md`, alias `gcn_vae`); **GCN cascade** reuse/infer/train (`splits/GCN.md`); **MLP-VAE** baseline без графа (`splits/vae.md`). Homology labels **не** входят в encoder — только post-assignment `L_hom`.

## Документация (.md)

| путь | описание |
|------|----------|
| `splits/vgae.md` | Classic VGAE: GC/k-mer + weighted adj → latent; Stage1 region / Stage2 hash; firewall `L_hom`. |
| `splits/GCN.md` | Cascade: reuse labeling → infer checkpoint → train; `type=gcn`, `gcn_model=…`. |
| `splits/vae.md` | MLP-VAE на k-mer features (без GCN); homology-first loss. |
| `VGAE/vae_vgae_architecture_comparison.md` | Сравнение архитектур VGAE vs MLP-VAE (если прогон `compare_vae_vgae_architectures` выполнен). |

См. также связанные пункты в [Сплитах](#сплиты) и [Ортологах](#ортологи-и-паралоги) (`homology_loss`, `sd_random`).

## Таблицы и результаты

| путь | что внутри |
|------|------------|
| `VGAE/` | Корень всех VGAE (GCN-VAE) прогонов. |
| `VAE/` | Корень MLP-VAE baselines. |
| `VGAE/stage1_region_k5/` | Stage-1 region graph, k=5 (базовый). |
| `VGAE/stage1_region_k5_lossfix/` | Stage-1 + lossfix / homology_first. |
| `VGAE/stage1_region_k5_{gat,sage,gcl,gcl_gat,appnp,gcnii,multik457,structfeat}_lossfix/` | Архитектурные абляции encoder. |
| `VGAE/stage1_region_k5_hom_{robust,logbal}/` | Абляции агрегации homology loss. |
| `VGAE/stage2_hash_k5/` / `stage2_hash_k5_lossfix/` | Stage-2 hash-node graph → region roles. |
| `VGAE/stage2_hash_k7_lossfix/` / `stage1_region_k7_lossfix/` | k=7 варианты. |
| `VGAE/<run>/split.csv` | Роли `ID\|train_test\|fold`. |
| `VGAE/<run>/train_meta.json` | grain, k, stage, loss_mode, architecture. |
| `VGAE/<run>/pack/` | Graph tensors, features, `node_homology.tsv`, `feature_meta.json`. |
| `VGAE/<run>/checkpoints/` | `best.pt` и epoch checkpoints. |
| `VGAE/<run>/logs/` | `train_metrics.jsonl`, epoch metrics. |
| `VGAE/checks/` | Offline OG/PG checker outputs. |
| `VGAE/vae_vgae_architecture_comparison.{json,md}` | Сводное сравнение архитектур. |
| `VGAE/arch_*_summary.json` | Summary GAT/SAGE, GCL, APPNP/GCNII/… |
| `VGAE/loss_comparison_k5_k7.json` | Сравнение loss modes k5 vs k7. |
| `VGAE/legacy_eval_existing_models.json` | Eval legacy checkpoints. |
| `VAE/mlp_vae_kmer_k4_lossfix/` | MLP-VAE k=4 baseline. |
| `VAE/mlp_vae_kmer_k7_full16384_lossfix/` | MLP-VAE k=7 full vocab. |
| `VAE/checks/` | Checker outputs для MLP-VAE. |
| `runs_unif/legnet/run41_legnet_vgae_stage1_k5/` | Unified LegNet train на VGAE stage1 k5. |
| `runs_unif/caduceus/run42_caduceus_vgae_stage1_k5/` | Unified Caduceus на том же split. |
| `runs_unif/legnet/run43_legnet_vgae_stage1_k5_lossfix/` | lossfix variant. |
| `runs_unif/caduceus/run44_caduceus_vgae_stage1_k5_lossfix/` | lossfix Caduceus. |
| `runs_unif/legnet/run45_legnet_gcn_stage1_k5_gcnii_lossfix/` | GCNII cascade → LegNet. |
| `runs_unif/caduceus/run46_caduceus_gcn_stage1_k5_gcnii_lossfix/` | GCNII cascade → Caduceus. |
| `logs/vgae_arch_ab/` | Логи архитектурных абляций. |

## Картинки / фигуры

| путь | что изображено |
|------|----------------|
| `VGAE/<run>/figures/` / `tensorboard/` | Per-run train-viz (если генерировались); **единого article-набора VGAE нет**. |
| `figures/presentation/split_stratification_{en,ru}.*` | Overlay методов `vgae`/`gcn` в стратификации (см. [Сплиты](#сплиты)). |
| `figures/article/Fig02_split_stratification.*` | Article-копия stratification (включает graph-split методы). |

**Entry points:**

```bash
python -m src.splits.vgae --stage 1 --out VGAE/stage1_region_k5_lossfix
python -m src.splits.vae --out VAE/mlp_vae_kmer_k4_lossfix
python -m src.pipeline.split_predict --type vgae --outdir <out> ...
python -m src.pipeline.split_predict --type gcn --gcn-model stage1_region_k5_gcnii_lossfix --outdir <out> ...
```
