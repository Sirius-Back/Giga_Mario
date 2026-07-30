# school14 — утилиты для работы с геномными базами данных

Проект содержит набор Python-утилит для скачивания и обработки
геномных данных из открытых баз (Ensembl, OrthoDB и др.).

## Приложения

### 1. [`ensembl/`](ensembl/) — Ensembl FTP Downloader

Скачивание данных с FTP Ensembl (https://ftp.ensembl.org/pub/):
- FASTA-файлы (DNA, cDNA, CDS, protein, ncRNA)
- GTF/GFF3-аннотации
- Compara (гомология)
- XRef (кросс-ссылки)

Используется для получения сырья для построения колонки
паралогов и ортологов.

Подробнее: [`ensembl/README.md`](ensembl/README.md),
[`ensembl/ARCHITECTURE.md`](ensembl/ARCHITECTURE.md)

### 2. [`orthodb/`](orthodb/) — OrthoDB Downloader (заглушка)

Заглушка для будущей утилиты скачивания данных OrthoDB
(https://www.orthodb.org/) — базы ортологов.

Будет реализована в дальнейшем.

## Структура

```
school14/
├── ensembl/      # Приложение 1 (реализовано)
├── orthodb/      # Приложение 2 (заглушка)
└── README.md     # Этот файл
```

## Развёртывание

Все приложения разворачиваются на сервере 85.208.85.123
в директории `/home/User14/mag/`:

```
/home/User14/mag/
├── ensembl/
└── orthodb/
```
