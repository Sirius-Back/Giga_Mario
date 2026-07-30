# Ensembl FTP Downloader

Python-утилита для автоматизированного скачивания геномных данных с FTP Ensembl
([https://ftp.ensembl.org/pub/](https://ftp.ensembl.org/pub/)).

Поддерживает скачивание FASTA-файлов (DNA, cDNA, CDS, protein, ncRNA), GTF/GFF3-аннотаций,
VCF-файлов вариантов, данных Compara (гомология), регуляторных элементов и кросс-ссылок.

---

## Возможности

- **Resume прерванных загрузок** — докачка с места обрыва через FTP-команду `REST`
- **Проверка целостности** — MD5-суммы (если доступны на FTP) + проверка размера
- **Параллельное скачивание** — через `ThreadPoolExecutor` (настраивается)
- **Манифест скачанных файлов** — JSON-файл со статусом каждого файла
- **Подробное логирование** — в файл (с ротацией по размеру) и в консоль
- **Гибкая настройка** — YAML-конфиг + переопределение через CLI
- **Dry-run режим** — посмотреть план без скачивания
- **Пропуск уже скачанных файлов** — идемпотентность повторных запусков

---

## Структура проекта

```
school14/
├── src/                          # Исходный код утилиты
│   ├── __init__.py
│   ├── config.py                 # Загрузка и валидация конфигурации
│   ├── logger.py                 # Настройка логирования
│   ├── ftp_client.py             # Обёртка над ftplib (resume, retry, progress)
│   ├── downloader.py             # Оркестратор скачивания
│   └── manifest.py               # Манифест скачанных файлов (JSON)
│
├── config/
│   └── config.yaml               # Конфигурация по умолчанию
│
├── main.py                       # CLI-точка входа (argparse)
├── requirements.txt              # Зависимости Python
├── ARCHITECTURE.md               # Подробная архитектура утилиты
└── README.md                     # Этот файл
```

---

## Установка

```bash
# Клонировать репозиторий
cd /home/User14/mag

# Создать виртуальное окружение
python3 -m venv venv
source venv/bin/activate

# Установить зависимости
pip install --upgrade pip
pip install -r requirements.txt
```

**Зависимости** (см. [`requirements.txt`](requirements.txt:1)):
- [`PyYAML>=6.0`](requirements.txt:1) — парсинг YAML-конфига
- [`tqdm>=4.65.0`](requirements.txt:2) — прогресс-бар
- [`tenacity>=8.2.0`](requirements.txt:3) — повторные попытки при ошибках

---

## Использование

### Базовый запуск

```bash
# Скачать всё по конфигурации
python main.py

# С указанным конфигом
python main.py --config /path/to/config.yaml
```

### Примеры CLI-команд

```bash
# Скачать только человека
python main.py --species homo_sapiens

# Скачать только GTF и protein для человека и мыши
python main.py \
  --species homo_sapiens,mus_musculus \
  --data-types gtf,fasta_pep

# Скачать конкретный релиз
python main.py --release 110

# Скачать последний доступный релиз (авто-определение с FTP)
python main.py --release latest
# -> подключается к ftp.ensembl.org, находит максимальный release-N
# -> сохраняет выбранный релиз в config.yaml

# Параллельное скачивание (4 потока)
python main.py --parallel 4

# Dry-run (показать план без скачивания)
python main.py --dry-run

# Перекачать существующие файлы
python main.py --force

# Без проверки checksum
python main.py --no-checksum

# Debug-режим
python main.py --verbose

# Вывести все источники данных
python main.py --list-sources

# Вывести поддерживаемые виды
python main.py --list-species

# Вывести статическую информацию о релизах
python main.py --list-releases

# Подключиться к FTP и вывести список всех доступных релизов
python main.py --list-releases-remote
```

### Все аргументы CLI

| Аргумент | Описание |
|----------|----------|
| `--config PATH` | Путь к YAML-конфигу (по умолчанию `config/config.yaml`) |
| `--species SPECIES` | Список видов через запятую |
| `--release RELEASE` | Номер релиза Ensembl (целое число или `latest`) |
| `--data-types TYPES` | Список типов данных через запятую |
| `--output-dir DIR` | Выходная директория |
| `--dry-run` | Только показать план |
| `--force` | Перекачать существующие файлы |
| `--list-species` | Вывести поддерживаемые виды |
| `--list-releases` | Вывести статическую информацию о релизах |
| `--list-releases-remote` | Подключиться к FTP и вывести список доступных релизов |
| `--list-sources` | Вывести все источники данных |
| `--verbose` | Подробный вывод (DEBUG) |
| `--log-level LEVEL` | Уровень логирования |
| `--parallel N` | Количество параллельных загрузок |
| `--no-checksum` | Отключить проверку MD5 |

---

## Что такое `release` и за что он отвечает

**Release (релиз)** — это версия базы данных Ensembl. Ensembl выпускает
несколько релизов в год; каждый релиз содержит обновлённые геномные сборки,
аннотации и данные по гомологии.

### Где используется `release`

Параметр `release` подставляется в путь на FTP Ensembl:

```
ftp://ftp.ensembl.org/pub/release-<N>/<subdir>/<species>/
```

Например, для релиза 110 и человека:

```
ftp://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/
ftp://ftp.ensembl.org/pub/release-110/gtf/homo_sapiens/
ftp://ftp.ensembl.org/pub/release-110/compara/homology/
```

То есть `release` определяет, **из какой версии базы данных** будут
скачиваться файлы. Разные релизы могут содержать:

- разные версии сборок (assembly): GRCh38 → GRCh38.p14
- разные аннотации генов (новые/удалённые гены, транскрипты)
- разные данные по гомологии (Compara)
- разные кросс-ссылки (XRef)

### Как задать `release`

1. **В конфиге** [`config/config.yaml`](config/config.yaml:1) — поле `release: <N>`.
2. **Через CLI** — `--release <N>` (например, `--release 110`).
3. **Через CLI с авто-определением** — `--release latest`:
   - утилита подключается к `ftp.ensembl.org`,
   - сканирует `/pub/` и находит все поддиректории `release-<N>`,
   - выбирает максимальный номер,
   - **сохраняет выбранный релиз в `config/config.yaml`**,
   - использует его для скачивания.

### Зачем сохранять в конфиг

После `--release latest` утилита записывает определённый номер в
`config/config.yaml`. Это нужно, чтобы:

- при следующих запусках **не подключаться к FTP лишний раз**;
- **зафиксировать версию** для воспроизводимости эксперимента;
- видеть в git-истории, какая версия данных использовалась.

### Посмотреть доступные релизы

```bash
# Статическая справка (без подключения к FTP)
python main.py --list-releases

# Реальный список с FTP (требуется подключение к интернету)
python main.py --list-releases-remote
```

---

## Конфигурация

Файл [`config/config.yaml`](config/config.yaml:1):

```yaml
ftp_host: ftp.ensembl.org
ftp_base_path: /pub
release: 110

species:
  - homo_sapiens
  - mus_musculus
  - danio_rerio

data_types:
  - fasta_dna
  - fasta_cdna
  - fasta_pep
  - gtf
  - gff3

output_dir: ./data
log_dir: ./logs

max_retries: 3
chunk_size: 8192
parallel_downloads: 2
timeout: 300
verify_checksum: true
force: false
```

---

## Скачиваемые данные

### Поддерживаемые типы данных

| Тип | Описание | Путь на FTP |
|-----|----------|-------------|
| `fasta_dna` | FASTA DNA (геномная последовательность) | `pub/release-X/fasta/<species>/dna/` |
| `fasta_cdna` | FASTA cDNA (кодирующие транскрипты) | `pub/release-X/fasta/<species>/cdna/` |
| `fasta_cds` | FASTA CDS (кодирующие последовательности) | `pub/release-X/fasta/<species>/cds/` |
| `fasta_pep` | FASTA protein (белковые последовательности) | `pub/release-X/fasta/<species>/pep/` |
| `fasta_ncrna` | FASTA ncRNA (некодирующие РНК) | `pub/release-X/fasta/<species>/ncrna/` |
| `gtf` | GTF-аннотация генов | `pub/release-X/gtf/<species>/` |
| `gff3` | GFF3-аннотация генов | `pub/release-X/gff3/<species>/` |
| `variation_vcf` | VCF-файлы вариантов | `pub/release-X/variation/vcf/<species>/` |
| `compara_homology` | Compara (гомология) | `pub/release-X/compara/homology/` |
| `regulation` | Регуляторные элементы (GFF) | `pub/release-X/regulation/<species>/` |
| `xref` | Кросс-ссылки идентификаторов (TSV) | `pub/release-X/xref/<species>/` |

### Структура локального хранилища

```
data/
├── homo_sapiens/
│   ├── fasta_dna/
│   │   └── Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
│   ├── fasta_cdna/
│   │   └── Homo_sapiens.GRCh38.cdna.all.fa.gz
│   ├── fasta_pep/
│   │   └── Homo_sapiens.GRCh38.pep.all.fa.gz
│   ├── gtf/
│   │   └── Homo_sapiens.GRCh38.110.gtf.gz
│   └── gff3/
│       └── Homo_sapiens.GRCh38.110.gff3.gz
├── mus_musculus/
│   └── ...
├── danio_rerio/
│   └── ...
└── manifest.json
```

---

## Структура FTP Ensembl

```
https://ftp.ensembl.org/pub/
├── release-110/
│   ├── fasta/
│   │   ├── homo_sapiens/
│   │   │   ├── dna/         # FASTA DNA
│   │   │   ├── cdna/        # FASTA cDNA
│   │   │   ├── cds/         # FASTA CDS
│   │   │   ├── pep/         # FASTA protein
│   │   │   └── ncrna/       # FASTA ncRNA
│   │   ├── mus_musculus/
│   │   └── ...
│   ├── gtf/
│   │   ├── homo_sapiens/
│   │   └── ...
│   ├── gff3/
│   ├── variation/
│   │   └── vcf/
│   ├── compara/
│   │   └── homology/
│   ├── regulation/
│   └── xref/
└── ...
```

---

## Манифест

Файл `data/manifest.json` содержит информацию о каждом скачанном файле:

```json
{
  "version": "1.0",
  "updated_at": "2026-07-29T14:25:00+00:00",
  "files": {
    "/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz": {
      "remote_path": "...",
      "local_path": "data/homo_sapiens/fasta_dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz",
      "size": 3141592653,
      "checksum": "a1b2c3d4...",
      "status": "verified",
      "species": "homo_sapiens",
      "data_type": "fasta_dna",
      "updated_at": "2026-07-29T14:25:00+00:00"
    }
  }
}
```

**Статусы:**
- `pending` — файл добавлен в манифест, но не скачивался
- `downloading` — идёт скачивание
- `completed` — скачан, размер совпадает
- `verified` — скачан и проверен по MD5
- `failed` — ошибка при скачивании

---

## Логирование

Логи пишутся в:
- `logs/ensembl_downloader.log` — все сообщения (с ротацией по 10 МБ)
- `logs/ensembl_downloader_errors.log` — только WARNING и выше
- stdout — консольный вывод

Формат: `timestamp | level | module | message`

Пример:
```
2026-07-29 14:25:00 | INFO     | ensembl_downloader | Начало скачивания: 3 видов, 5 типов данных
2026-07-29 14:25:01 | INFO     | ensembl_downloader.ftp | Подключение к FTP ftp.ensembl.org...
2026-07-29 14:25:02 | INFO     | ensembl_downloader.ftp | Успешное подключение к ftp.ensembl.org
2026-07-29 14:25:03 | INFO     | ensembl_downloader | Скачивание: /pub/release-110/fasta/homo_sapiens/dna/...
```

---

## Развёртывание на сервере

```bash
# Подключение
ssh User14@85.208.85.123

# Создание структуры каталогов
mkdir -p /home/User14/mag/{src,config,data,logs,tmp}
cd /home/User14/mag

# Копирование файлов (с локальной машины)
scp -r src/ config/ main.py requirements.txt User14@85.208.85.123:/home/User14/mag/

# Установка зависимостей
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Первый запуск (dry-run)
python main.py --dry-run

# Запуск скачивания
python main.py
```

---

## Обработка ошибок

| Ситуация | Поведение |
|----------|-----------|
| Таймаут соединения | Повторная попытка (до `max_retries` раз, экспоненциальная задержка) |
| Файл не найден на FTP | Логирование WARNING, пропуск |
| Размер не совпадает | Файл помечается как `failed`, можно перекачать через `--force` |
| Потеря соединения | Автоматическое переподключение + resume |
| Неподдерживаемый тип данных | Ошибка валидации при старте |

---

## Лицензия

Внутренняя утилита проекта school14.
