"""Модуль конфигурации утилиты Ensembl FTP Downloader.

Загружает параметры из YAML-файла и предоставляет типизированный доступ
к настройкам FTP-соединения, списку видов, релизам, типам данных и путям.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# Поддерживаемые типы данных и соответствующие им пути на FTP Ensembl.
# Структура FTP: https://ftp.ensembl.org/pub/release-<X>/<subdir>/<species>/
DATA_TYPE_PATHS: Dict[str, Dict[str, str]] = {
    "fasta_dna": {
        "subdir": "fasta",
        "species_subdir": "dna",
        "description": "FASTA DNA (genome sequence)",
        "file_pattern": "{species}.dna.{assembly}.fa.gz",
    },
    "fasta_cdna": {
        "subdir": "fasta",
        "species_subdir": "cdna",
        "description": "FASTA cDNA (coding transcripts)",
        "file_pattern": "{species}.cdna.all.fa.gz",
    },
    "fasta_cds": {
        "subdir": "fasta",
        "species_subdir": "cds",
        "description": "FASTA CDS (coding sequences)",
        "file_pattern": "{species}.cds.all.fa.gz",
    },
    "fasta_pep": {
        "subdir": "fasta",
        "species_subdir": "pep",
        "description": "FASTA protein (peptide sequences)",
        "file_pattern": "{species}.pep.all.fa.gz",
    },
    "fasta_ncrna": {
        "subdir": "fasta",
        "species_subdir": "ncrna",
        "description": "FASTA ncRNA (non-coding RNA)",
        "file_pattern": "{species}.ncrna.fa.gz",
    },
    "gtf": {
        "subdir": "gtf",
        "species_subdir": "",
        "description": "GTF gene annotation",
        "file_pattern": "{species}.{assembly}.gtf.gz",
    },
    "gff3": {
        "subdir": "gff3",
        "species_subdir": "",
        "description": "GFF3 gene annotation",
        "file_pattern": "{species}.{assembly}.gff3.gz",
    },
    "variation_vcf": {
        "subdir": "variation/vcf",
        "species_subdir": "",
        "description": "Variation data (VCF)",
        "file_pattern": "{species}.vcf.gz",
    },
    "compara_homology": {
        "subdir": "compara/homology",
        "species_subdir": "",
        "description": "Compara homology data",
        "file_pattern": "Compara.{homology_type}.{release}.gz",
    },
    "regulation": {
        "subdir": "regulation",
        "species_subdir": "",
        "description": "Regulatory features (GFF)",
        "file_pattern": "{species}.regulation.gff3.gz",
    },
    "xref": {
        "subdir": "xref",
        "species_subdir": "",
        "description": "Cross-reference mappings (TSV)",
        "file_pattern": "{species}.xref.tsv.gz",
    },
}


@dataclass
class Config:
    """Конфигурация утилиты Ensembl FTP Downloader.

    Атрибуты:
        ftp_host: Хост FTP-сервера Ensembl.
        ftp_base_path: Базовый путь на FTP (по умолчанию /pub).
        release: Номер релиза Ensembl (например, 110).
        species: Список видов для скачивания.
        data_types: Список типов данных для скачивания.
        output_dir: Локальная директория для сохранения файлов.
        log_dir: Директория для логов.
        max_retries: Максимальное количество повторных попыток.
        chunk_size: Размер чанка при скачивании (байт).
        parallel_downloads: Количество параллельных загрузок.
        timeout: Таймаут FTP-соединения (секунды).
        verify_checksum: Проверять ли MD5 после скачивания.
        force: Перезаписывать ли существующие файлы.
    """

    ftp_host: str = "ftp.ensembl.org"
    ftp_base_path: str = "/pub"
    release: int = 110
    species: List[str] = field(
        default_factory=lambda: ["homo_sapiens", "mus_musculus", "danio_rerio"]
    )
    data_types: List[str] = field(
        default_factory=lambda: [
            "fasta_dna",
            "fasta_cdna",
            "fasta_pep",
            "gtf",
            "gff3",
        ]
    )
    output_dir: Path = field(default_factory=lambda: Path("./data"))
    log_dir: Path = field(default_factory=lambda: Path("./logs"))
    max_retries: int = 3
    chunk_size: int = 8192
    parallel_downloads: int = 2
    timeout: int = 300
    verify_checksum: bool = True
    force: bool = False

    @classmethod
    def from_yaml(cls, config_path: Path) -> "Config":
        """Загружает конфигурацию из YAML-файла.

        Args:
            config_path: Путь к YAML-файлу конфигурации.

        Returns:
            Экземпляр Config с загруженными параметрами.

        Raises:
            FileNotFoundError: Если файл конфигурации не найден.
            yaml.YAMLError: Если файл содержит некорректный YAML.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}

        # Преобразуем строковые пути в Path
        if "output_dir" in data:
            data["output_dir"] = Path(data["output_dir"])
        if "log_dir" in data:
            data["log_dir"] = Path(data["log_dir"])

        # Фильтруем только известные поля
        valid_fields = {f for f in cls.__dataclass_fields__.keys()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered)

    def get_ftp_path(self, data_type: str, species: str) -> str:
        """Возвращает путь к директории на FTP для указанного типа данных и вида.

        Args:
            data_type: Тип данных (например, "fasta_dna", "gtf").
            species: Название вида (например, "homo_sapiens").

        Returns:
            Полный путь на FTP (например, "/pub/release-110/fasta/homo_sapiens/dna/").

        Raises:
            ValueError: Если указан неизвестный тип данных.
        """
        if data_type not in DATA_TYPE_PATHS:
            raise ValueError(
                f"Неизвестный тип данных: {data_type}. "
                f"Допустимые: {list(DATA_TYPE_PATHS.keys())}"
            )

        info = DATA_TYPE_PATHS[data_type]
        subdir = info["subdir"]
        species_subdir = info["species_subdir"]

        path = f"{self.ftp_base_path}/release-{self.release}/{subdir}/{species}"
        if species_subdir:
            path = f"{path}/{species_subdir}"
        return path + "/"

    def get_local_path(self, data_type: str, species: str) -> Path:
        """Возвращает локальный путь для сохранения файлов указанного типа и вида.

        Args:
            data_type: Тип данных.
            species: Название вида.

        Returns:
            Локальный путь (например, Path("./data/homo_sapiens/fasta_dna/")).
        """
        return self.output_dir / species / data_type

    def get_file_pattern(self, data_type: str) -> str:
        """Возвращает шаблон имени файла для указанного типа данных.

        Args:
            data_type: Тип данных.

        Returns:
            Шаблон имени файла с плейсхолдерами.

        Raises:
            ValueError: Если указан неизвестный тип данных.
        """
        if data_type not in DATA_TYPE_PATHS:
            raise ValueError(f"Неизвестный тип данных: {data_type}")
        return DATA_TYPE_PATHS[data_type]["file_pattern"]

    def get_data_type_description(self, data_type: str) -> str:
        """Возвращает человекочитаемое описание типа данных.

        Args:
            data_type: Тип данных.

        Returns:
            Описание типа данных.
        """
        if data_type not in DATA_TYPE_PATHS:
            return "Unknown"
        return DATA_TYPE_PATHS[data_type]["description"]

    def list_data_types(self) -> List[str]:
        """Возвращает список всех поддерживаемых типов данных."""
        return list(DATA_TYPE_PATHS.keys())

    def override(
        self,
        species: Optional[List[str]] = None,
        release: Optional[int] = None,
        data_types: Optional[List[str]] = None,
        output_dir: Optional[Path] = None,
        force: Optional[bool] = None,
    ) -> None:
        """Переопределяет параметры конфигурации из CLI-аргументов.

        Args:
            species: Новый список видов.
            release: Новый номер релиза.
            data_types: Новый список типов данных.
            output_dir: Новая выходная директория.
            force: Перезаписывать ли существующие файлы.
        """
        if species is not None:
            self.species = species
        if release is not None:
            self.release = release
        if data_types is not None:
            self.data_types = data_types
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        if force is not None:
            self.force = force

    def save_release(self, config_path: Path, release: Optional[int] = None) -> None:
        """Сохраняет значение ``release`` обратно в YAML-файл конфигурации.

        Используется после того, как пользователь выбрал релиз через CLI
        (например, ``--release latest``), чтобы зафиксировать выбор в файле
        и не запрашивать его повторно при следующих запусках.

        Сохраняет только поле ``release``; остальные параметры не трогает.
        Если файл не существует или повреждён, создаёт минимальный конфиг
        с единственным полем ``release``.

        Args:
            config_path: Путь к YAML-файлу конфигурации.
            release: Номер релиза для записи. Если None — используется
                текущее значение ``self.release``.
        """
        config_path = Path(config_path)
        target_release = self.release if release is None else release

        data: Dict[str, Any] = {}
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        data = loaded
            except (yaml.YAMLError, OSError):
                # Если файл повреждён — перезаписываем минимальным конфигом
                data = {}

        data["release"] = int(target_release)

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
