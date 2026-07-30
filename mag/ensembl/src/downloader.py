"""Основной модуль скачивания данных с FTP Ensembl.

Оркестрирует работу FTP-клиента, конфигурации и манифеста.
Поддерживает:
- скачивание всех типов данных для одного вида
- скачивание всего по конфигурации
- resume прерванных загрузок
- проверку размера файла
- параллельное скачивание через ThreadPoolExecutor
"""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from .config import Config, DATA_TYPE_PATHS
from .ftp_client import (
    EnsemblFTPClient,
    FileNotFoundOnFTPError,
    FTPConnectionError,
)
from .manifest import (
    Manifest,
    STATUS_COMPLETED,
    STATUS_DOWNLOADING,
    STATUS_FAILED,
    STATUS_VERIFIED,
)


@dataclass
class DownloadResult:
    """Результат скачивания одного файла.

    Attributes:
        remote_path: Путь к файлу на FTP.
        local_path: Локальный путь к файлу.
        success: Успешно ли завершено скачивание.
        size: Размер скачанного файла в байтах.
        checksum: MD5-сумма файла (если проверялась).
        error: Текст ошибки (если была).
        skipped: Был ли файл пропущен (уже скачан).
    """

    remote_path: str
    local_path: Path
    success: bool
    size: int = 0
    checksum: Optional[str] = None
    error: Optional[str] = None
    skipped: bool = False


class EnsemblDownloader:
    """Оркестратор скачивания данных с FTP Ensembl.

    Attributes:
        config: Конфигурация утилиты.
        logger: Логгер.
        manifest: Манифест скачанных файлов.
        ftp_client: FTP-клиент.
    """

    def __init__(
        self,
        config: Config,
        logger: logging.Logger,
        manifest: Optional[Manifest] = None,
        ftp_client: Optional[EnsemblFTPClient] = None,
    ) -> None:
        """Инициализирует загрузчик.

        Args:
            config: Конфигурация утилиты.
            logger: Логгер.
            manifest: Манифест (если None — создаётся автоматически).
            ftp_client: FTP-клиент (если None — создаётся автоматически).
        """
        self.config = config
        self.logger = logger
        self.manifest = manifest or Manifest(
            config.output_dir / "manifest.json",
            logger=logger,
        )
        self.ftp_client = ftp_client or EnsemblFTPClient(
            host=config.ftp_host,
            timeout=config.timeout,
            max_retries=config.max_retries,
            chunk_size=config.chunk_size,
            logger=logger,
        )

    # ------------------------------------------------------------------ #
    # Публичные методы
    # ------------------------------------------------------------------ #

    def download_all(self) -> List[DownloadResult]:
        """Скачивает все данные согласно конфигурации.

        Returns:
            Список результатов скачивания.
        """
        self.logger.info(
            f"Начало скачивания: {len(self.config.species)} видов, "
            f"{len(self.config.data_types)} типов данных, релиз {self.config.release}"
        )

        all_results: List[DownloadResult] = []
        try:
            self.ftp_client.connect()
            for species in self.config.species:
                results = self.download_species(species, self.config.data_types)
                all_results.extend(results)
        finally:
            self.ftp_client.disconnect()
            self.manifest.save()

        # Итоговая статистика
        successful = sum(1 for r in all_results if r.success)
        skipped = sum(1 for r in all_results if r.skipped)
        failed = sum(1 for r in all_results if not r.success and not r.skipped)
        self.logger.info(
            f"Скачивание завершено: успешно {successful}, "
            f"пропущено {skipped}, ошибок {failed}"
        )
        return all_results

    def download_species(
        self,
        species: str,
        data_types: Optional[List[str]] = None,
    ) -> List[DownloadResult]:
        """Скачивает все указанные типы данных для одного вида.

        Args:
            species: Название вида (например, "homo_sapiens").
            data_types: Список типов данных. Если None — берётся из конфига.

        Returns:
            Список результатов скачивания.
        """
        if data_types is None:
            data_types = self.config.data_types

        self.logger.info(f"=== Скачивание для вида: {species} ===")
        results: List[DownloadResult] = []

        # Собираем список задач (remote_path, local_path, data_type)
        tasks: List[Tuple[str, Path, str]] = []
        for data_type in data_types:
            try:
                remote_dir = self.config.get_ftp_path(data_type, species)
                local_dir = self.config.get_local_path(data_type, species)
                local_dir.mkdir(parents=True, exist_ok=True)

                # Получаем список файлов на FTP
                try:
                    files = self.ftp_client.list_files(remote_dir)
                except FileNotFoundOnFTPError:
                    self.logger.warning(
                        f"Директория не найдена на FTP: {remote_dir}"
                    )
                    continue

                if not files:
                    self.logger.warning(
                        f"Нет файлов в {remote_dir} для {data_type}"
                    )
                    continue

                # Берём первый подходящий файл (для FASTA — основной)
                target_file = self._select_target_file(files, data_type)
                if not target_file:
                    self.logger.warning(
                        f"Не удалось выбрать файл из {files} для {data_type}"
                    )
                    continue

                remote_path = remote_dir + target_file
                local_path = local_dir / target_file
                tasks.append((remote_path, local_path, data_type))

            except Exception as e:
                self.logger.error(
                    f"Ошибка при подготовке {data_type} для {species}: {e}"
                )

        # Скачиваем (последовательно или параллельно)
        if self.config.parallel_downloads > 1 and len(tasks) > 1:
            results = self._download_parallel(tasks, species)
        else:
            results = self._download_sequential(tasks, species)

        # Сохраняем манифест после каждого вида
        self.manifest.save()
        return results

    def print_plan(self) -> None:
        """Выводит план скачивания (для режима --dry-run)."""
        self.logger.info("=== План скачивания (dry-run) ===")
        self.logger.info(f"FTP хост: {self.config.ftp_host}")
        self.logger.info(f"Релиз: {self.config.release}")
        self.logger.info(f"Виды: {', '.join(self.config.species)}")
        self.logger.info(f"Типы данных: {', '.join(self.config.data_types)}")
        self.logger.info(f"Выходная директория: {self.config.output_dir}")
        self.logger.info("")

        total_files = 0
        for species in self.config.species:
            self.logger.info(f"Вид: {species}")
            for data_type in self.config.data_types:
                try:
                    remote_dir = self.config.get_ftp_path(data_type, species)
                    local_dir = self.config.get_local_path(data_type, species)
                    desc = self.config.get_data_type_description(data_type)
                    self.logger.info(
                        f"  [{data_type}] {desc}\n"
                        f"    FTP: {remote_dir}\n"
                        f"    Local: {local_dir}/"
                    )
                    total_files += 1
                except ValueError as e:
                    self.logger.warning(f"  {e}")
        self.logger.info(f"\nВсего задач: {total_files}")

    # ------------------------------------------------------------------ #
    # Внутренние методы
    # ------------------------------------------------------------------ #

    def _select_target_file(self, files: List[str], data_type: str) -> Optional[str]:
        """Выбирает целевой файл из списка файлов на FTP.

        Приоритеты:
        - для fasta_dna: основной файл *.dna.primary_assembly.fa.gz или *.dna.fa.gz
        - для остальных: первый файл, подходящий под шаблон

        Args:
            files: Список имён файлов.
            data_type: Тип данных.

        Returns:
            Имя выбранного файла или None.
        """
        if not files:
            return None

        # Приоритетные суффиксы для разных типов
        priority_suffixes = {
            "fasta_dna": ["primary_assembly", "toplevel", ""],
            "fasta_cdna": ["all", ""],
            "fasta_cds": ["all", ""],
            "fasta_pep": ["all", ""],
            "fasta_ncrna": ["", "all"],
            "gtf": ["", "primary_assembly"],
            "gff3": ["", "primary_assembly"],
        }

        suffixes = priority_suffixes.get(data_type, [""])

        for suffix in suffixes:
            for f in files:
                if f.endswith(".fa.gz") or f.endswith(".fa.bgz") or \
                   f.endswith(".gtf.gz") or f.endswith(".gff3.gz") or \
                   f.endswith(".vcf.gz") or f.endswith(".tsv.gz"):
                    if suffix == "" or suffix in f:
                        return f

        # Если ничего не подошло — возвращаем первый файл
        return files[0]

    def _download_sequential(
        self,
        tasks: List[Tuple[str, Path, str]],
        species: str,
    ) -> List[DownloadResult]:
        """Скачивает файлы последовательно.

        Args:
            tasks: Список кортежей (remote_path, local_path, data_type).
            species: Название вида.

        Returns:
            Список результатов скачивания.
        """
        results: List[DownloadResult] = []
        for remote_path, local_path, data_type in tasks:
            result = self._download_single_file(
                remote_path, local_path, species, data_type
            )
            results.append(result)
        return results

    def _download_parallel(
        self,
        tasks: List[Tuple[str, Path, str]],
        species: str,
    ) -> List[DownloadResult]:
        """Скачивает файлы параллельно через ThreadPoolExecutor.

        Args:
            tasks: Список кортежей (remote_path, local_path, data_type).
            species: Название вида.

        Returns:
            Список результатов скачивания.
        """
        results: List[DownloadResult] = []
        max_workers = min(self.config.parallel_downloads, len(tasks))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(
                    self._download_single_file, rp, lp, species, dt
                ): (rp, lp)
                for rp, lp, dt in tasks
            }
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    rp, lp = future_to_task[future]
                    self.logger.error(f"Ошибка при скачивании {rp}: {e}")
                    results.append(
                        DownloadResult(
                            remote_path=rp,
                            local_path=lp,
                            success=False,
                            error=str(e),
                        )
                    )
        return results

    def _download_single_file(
        self,
        remote_path: str,
        local_path: Path,
        species: Optional[str] = None,
        data_type: Optional[str] = None,
    ) -> DownloadResult:
        """Скачивает один файл с проверкой и обновлением манифеста.

        Args:
            remote_path: Путь к файлу на FTP.
            local_path: Локальный путь для сохранения.
            species: Название вида (для манифеста).
            data_type: Тип данных (для манифеста).

        Returns:
            Результат скачивания.
        """
        # Проверяем, не скачан ли уже файл
        if not self.config.force and self.manifest.is_downloaded(remote_path):
            entry = self.manifest.get_entry(remote_path)
            if entry and Path(entry.get("local_path", "")).exists():
                self.logger.info(f"Пропуск (уже скачан): {local_path.name}")
                return DownloadResult(
                    remote_path=remote_path,
                    local_path=local_path,
                    success=True,
                    size=entry.get("size", 0),
                    checksum=entry.get("checksum"),
                    skipped=True,
                )

        self.logger.info(f"Скачивание: {remote_path}")
        # Добавляем/обновляем запись в манифесте с полной информацией
        self.manifest.add_file(
            remote_path=remote_path,
            local_path=str(local_path),
            size=0,
            status=STATUS_DOWNLOADING,
            species=species,
            data_type=data_type,
        )

        try:
            # Получаем размер файла
            try:
                expected_size = self.ftp_client.file_size(remote_path)
            except FileNotFoundOnFTPError:
                self.logger.warning(f"Файл не найден на FTP: {remote_path}")
                self.manifest.update_status(remote_path, STATUS_FAILED)
                return DownloadResult(
                    remote_path=remote_path,
                    local_path=local_path,
                    success=False,
                    error="File not found on FTP",
                )

            # Скачиваем
            self.ftp_client.download_file(remote_path, local_path)

            # Проверяем размер
            if not self._verify_file(local_path, expected_size):
                self.logger.error(
                    f"Размер не совпадает для {local_path.name}: "
                    f"ожидалось {expected_size}, получено {local_path.stat().st_size}"
                )
                self.manifest.update_status(remote_path, STATUS_FAILED)
                return DownloadResult(
                    remote_path=remote_path,
                    local_path=local_path,
                    success=False,
                    error="Size mismatch",
                )

            # Получаем checksum (если включено)
            checksum = None
            if self.config.verify_checksum:
                try:
                    checksum = self.ftp_client.get_checksum(remote_path)
                except Exception as e:
                    self.logger.debug(f"Не удалось получить checksum: {e}")

            # Обновляем манифест
            self.manifest.update_status(
                remote_path,
                STATUS_VERIFIED if checksum else STATUS_COMPLETED,
                size=expected_size,
                checksum=checksum,
            )

            return DownloadResult(
                remote_path=remote_path,
                local_path=local_path,
                success=True,
                size=expected_size,
                checksum=checksum,
            )

        except (FTPConnectionError, FileNotFoundOnFTPError) as e:
            self.logger.error(f"Ошибка FTP при скачивании {remote_path}: {e}")
            self.manifest.update_status(remote_path, STATUS_FAILED)
            return DownloadResult(
                remote_path=remote_path,
                local_path=local_path,
                success=False,
                error=str(e),
            )
        except Exception as e:
            self.logger.exception(f"Неожиданная ошибка при скачивании {remote_path}")
            self.manifest.update_status(remote_path, STATUS_FAILED)
            return DownloadResult(
                remote_path=remote_path,
                local_path=local_path,
                success=False,
                error=str(e),
            )

    def _verify_file(self, local_path: Path, expected_size: int) -> bool:
        """Проверяет, что локальный файл имеет ожидаемый размер.

        Args:
            local_path: Путь к локальному файлу.
            expected_size: Ожидаемый размер в байтах.

        Returns:
            True, если размер совпадает.
        """
        if not local_path.exists():
            return False
        actual_size = local_path.stat().st_size
        return actual_size == expected_size

    def calculate_local_md5(self, local_path: Path) -> str:
        """Вычисляет MD5 локального файла.

        Args:
            local_path: Путь к файлу.

        Returns:
            MD5-сумма в виде hex-строки.
        """
        md5 = hashlib.md5()
        with open(local_path, "rb") as f:
            while True:
                chunk = f.read(self.config.chunk_size)
                if not chunk:
                    break
                md5.update(chunk)
        return md5.hexdigest()
