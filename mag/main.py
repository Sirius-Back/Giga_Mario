"""CLI-точка входа утилиты Ensembl FTP Downloader.

Позволяет запускать скачивание данных с FTP Ensembl с настройкой
параметров через аргументы командной строки.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from src.config import Config, DATA_TYPE_PATHS
from src.downloader import EnsemblDownloader
from src.logger import setup_logger


def build_parser() -> argparse.ArgumentParser:
    """Создаёт парсер аргументов командной строки.

    Returns:
        Настроенный ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="ensembl-downloader",
        description=(
            "Утилита для скачивания геномных данных с FTP Ensembl "
            "(https://ftp.ensembl.org/pub/)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Путь к YAML-файлу конфигурации (по умолчанию: config/config.yaml)",
    )

    parser.add_argument(
        "--species",
        type=str,
        help="Список видов через запятую (переопределяет конфиг). Пример: homo_sapiens,mus_musculus",
    )

    parser.add_argument(
        "--release",
        type=str,
        help=(
            "Номер релиза Ensembl (переопределяет конфиг). "
            "Допустимые значения: целое число (например, 110) или "
            "'latest' для автоматического определения последнего релиза "
            "с FTP Ensembl. Выбранное значение сохраняется в config.yaml."
        ),
    )

    parser.add_argument(
        "--data-types",
        type=str,
        help=(
            "Список типов данных через запятую (переопределяет конфиг). "
            "Пример: fasta_dna,gtf,gff3"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Выходная директория (переопределяет конфиг)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать план скачивания без реальной загрузки",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Перекачать существующие файлы (по умолчанию — пропускать)",
    )

    parser.add_argument(
        "--list-species",
        action="store_true",
        help="Вывести список поддерживаемых видов и завершить работу",
    )

    parser.add_argument(
        "--list-releases",
        action="store_true",
        help="Вывести статическую информацию о релизах и завершить работу",
    )

    parser.add_argument(
        "--list-releases-remote",
        action="store_true",
        help="Подключиться к FTP Ensembl и вывести список доступных релизов",
    )

    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="Вывести все источники данных (типы файлов, пути на FTP, описания)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Подробный вывод (уровень логирования DEBUG)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Уровень логирования (по умолчанию: INFO)",
    )

    parser.add_argument(
        "--parallel",
        type=int,
        help="Количество параллельных загрузок (переопределяет конфиг)",
    )

    parser.add_argument(
        "--no-checksum",
        action="store_true",
        help="Отключить проверку контрольных сумм",
    )

    return parser


def parse_csv(value: Optional[str]) -> Optional[List[str]]:
    """Разбирает строку с разделителем-запятой в список.

    Args:
        value: Строка вида "a,b,c" или None.

    Returns:
        Список строк или None.
    """
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def cmd_list_species() -> int:
    """Выводит список поддерживаемых видов.

    Returns:
        Код возврата (0 — успех).
    """
    print("Поддерживаемые виды (по умолчанию в конфиге):")
    print("  - homo_sapiens    (Human,    GRCh38)")
    print("  - mus_musculus    (Mouse,    GRCm39)")
    print("  - danio_rerio     (Zebrafish, GRCz11)")
    print()
    print("Полный список видов Ensembl доступен на:")
    print("  https://ftp.ensembl.org/pub/release-<X>/fasta/")
    print("  https://ftp.ensembl.org/pub/release-<X>/gtf/")
    print()
    print("Для добавления нового вида отредактируйте config/config.yaml")
    return 0


def cmd_list_releases() -> int:
    """Выводит информацию о релизах Ensembl.

    Returns:
        Код возврата (0 — успех).
    """
    print("Информация о релизах Ensembl:")
    print()
    print("  Релиз 110 — 2023-10 (текущий по умолчанию)")
    print("  Релиз 111 — 2024-02")
    print("  Релиз 112 — 2024-06")
    print()
    print("Список всех релизов:")
    print("  https://ftp.ensembl.org/pub/")
    print()
    print("Для выбора релиза используйте --release <N> или release в config.yaml")
    return 0


def cmd_list_releases_remote(config: Config) -> int:
    """Подключается к FTP Ensembl и выводит список доступных релизов.

    Args:
        config: Конфигурация утилиты (используется ftp_host и ftp_base_path).

    Returns:
        Код возврата (0 — успех, 1 — ошибка).
    """
    from src.ftp_client import EnsemblFTPClient, FTPConnectionError, FileNotFoundOnFTPError

    print(f"Подключение к {config.ftp_host} для получения списка релизов...")
    client = EnsemblFTPClient(host=config.ftp_host, timeout=config.timeout)
    try:
        client.connect()
    except FTPConnectionError as e:
        print(f"Ошибка подключения: {e}", file=sys.stderr)
        return 1
    try:
        try:
            releases = client.list_releases(base_path=config.ftp_base_path)
        except FileNotFoundOnFTPError as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            return 1
    finally:
        client.disconnect()

    if not releases:
        print("Релизы не найдены.")
        return 1

    print(f"Доступные релизы Ensembl ({len(releases)} шт.):")
    print(f"  Минимальный: {releases[0]}")
    print(f"  Максимальный (latest): {releases[-1]}")
    print()
    # Показываем последние 20 релизов для удобства
    print("Последние 20 релизов:")
    for r in releases[-20:]:
        marker = "  <-- latest" if r == releases[-1] else ""
        print(f"  release-{r}{marker}")
    print()
    print("Использование:")
    print(f"  python main.py --release {releases[-1]}        # скачать последний релиз")
    print("  python main.py --release latest                # то же самое, авто-определение")
    print("  python main.py --release 110                   # скачать конкретный релиз")
    return 0


def cmd_list_sources(config: Config) -> int:
    """Выводит все источники данных (типы файлов, пути на FTP, описания).

    Args:
        config: Конфигурация утилиты.

    Returns:
        Код возврата (0 — успех).
    """
    print(f"Источники данных Ensembl (релиз {config.release}):")
    print("=" * 80)
    for data_type, info in DATA_TYPE_PATHS.items():
        print(f"\n[{data_type}]")
        print(f"  Описание: {info['description']}")
        print(f"  Поддиректория: {info['subdir']}")
        if info.get("species_subdir"):
            print(f"  Поддиректория вида: {info['species_subdir']}")
        print(f"  Шаблон файла: {info['file_pattern']}")
        print(f"  Пример FTP-пути: {config.get_ftp_path(data_type, 'homo_sapiens')}")
        print(f"  Локальный путь: {config.get_local_path(data_type, 'homo_sapiens')}/")
    print()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Главная точка входа CLI.

    Args:
        argv: Аргументы командной строки (если None — берутся из sys.argv).

    Returns:
        Код возврата (0 — успех, 1 — ошибка).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Загружаем конфигурацию
    try:
        config = Config.from_yaml(args.config)
    except FileNotFoundError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Ошибка загрузки конфигурации: {e}", file=sys.stderr)
        return 1

    # Команды вывода информации (не требуют подключения к FTP)
    if args.list_species:
        return cmd_list_species()
    if args.list_releases:
        return cmd_list_releases()
    if args.list_releases_remote:
        return cmd_list_releases_remote(config)
    if args.list_sources:
        return cmd_list_sources(config)

    # Переопределение параметров из CLI
    species = parse_csv(args.species)
    data_types = parse_csv(args.data_types)

    # Обработка --release: поддержка целого числа и значения 'latest'
    release_value: Optional[int] = None
    release_was_explicit = False
    if args.release is not None:
        release_was_explicit = True
        raw = str(args.release).strip().lower()
        if raw in ("latest", "l", "newest", "current"):
            # Автоматически определяем последний релиз с FTP
            from src.ftp_client import (
                EnsemblFTPClient,
                FTPConnectionError,
                FileNotFoundOnFTPError,
            )

            print(f"--release=latest: подключение к {config.ftp_host}...")
            client = EnsemblFTPClient(host=config.ftp_host, timeout=config.timeout)
            try:
                client.connect()
                try:
                    latest = client.get_latest_release(base_path=config.ftp_base_path)
                except FileNotFoundOnFTPError as e:
                    print(f"Ошибка: {e}", file=sys.stderr)
                    return 1
            except FTPConnectionError as e:
                print(f"Ошибка подключения к FTP: {e}", file=sys.stderr)
                return 1
            finally:
                client.disconnect()
            release_value = latest
            print(f"--release=latest: определён релиз {latest}")
        else:
            try:
                release_value = int(raw)
            except ValueError:
                print(
                    f"Ошибка: неверное значение --release: {args.release!r}. "
                    f"Ожидается целое число или 'latest'.",
                    file=sys.stderr,
                )
                return 1

    config.override(
        species=species,
        release=release_value,
        data_types=data_types,
        output_dir=args.output_dir,
        force=args.force if args.force else None,
    )

    # Если релиз был запрошен явно (включая 'latest') — сохраняем в config.yaml
    if release_was_explicit and release_value is not None:
        try:
            config.save_release(args.config, release=release_value)
            print(f"Релиз {release_value} сохранён в {args.config}")
        except OSError as e:
            print(
                f"Предупреждение: не удалось сохранить релиз в {args.config}: {e}",
                file=sys.stderr,
            )
    if args.parallel is not None:
        config.parallel_downloads = args.parallel
    if args.no_checksum:
        config.verify_checksum = False

    # Настраиваем логирование
    log_level = "DEBUG" if args.verbose else args.log_level
    log_dir = config.log_dir
    if log_dir is not None and not log_dir.is_absolute():
        log_dir = (Path(__file__).resolve().parent / log_dir).resolve()
    logger = setup_logger(
        name="ensembl_downloader",
        log_dir=log_dir,
        level=log_level,
        console=True,
    )

    if log_dir is not None:
        print(
            f"Логи пишутся в: {(log_dir / 'ensembl_downloader.log').resolve()}",
            file=sys.stderr,
        )
    logger.info(f"Конфигурация загружена из {args.config}")
    logger.info(f"Релиз: {config.release}")
    logger.info(f"Виды: {config.species}")
    logger.info(f"Типы данных: {config.data_types}")
    logger.info(f"Выходная директория: {config.output_dir}")

    # Создаём загрузчик
    downloader = EnsemblDownloader(config=config, logger=logger)

    # Dry-run: только план
    if args.dry_run:
        downloader.print_plan()
        return 0

    # Запускаем скачивание
    try:
        results = downloader.download_all()
        failed = [r for r in results if not r.success and not r.skipped]
        if failed:
            logger.warning(f"Не удалось скачать {len(failed)} файлов")
            return 2
        return 0
    except KeyboardInterrupt:
        logger.warning("Скачивание прервано пользователем (Ctrl+C)")
        return 130
    except Exception as e:
        logger.exception(f"Критическая ошибка: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
