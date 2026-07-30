"""Модуль настройки логирования.

Предоставляет функцию setup_logger для создания логгера с выводом
в файл (с ротацией по размеру) и в консоль.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# Формат логов: timestamp | level | module | message
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Размер файла лога по умолчанию: 10 МБ
DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 5


def setup_logger(
    name: str = "ensembl_downloader",
    log_dir: Optional[Path] = None,
    level: str = "INFO",
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    console: bool = True,
) -> logging.Logger:
    """Настраивает и возвращает логгер.

    Логгер пишет в:
    - файл logs/<name>.log (с ротацией по размеру)
    - файл logs/<name>_errors.log (только ошибки)
    - stdout (если console=True)

    Args:
        name: Имя логгера.
        log_dir: Директория для логов. Если None — только консоль.
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        max_bytes: Максимальный размер файла лога в байтах.
        backup_count: Количество резервных файлов при ротации.
        console: Выводить ли логи в консоль.

    Returns:
        Настроенный экземпляр logging.Logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Удаляем существующие обработчики, чтобы избежать дублирования
    logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Файловый обработчик с ротацией
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Основной лог-файл
        log_file = log_dir / f"{name}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Файл только для ошибок
        error_file = log_dir / f"{name}_errors.log"
        error_handler = RotatingFileHandler(
            error_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

    # Консольный обработчик
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Не пробрасываем сообщения в корневой логгер
    logger.propagate = False

    return logger


def get_logger(name: str = "ensembl_downloader") -> logging.Logger:
    """Возвращает существующий логгер по имени.

    Args:
        name: Имя логгера.

    Returns:
        Экземпляр logging.Logger.
    """
    return logging.getLogger(name)
