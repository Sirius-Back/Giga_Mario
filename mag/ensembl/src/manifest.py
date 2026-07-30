"""Модуль манифеста скачанных файлов.

Манифест хранится в формате JSON и содержит информацию о каждом
скачанном файле: путь на FTP, локальный путь, размер, контрольную сумму,
статус и временную метку.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# Статусы файлов в манифесте
STATUS_PENDING = "pending"
STATUS_DOWNLOADING = "downloading"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_VERIFIED = "verified"


class Manifest:
    """Манифест скачанных файлов.

    Attributes:
        manifest_path: Путь к JSON-файлу манифеста.
        data: Словарь с записями о файлах (ключ — remote_path).
    """

    def __init__(
        self,
        manifest_path: Path,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Инициализирует манифест.

        Args:
            manifest_path: Путь к JSON-файлу манифеста.
            logger: Логгер для вывода сообщений.
        """
        self.manifest_path = Path(manifest_path)
        self.logger = logger or logging.getLogger("ensembl_downloader.manifest")
        self.data: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Загружает манифест из файла, если он существует."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                # Поддерживаем два формата: {"files": {...}} или просто {...}
                if isinstance(raw, dict) and "files" in raw:
                    self.data = raw["files"]
                elif isinstance(raw, dict):
                    self.data = raw
                else:
                    self.data = {}
                self.logger.debug(
                    f"Загружен манифест: {len(self.data)} записей"
                )
            except (json.JSONDecodeError, OSError) as e:
                self.logger.warning(f"Не удалось загрузить манифест: {e}")
                self.data = {}

    def save(self) -> None:
        """Сохраняет манифест в JSON-файл."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "1.0",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "files": self.data,
        }
        # Атомарная запись через временный файл
        tmp_path = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        tmp_path.replace(self.manifest_path)
        self.logger.debug(f"Манифест сохранён: {self.manifest_path}")

    def add_file(
        self,
        remote_path: str,
        local_path: str,
        size: int = 0,
        checksum: Optional[str] = None,
        status: str = STATUS_PENDING,
        species: Optional[str] = None,
        data_type: Optional[str] = None,
    ) -> None:
        """Добавляет или обновляет запись о файле в манифесте.

        Args:
            remote_path: Путь к файлу на FTP.
            local_path: Локальный путь к файлу.
            size: Размер файла в байтах.
            checksum: MD5-сумма файла (если есть).
            status: Статус файла.
            species: Название вида.
            data_type: Тип данных.
        """
        self.data[remote_path] = {
            "remote_path": remote_path,
            "local_path": str(local_path),
            "size": size,
            "checksum": checksum,
            "status": status,
            "species": species,
            "data_type": data_type,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def update_status(
        self,
        remote_path: str,
        status: str,
        size: Optional[int] = None,
        checksum: Optional[str] = None,
    ) -> None:
        """Обновляет статус файла в манифесте.

        Args:
            remote_path: Путь к файлу на FTP.
            status: Новый статус.
            size: Новый размер (если изменился).
            checksum: Новая контрольная сумма.
        """
        if remote_path not in self.data:
            self.data[remote_path] = {
                "remote_path": remote_path,
                "local_path": "",
                "size": 0,
                "checksum": None,
                "status": status,
                "species": None,
                "data_type": None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        self.data[remote_path]["status"] = status
        if size is not None:
            self.data[remote_path]["size"] = size
        if checksum is not None:
            self.data[remote_path]["checksum"] = checksum
        self.data[remote_path]["updated_at"] = datetime.now(timezone.utc).isoformat()

    def is_downloaded(self, remote_path: str) -> bool:
        """Проверяет, был ли файл успешно скачан.

        Args:
            remote_path: Путь к файлу на FTP.

        Returns:
            True, если файл имеет статус completed или verified.
        """
        entry = self.data.get(remote_path)
        if not entry:
            return False
        return entry.get("status") in (STATUS_COMPLETED, STATUS_VERIFIED)

    def get_entry(self, remote_path: str) -> Optional[Dict[str, Any]]:
        """Возвращает запись о файле.

        Args:
            remote_path: Путь к файлу на FTP.

        Returns:
            Словарь с информацией о файле или None.
        """
        return self.data.get(remote_path)

    def list_files(self) -> List[Dict[str, Any]]:
        """Возвращает список всех записей манифеста."""
        return list(self.data.values())

    def get_stats(self) -> Dict[str, int]:
        """Возвращает статистику по манифесту.

        Returns:
            Словарь со счётчиками по каждому статусу.
        """
        stats: Dict[str, int] = {}
        for entry in self.data.values():
            status = entry.get("status", "unknown")
            stats[status] = stats.get(status, 0) + 1
        return stats

    def clear(self) -> None:
        """Очищает манифест."""
        self.data.clear()
