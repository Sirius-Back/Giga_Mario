"""FTP-клиент для работы с Ensembl.

Обёртка над ftplib с поддержкой:
- анонимного подключения
- resume прерванных загрузок (REST + APPE)
- прогресс-бара через tqdm
- получения контрольных сумм (MD5)
- повторных попыток при ошибках (через tenacity)
"""

from __future__ import annotations

import ftplib
import io
import logging
from pathlib import Path
from typing import Callable, List, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm


# Исключения, при которых выполняется повторная попытка
RETRYABLE_EXCEPTIONS = (
    ftplib.error_temp,
    ftplib.error_reply,
    ConnectionError,
    TimeoutError,
    OSError,
)


class EnsemblFTPError(Exception):
    """Базовое исключение EnsemblFTPClient."""


class FTPConnectionError(EnsemblFTPError):
    """Ошибка подключения к FTP."""


class FileNotFoundOnFTPError(EnsemblFTPError):
    """Файл не найден на FTP."""


class EnsemblFTPClient:
    """Обёртка над ftplib для работы с FTP Ensembl.

    Attributes:
        host: Хост FTP-сервера.
        timeout: Таймаут соединения в секундах.
        max_retries: Максимальное количество повторных попыток.
        chunk_size: Размер чанка при скачивании (байт).
    """

    def __init__(
        self,
        host: str = "ftp.ensembl.org",
        timeout: int = 300,
        max_retries: int = 3,
        chunk_size: int = 8192,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Инициализирует FTP-клиент.

        Args:
            host: Хост FTP-сервера.
            timeout: Таймаут соединения в секундах.
            max_retries: Максимальное количество повторных попыток.
            chunk_size: Размер чанка при скачивании (байт).
            logger: Логгер для вывода сообщений.
        """
        self.host = host
        self.timeout = timeout
        self.max_retries = max_retries
        self.chunk_size = chunk_size
        self.logger = logger or logging.getLogger("ensembl_downloader.ftp")
        self._ftp: Optional[ftplib.FTP] = None

    # ------------------------------------------------------------------ #
    # Подключение / отключение
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        """Устанавливает анонимное соединение с FTP-сервером.

        Raises:
            FTPConnectionError: Если не удалось подключиться.
        """
        try:
            self.logger.info(f"Подключение к FTP {self.host}...")
            self._ftp = ftplib.FTP()
            self._ftp.connect(self.host, timeout=self.timeout)
            self._ftp.login("anonymous", "")
            self._ftp.set_pasv(True)
            self.logger.info(f"Успешное подключение к {self.host}")
        except Exception as e:
            raise FTPConnectionError(
                f"Не удалось подключиться к {self.host}: {e}"
            ) from e

    def disconnect(self) -> None:
        """Закрывает соединение с FTP-сервером."""
        if self._ftp is not None:
            try:
                self._ftp.quit()
            except Exception:
                try:
                    self._ftp.close()
                except Exception:
                    pass
            self._ftp = None
            self.logger.debug("FTP-соединение закрыто")

    def _ensure_connected(self) -> ftplib.FTP:
        """Проверяет наличие соединения и возвращает FTP-объект.

        Returns:
            Активный экземпляр ftplib.FTP.

        Raises:
            FTPConnectionError: Если соединение не установлено.
        """
        if self._ftp is None:
            raise FTPConnectionError("FTP-соединение не установлено")
        return self._ftp

    # ------------------------------------------------------------------ #
    # Операции с файлами и директориями
    # ------------------------------------------------------------------ #

    def list_files(self, remote_path: str) -> List[str]:
        """Возвращает список файлов в указанной директории.

        Args:
            remote_path: Путь к директории на FTP.

        Returns:
            Список имён файлов.

        Raises:
            FTPConnectionError: Если соединение не установлено.
        """
        ftp = self._ensure_connected()
        try:
            lines: List[str] = []
            ftp.retrlines(f"LIST {remote_path}", lines.append)
            files = []
            for line in lines:
                parts = line.split(maxsplit=8)
                if len(parts) >= 9 and not parts[0].startswith("d"):
                    files.append(parts[8])
            return files
        except ftplib.error_perm as e:
            if "550" in str(e):
                raise FileNotFoundOnFTPError(
                    f"Директория не найдена: {remote_path}"
                ) from e
            raise

    def file_exists(self, remote_path: str) -> bool:
        """Проверяет существование файла на FTP.

        Args:
            remote_path: Полный путь к файлу на FTP.

        Returns:
            True, если файл существует.
        """
        ftp = self._ensure_connected()
        try:
            ftp.size(remote_path)
            return True
        except ftplib.error_perm:
            return False

    def file_size(self, remote_path: str) -> int:
        """Возвращает размер файла на FTP в байтах.

        Args:
            remote_path: Полный путь к файлу на FTP.

        Returns:
            Размер файла в байтах.

        Raises:
            FileNotFoundOnFTPError: Если файл не найден.
        """
        ftp = self._ensure_connected()
        try:
            size = ftp.size(remote_path)
            if size is None:
                raise FileNotFoundOnFTPError(f"Не удалось получить размер: {remote_path}")
            return int(size)
        except ftplib.error_perm as e:
            raise FileNotFoundOnFTPError(f"Файл не найден: {remote_path}") from e

    def get_checksum(self, remote_path: str) -> Optional[str]:
        """Получает MD5-сумму файла, если на FTP есть соответствующий файл.

        Ищет файлы CHECKSUM или MD5SUM рядом с целевым файлом.

        Args:
            remote_path: Полный путь к файлу на FTP.

        Returns:
            MD5-сумма в виде строки или None, если не удалось получить.
        """
        ftp = self._ensure_connected()
        # Пробуем разные имена файлов с контрольными суммами
        for suffix in (".md5", ".md5sum", ".MD5", ".MD5SUM"):
            try:
                checksum_path = remote_path + suffix
                buf = io.BytesIO()
                ftp.retrbinary(f"RETR {checksum_path}", buf.write)
                content = buf.getvalue().decode("utf-8", errors="ignore").strip()
                # Формат: "<md5>  filename" или просто "<md5>"
                parts = content.split()
                if parts:
                    return parts[0].lower()
            except (ftplib.error_perm, OSError):
                continue
        return None

    def list_releases(self, base_path: str = "/pub") -> List[int]:
        """Возвращает список доступных релизов Ensembl на FTP.

        Сканирует содержимое директории ``base_path`` (по умолчанию ``/pub``)
        и извлекает номера релизов из поддиректорий вида ``release-<N>``.

        Args:
            base_path: Базовая директория на FTP, в которой лежат релизы.

        Returns:
            Список целочисленных номеров релизов, отсортированный по возрастанию.

        Raises:
            FTPConnectionError: Если соединение не установлено.
            FileNotFoundOnFTPError: Если базовая директория не найдена.
        """
        import re

        ftp = self._ensure_connected()
        try:
            lines: List[str] = []
            ftp.retrlines(f"LIST {base_path}", lines.append)
        except ftplib.error_perm as e:
            if "550" in str(e):
                raise FileNotFoundOnFTPError(
                    f"Директория не найдена: {base_path}"
                ) from e
            raise

        releases: List[int] = []
        pattern = re.compile(r"^release-(\d+)$")
        for line in lines:
            parts = line.split(maxsplit=8)
            if len(parts) < 9 or not parts[0].startswith("d"):
                continue
            name = parts[8]
            m = pattern.match(name)
            if m:
                try:
                    releases.append(int(m.group(1)))
                except ValueError:
                    continue
        releases.sort()
        return releases

    def get_latest_release(self, base_path: str = "/pub") -> int:
        """Возвращает номер последнего (максимального) доступного релиза.

        Args:
            base_path: Базовая директория на FTP, в которой лежат релизы.

        Returns:
            Номер последнего релиза (int).

        Raises:
            FTPConnectionError: Если соединение не установлено.
            FileNotFoundOnFTPError: Если релизов не найдено.
        """
        releases = self.list_releases(base_path=base_path)
        if not releases:
            raise FileNotFoundOnFTPError(
                f"Не найдено ни одного релиза в {base_path}"
            )
        return releases[-1]

    # ------------------------------------------------------------------ #
    # Скачивание
    # ------------------------------------------------------------------ #

    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def download_file(
        self,
        remote_path: str,
        local_path: Path,
        callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Скачивает файл с FTP с поддержкой resume.

        Если локальный файл уже существует, скачивание продолжается
        с места обрыва (используется FTP-команда REST).

        Args:
            remote_path: Полный путь к файлу на FTP.
            local_path: Локальный путь для сохранения.
            callback: Опциональный колбэк (bytes_downloaded, total_bytes).

        Raises:
            FTPConnectionError: Если соединение не установлено.
            FileNotFoundOnFTPError: Если файл не найден на FTP.
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        ftp = self._ensure_connected()

        # Получаем размер удалённого файла
        try:
            total_size = self.file_size(remote_path)
        except FileNotFoundOnFTPError:
            raise

        # Определяем offset для resume
        offset = 0
        mode = "wb"
        if local_path.exists():
            offset = local_path.stat().st_size
            if offset >= total_size:
                self.logger.info(
                    f"Файл уже скачан: {local_path.name} ({offset} байт)"
                )
                if callback:
                    callback(offset, total_size)
                return
            mode = "ab"
            self.logger.info(
                f"Resume: {local_path.name} с offset {offset}/{total_size}"
            )

        # Используем REST для resume
        if offset > 0:
            ftp.sendcmd(f"REST {offset}")

        # Скачиваем с прогресс-баром
        with open(local_path, mode) as f:
            with tqdm(
                total=total_size,
                initial=offset,
                unit="B",
                unit_scale=True,
                desc=local_path.name[:40],
                disable=callback is None and not self.logger.isEnabledFor(logging.DEBUG),
            ) as pbar:
                def write_chunk(data: bytes) -> None:
                    f.write(data)
                    pbar.update(len(data))
                    if callback:
                        callback(offset + pbar.n, total_size)

                try:
                    ftp.retrbinary(f"RETR {remote_path}", write_chunk, blocksize=self.chunk_size)
                except ftplib.error_perm as e:
                    if "550" in str(e):
                        raise FileNotFoundOnFTPError(
                            f"Файл не найден: {remote_path}"
                        ) from e
                    raise

        self.logger.info(
            f"Скачан: {local_path.name} ({total_size} байт)"
        )

    def download_to_memory(self, remote_path: str) -> bytes:
        """Скачивает файл в оперативную память.

        Args:
            remote_path: Полный путь к файлу на FTP.

        Returns:
            Содержимое файла в виде bytes.
        """
        ftp = self._ensure_connected()
        buf = io.BytesIO()
        ftp.retrbinary(f"RETR {remote_path}", buf.write)
        return buf.getvalue()

    # ------------------------------------------------------------------ #
    # Контекстный менеджер
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "EnsemblFTPClient":
        """Вход в контекст — устанавливает соединение."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Выход из контекста — закрывает соединение."""
        self.disconnect()
