"""OrthoDB Downloader — заглушка.

Будет реализован в дальнейшем для скачивания данных OrthoDB
(https://www.orthodb.org/) для построения колонки паралогов/ортологов.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="orthodb-downloader",
        description="Заглушка OrthoDB Downloader (будет реализован)",
    )
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    args = parser.parse_args(argv)
    print("OrthoDB Downloader — заглушка.")
    print(f"Конфиг: {args.config}")
    print("Будет реализован в дальнейшем.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
