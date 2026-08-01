#!/usr/bin/env python3
"""Compile libparalogs_only.so for the paralogs_only split strategy."""
from __future__ import annotations

from src.splits.paralogs_only_native import ensure_built, library_path


def main() -> int:
    path = ensure_built(force=True)
    print(path)
    assert library_path() == path
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
