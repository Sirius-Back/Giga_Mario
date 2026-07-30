#!/usr/bin/env python3
"""Compile libpangenome_repeat_graph.so for the pangenome split strategy."""
from __future__ import annotations

from src.splits.pangenome_native import ensure_built, library_path


def main() -> int:
    path = ensure_built(force=True)
    print(path)
    assert library_path() == path
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
