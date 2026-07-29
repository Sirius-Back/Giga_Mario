#!/usr/bin/env python3
"""Compile libkmer_count.so for the SBS k-mer backend."""
from __future__ import annotations

from src.splits.sbs.backends.native import ensure_built, library_path


def main() -> int:
    path = ensure_built(force=True)
    print(path)
    assert library_path() == path
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
