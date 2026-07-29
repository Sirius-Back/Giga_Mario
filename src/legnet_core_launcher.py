"""Run human_legnet ``core.py`` after restoring NumPy legacy aliases.

Lightning 2.0.x → deepdiff 7.x may touch ``np.float_`` / ``np.int_``, which
newer NumPy builds omit. Patch aliases *before* importing Lightning so LegNet
trains work without changing the shared Hydra pipeline or Caduceus stack.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def _patch_numpy_legacy_aliases() -> None:
    import numpy as np

    for modern, legacy in (
        ("float64", "float_"),
        ("int64", "int_"),
        ("bool", "bool_"),
        ("complex128", "complex_"),
    ):
        if not hasattr(np, legacy) and hasattr(np, modern):
            setattr(np, legacy, getattr(np, modern))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(
            "usage: python -m src.legnet_core_launcher <core.py> [core args...]",
            file=sys.stderr,
        )
        return 2
    core = Path(argv[0]).resolve()
    if not core.is_file():
        print(f"core.py missing: {core}", file=sys.stderr)
        return 2
    vendor = core.parent
    # human_legnet uses bare imports (``datamodule``, …); match subprocess cwd.
    os.chdir(vendor)
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    _patch_numpy_legacy_aliases()
    sys.argv = [str(core), *argv[1:]]
    runpy.run_path(str(core), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
