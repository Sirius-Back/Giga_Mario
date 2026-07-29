"""DEPRECATED — accidental LegNet launcher for run8_2mer_caduceus.

Use ``pipeline_ready_caduceus`` instead (Caduceus on ``ready_caduceus``).
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    _ = argv
    print(
        "ERROR: pipeline_ready_legnet is deprecated for run8_2mer_caduceus.\n"
        "Launch: python -m src.runs.run8_2mer_caduceus.pipeline_ready_caduceus",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
